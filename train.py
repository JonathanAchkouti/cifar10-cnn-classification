import argparse
import csv
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from data import get_loaders
from model import build_model, count_parameters


def set_seed(seed):
    """Makes runs repeatable, so an accuracy difference between two
    experiments reflects the change made and not the random initialisation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')      # Apple Silicon GPU
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if tta:
            # Test-time augmentation: average the predictions for the image
            # and its mirror. Costs one extra forward pass, no extra training.
            outputs = (outputs + model(torch.flip(images, dims=[3]))) / 2
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100 * correct / total


def train(model, train_loader, val_loader, criterion, optimizer,
          scheduler, scaler, device, config):
    epochs = config['hyperparameters']['epochs']
    ckpt_path = config['paths']['model_path']
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    use_amp = device.type == 'cuda'
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for inputs, labels in bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Mixed precision: most operations run in 16-bit, which is roughly
            # twice as fast on a modern GPU, while the weights stay 32-bit.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()                    # one-cycle steps per batch

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            bar.set_postfix({'loss': running_loss / seen,
                             'lr': scheduler.get_last_lr()[0]})

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Keep the weights from the best epoch, not the last one. The final
        # epoch is not reliably the best, and this costs nothing.
        marker = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            marker = '  <- best'

        print(f'Epoch {epoch + 1}: train loss {train_loss:.3f} | '
              f'val loss {val_loss:.3f} | val acc {val_acc:.2f}%{marker}')

    print(f'Training finished. Best validation accuracy: {best_val_acc:.2f}%')
    return best_val_acc


def log_result(config, params, val_acc, test_acc, minutes):
    """Appends one row per run so the ablation table builds itself."""
    path = config['paths']['results_path']
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'run_name': config['run_name'],
        'model': config['model']['name'],
        'params': params,
        'epochs': config['hyperparameters']['epochs'],
        'batch_size': config['hyperparameters']['batch_size'],
        'max_lr': config['hyperparameters']['max_lr'],
        'augment': config['data']['augment'],
        'cutout': config['data']['cutout'],
        'label_smoothing': config['hyperparameters']['label_smoothing'],
        'tta': config['eval']['tta'],
        'val_acc': round(val_acc, 2),
        'test_acc': round(test_acc, 2),
        'minutes': round(minutes, 1),
    }
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f'Logged to {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config['hyperparameters']['seed'])
    device = get_device()
    print(f"Run '{config['run_name']}' on {device}")

    train_loader, val_loader, test_loader = get_loaders(config, device)

    model = build_model(config['model']['name']).to(device)
    print(f"Model: {config['model']['name']} "
          f'({count_parameters(model):,} trainable parameters)')

    h = config['hyperparameters']
    # Label smoothing: the target becomes 0.9 for the true class instead of
    # 1.0, which discourages overconfidence and generalizes slightly better.
    criterion = nn.CrossEntropyLoss(label_smoothing=h['label_smoothing'])
    optimizer = optim.SGD(model.parameters(), lr=h['max_lr'],
                          momentum=h['momentum'], nesterov=True,
                          weight_decay=h['weight_decay'])
    # One-cycle: warm the learning rate up, then anneal it to almost zero.
    # The warmup avoids early instability, the anneal lets the model settle.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=h['max_lr'],
        steps_per_epoch=len(train_loader), epochs=h['epochs'],
        pct_start=0.25, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    start = time.time()
    val_acc = train(model, train_loader, val_loader, criterion,
                    optimizer, scheduler, scaler, device, config)
    minutes = (time.time() - start) / 60

    # Reload the best checkpoint before touching the test set, which is
    # evaluated exactly once.
    model.load_state_dict(torch.load(config['paths']['model_path'],
                                     map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                   tta=config['eval']['tta'])
    print(f'\nTEST: loss {test_loss:.3f} | accuracy {test_acc:.2f}% '
          f'| {minutes:.1f} min')

    log_result(config, count_parameters(model), val_acc, test_acc, minutes)


if __name__ == '__main__':
    main()