import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel mean and standard deviation of the CIFAR-10 training set,
# computed over all 50,000 images. The baseline used 0.5 for every value,
# which is a placeholder rather than a measurement.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def build_transforms(augment=False, cutout=False):
    """Returns (train_transform, eval_transform).

    Evaluation must never be augmented: augmentation exists to make the
    training task harder so the model generalizes, but a randomly cropped
    test image would just make the reported accuracy wrong.
    """
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if not augment:
        return eval_tf, eval_tf

    train_steps = [
        # Pad by 4 pixels on every side, then crop a random 32x32 window.
        # Teaches tolerance to the object not being perfectly centred.
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        # Horizontal flip only. A vertical flip would produce upside-down
        # cars and horses, which never occur in this dataset.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    if cutout:
        # Blanks out a random rectangle, forcing the model to rely on several
        # parts of the object rather than one distinctive patch.
        train_steps.append(
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.15), value='random')
        )

    return transforms.Compose(train_steps), eval_tf


def get_loaders(config, device):
    d, h = config['data'], config['hyperparameters']
    train_tf, eval_tf = build_transforms(d['augment'], d['cutout'])

    # Two dataset objects over the same files, carrying different transforms.
    # A single dataset split with random_split would share one transform
    # between train and validation, silently augmenting the validation set.
    train_full = datasets.CIFAR10(root=d['root'], train=True,
                                  download=True, transform=train_tf)
    val_full = datasets.CIFAR10(root=d['root'], train=True,
                                download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root=d['root'], train=False,
                                download=True, transform=eval_tf)

    # Seeded split so every experiment sees exactly the same images.
    generator = torch.Generator().manual_seed(h['seed'])
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    n_val = int(len(train_full) * d['val_split'])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    cuda = device.type == 'cuda'
    common = dict(
        batch_size=h['batch_size'],
        num_workers=d['num_workers'],
        pin_memory=cuda,                       # CUDA-only optimisation
        persistent_workers=d['num_workers'] > 0,
    )

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    print(f'Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}')
    return train_loader, val_loader, test_loader