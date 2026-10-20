# CIFAR-10 Classification — inmind Academy Assignment

Jonathan Achkouti

Starting from the provided baseline (a LeNet-style CNN reaching roughly 60% test
accuracy), this repository reaches **95.06% test accuracy** on CIFAR-10 with a
6.6M-parameter residual network trained from scratch in 23 minutes on a single
T4 GPU. No pretrained weights and no external data are used.

For reference, human accuracy on CIFAR-10 is estimated at around 94%.

## Results

| # | Run | Model | Params | Epochs | Aug | Val acc | Test acc | Time |
|---|-----|-------|--------|--------|-----|---------|----------|------|
| 0 | baseline (published) | SimpleNet | 62k | 10 | no | — | ~60% | — |
| 1 | `resnet9_full` | ResNet-9 | 6.6M | 30 | yes | 94.52% | **94.25%** | 13.8 min |
| 2 | `resnet9_50ep` | ResNet-9 | 6.6M | 50 | yes | 95.20% | **95.06%** | 23.3 min |

Run 0 is the accuracy documented for the original baseline configuration; it was
not re-run locally. Runs 1 and 2 were executed on a Colab T4.

The test set is evaluated exactly once per run, using the checkpoint from the
best validation epoch. No hyperparameter was selected by looking at test
accuracy.

## What changed from the baseline, and why

### Architecture: SimpleNet → ResNet-9

The largest single contributor. The baseline has 62k parameters, no
normalization layers and no residual connections, which caps it well below 70%
regardless of how carefully it is trained.

**Residual connections.** Each residual block computes `x + f(x)` rather than
`f(x)`. This gives gradients a direct path back through the network instead of
forcing them through every layer, and means each block only has to learn a
correction to its input rather than an entire new representation.

**Batch normalization.** Normalizing activations within each mini-batch keeps
them in a stable range as they pass through the network. This is what makes the
high peak learning rate below usable without the loss diverging.

**A CIFAR-appropriate stem.** `torchvision.models.resnet18` opens with a 7×7
stride-2 convolution followed by a stride-2 max pool. That is designed for
224×224 ImageNet images; applied to a 32×32 image it discards three quarters of
the input before any useful feature is learned. The stem here is a single 3×3
stride-1 convolution with no pooling. Spatial resolution through the network
runs 32 → 16 → 8 → 4 → 1.

### Data augmentation

`RandomCrop(32, padding=4)` and `RandomHorizontalFlip()`, plus `RandomErasing`
as a cutout equivalent. The model never sees the same image twice, which
prevents it from memorizing the training set.

Vertical flips are deliberately not used: upside-down cars, horses and ships do
not occur in this dataset, so the transform would create inputs unlike anything
at test time.

### One-cycle learning rate schedule

The learning rate warms up to 0.4 over the first quarter of training, then
anneals toward zero. The effect is clearly visible in the logs: during the
high-LR phase, validation accuracy oscillates between roughly 74% and 88% and
sometimes moves backwards, because the optimizer is taking large steps and
exploring rather than settling. Once the learning rate decays, accuracy climbs
monotonically — 91.6% → 92.7% → 93.6% → 94.3% → 95.2% over the final epochs.

The baseline's constant learning rate of 0.001 does neither: too small to
explore, never annealed to converge.

### Correct normalization statistics

The baseline normalized with a mean and standard deviation of 0.5 for every
channel, which is a placeholder rather than a measurement. This repository uses
the actual CIFAR-10 training-set statistics, mean `(0.4914, 0.4822, 0.4465)` and
standard deviation `(0.2470, 0.2435, 0.2616)`.

### Batch size 8 → 512

The baseline performs 5,625 optimizer steps per epoch, most of the time spent on
Python and kernel-launch overhead rather than computation. At batch 512 an epoch
is 87 steps and the GPU is actually saturated.

Learning rate was scaled with batch size — larger batches produce lower-variance
gradient estimates and therefore tolerate, and require, larger steps.

### Mixed precision (AMP)

Most operations run in 16-bit while the weights are kept in 32-bit, roughly
doubling throughput on the T4 with no measured accuracy cost.

### Correctness fixes

**Validation transform leak.** The baseline splits one dataset object with
`random_split`, so the resulting train and validation subsets share a single
transform. Adding augmentation would therefore have augmented the validation set
too, making the reported validation accuracy meaningless. This repository builds
two dataset objects with different transforms and indexes both with the same
seeded split.

**Best-checkpoint saving.** The baseline saves the model after the final epoch.
The final epoch is not reliably the best one, so the checkpoint is now written
whenever validation accuracy improves.

**Reproducibility.** Python, NumPy and PyTorch seeds are all set, and the
train/validation split uses a seeded generator. Without this, an accuracy
difference between two experiments cannot be attributed to the change made.

### Smaller contributions

- **Label smoothing (0.1)** — the target for the true class becomes 0.9 rather
  than 1.0, discouraging overconfidence. Note that this puts a floor under the
  cross-entropy loss, so loss values are not comparable to runs without it.
- **Nesterov momentum and weight decay (5e-4)** — standard regularization.
- **Test-time augmentation** — predictions for the image and its horizontal
  mirror are averaged. One extra forward pass, no extra training.

## Observations

**No overfitting at 50 epochs.** Final train loss 0.541 against validation loss
0.625. The gap barely widened from the 30-epoch run (0.565 / 0.625), indicating
the augmentation is doing its job and that longer training would still yield
small gains.

**Diminishing returns.** 30 → 50 epochs bought +0.81% for +9.5 minutes.
Extrapolating, further epochs were judged a poor use of the remaining time
compared to documenting the work.

**Honest validation.** Validation 95.20% against test 95.06%, a gap of 0.14%.
The split is representative and the test set was not leaked into any decision.

## Comparison: from-scratch ResNet-9 vs. pretrained Vision Transformer

As a follow-up experiment, the `vit/` folder fine-tunes an ImageNet-pretrained
Vision Transformer (ViT-B/16, `vit_base_patch16_224.augreg2_in21k_ft_in1k` from
`timm`) on the same CIFAR-10 splits, using the same training and evaluation
pipeline. The two approaches answer different questions: ResNet-9 shows how far
a small network trained from scratch can go, while the ViT shows what transfer
learning from millions of external images adds.

| | ResNet-9 (from scratch) | ViT-B/16 (fine-tuned) |
|---|---|---|
| Initialization | Random | Pretrained on ImageNet-21k, fine-tuned on ImageNet-1k |
| Parameters | 6.6M | ~86M |
| Input resolution | 32×32 (native) | 224×224 (bicubic upsampling) |
| Normalization | CIFAR-10 statistics | The checkpoint's own statistics, read from `timm` |
| Optimizer | SGD + Nesterov, peak LR 0.4 | AdamW, peak LR 1e-4 (4000× smaller) |
| Schedule | One-cycle, 30–50 epochs | One-cycle with a short 10% warmup, 3 epochs |
| Batch size | 512 | 64 (224×224 inputs are 49× more pixels) |
| Regularization | Augmentation + cutout, label smoothing, weight decay 5e-4 | Augmentation, label smoothing, weight decay 0.01 (no cutout: 3 epochs is too short to overfit) |
| External data | None | ImageNet-21k / ImageNet-1k |

Key design points in the ViT pipeline:

- **Matching the pretraining preprocessing.** `build_model` returns the
  normalization statistics and input size alongside the model, because a
  pretrained network must see inputs preprocessed the way it was during
  pretraining. Feeding it CIFAR-10 statistics would shift every input away from
  the distribution its features expect.
- **A much smaller learning rate.** The weights are already good; large steps
  would destroy the pretrained features rather than adapt them.
- **Same evaluation protocol.** Best-validation checkpointing, a single test
  evaluation and flip test-time augmentation, so the results are directly
  comparable with the ResNet-9 runs and land in the same `results.csv` format.

Running it:

```bash
pip install timm
cd vit
python train.py --config config_vit.yaml
```

## Reproducing

```bash
git clone https://github.com/JonathanAchkouti/cifar10-cnn-classification.git
cd cifar10-cnn-classification
uv sync                           # or: pip install -e .
python train.py                   # reads config.yaml
```

Every experiment is driven by `config.yaml`; no code changes are needed to
reproduce any row in the results table. Each run appends to `results.csv`.

Trains on CUDA, Apple Silicon (MPS) or CPU without modification — see
`get_device()` in `train.py`. Mixed precision and `pin_memory` are enabled only
on CUDA.

## Repository structure

```
config.yaml   All experiment settings
data.py       Transforms, CIFAR-10 statistics, train/val/test loaders
model.py      SimpleNet (original baseline) and ResNet-9
train.py      Training loop, evaluation, checkpointing, results logging
results.csv   One row per run, appended automatically
vit/          Pretrained ViT-B/16 fine-tuning (same pipeline, timm backbone)
```

## Possible next steps

- Longer training — the model is still slightly underfitting at 50 epochs.
- Mixup or CutMix, which typically add a few tenths of a percent.
- Combining the two: knowledge distillation from the fine-tuned ViT into
  the small ResNet-9.
