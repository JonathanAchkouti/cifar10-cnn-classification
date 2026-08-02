import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleNet(nn.Module):
    """The original baseline: the classic PyTorch-tutorial LeNet-style CNN.

    Kept unchanged so it can be used as a control in the ablation table.
    Roughly 62k parameters, no normalization layers, no residual connections.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def conv_block(in_ch, out_ch, pool=False):
    """3x3 conv -> batch norm -> ReLU, optionally followed by 2x2 max pool.

    bias=False because the BatchNorm immediately after has its own shift
    parameter, which makes the conv bias redundant.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    """Two conv blocks whose output is added back to their input.

    The addition gives gradients a direct path back through the network,
    which is what makes deep networks trainable. The block learns a
    correction to its input rather than a whole new representation.
    """

    def __init__(self, channels):
        super().__init__()
        self.block1 = conv_block(channels, channels)
        self.block2 = conv_block(channels, channels)

    def forward(self, x):
        return x + self.block2(self.block1(x))


class ResNet9(nn.Module):
    """A residual CNN sized for 32x32 inputs.

    Note the stem: a single 3x3 stride-1 convolution with no max pool.
    The torchvision ResNets open with a 7x7 stride-2 conv followed by a
    stride-2 max pool, which is appropriate for 224x224 ImageNet images but
    discards 3/4 of a CIFAR image before any useful features are learned.

    Spatial size through the network: 32 -> 32 -> 16 -> 8 -> 4 -> 1.
    Roughly 6.6M parameters.
    """

    def __init__(self, num_classes=10, scale=0.125):
        super().__init__()
        self.stem = conv_block(3, 64)
        self.layer1 = conv_block(64, 128, pool=True)
        self.res1 = Residual(128)
        self.layer2 = conv_block(128, 256, pool=True)
        self.layer3 = conv_block(256, 512, pool=True)
        self.res2 = Residual(512)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        # Shrinking the logits keeps the softmax from saturating early, which
        # lets the high peak learning rate be used without the loss blowing up.
        self.scale = scale

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(self.layer1(x))
        x = self.layer2(x)
        x = self.res2(self.layer3(x))
        x = self.pool(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        return self.fc(x) * self.scale


MODELS = {
    'simplenet': SimpleNet,
    'resnet9': ResNet9,
}


def build_model(name, num_classes=10):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODELS)}")
    return MODELS[name](num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)