import torch
from einops import rearrange
from torch import nn

from nd_mamba2 import Mamba2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        distance = torch.linalg.vector_norm(x0 - x1, dim=1)
        positive = y * distance.pow(2)
        negative = (1 - y) * torch.clamp(self.margin - distance, min=0).pow(2)
        return (positive + negative).mean() / 2


def initialize_mamba2(block):
    nn.init.uniform_(block.dt_bias, -4.0, -2.0)
    nn.init.zeros_(block.A_log)
    nn.init.ones_(block.D)


class NdMamba2_2d(nn.Module):
    def __init__(self, cin, mamba_dim, cout):
        super().__init__()
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)
        self.mamba2_forward = Mamba2(mamba_dim)
        self.mamba2_backward = Mamba2(mamba_dim)
        initialize_mamba2(self.mamba2_forward)
        initialize_mamba2(self.mamba2_backward)
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)

    def forward(self, x):
        height, width = x.shape[2:]
        x = nn.functional.pad(
            x,
            (0, (8 - width % 8) % 8, 0, (8 - height % 8) % 8),
        )
        padded_height = x.shape[2]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.fc_in(x)
        forward, _ = self.mamba2_forward(x)
        backward, _ = self.mamba2_backward(x.flip(1))
        x = self.fc_out(forward + backward.flip(1))
        x = rearrange(x, "b (h w) c -> b c h w", h=padded_height)
        return x[:, :, :height, :width]


class cnn_module(nn.Module):
    def __init__(self, kernel_size=7, dr=0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dr)
        self.ndmamba2_2d = NdMamba2_2d(128, 128, 128)
        self.fc1 = nn.Linear(4608, 512)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.ndmamba2_2d(x)
        return self.fc1(self.dropout(torch.flatten(x, 1)))


def build_model(kmer=6):
    if kmer != 6:
        raise ValueError("The released BMKMPHI architecture currently supports only k=6")
    return cnn_module(kernel_size=7, dr=0)
