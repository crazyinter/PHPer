"""Model and representation variants for ablation."""
import torch
from torch import nn

from model_bmk import NdMamba2_2d


class CNNOnly(nn.Module):
    """Same CNN trunk as BMKMPHI but without the Nd-BiMamba2 block."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.maxpool(x)
        return self.fc1(torch.flatten(x, 1))


class PhageCGRCNN(nn.Module):
    """PhageCGRNet CNN for 3-channel 128x128 CGR, parameterized class count."""

    def __init__(self, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.s1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2, 2)
        self.f1 = nn.Linear(128 * 32 * 32, 512)
        self.f2 = nn.Linear(512, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.s1(self.relu(self.c1(x)))
        x = self.s2(self.relu(self.c2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.f1(x))
        return self.f2(x)


def build_variant(encoder):
    if encoder == "cnn":
        return CNNOnly(7)
    if encoder == "cnn_mamba":
        from model_bmk import build_model
        return build_model(6)
    raise ValueError(encoder)


class ProteinOnlyNet(nn.Module):
    """Embedding from protein-level features alone."""

    def __init__(self, prot_dim=96, emb=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prot_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, emb))

    def forward(self, prot):
        return self.net(prot)


class FusionModel(nn.Module):
    """Late fusion of a sequence (FCGR) encoder and a protein MLP."""

    def __init__(self, seq_encoder, prot_dim=96, emb=512):
        super().__init__()
        self.seq = seq_encoder
        self.prot = nn.Sequential(
            nn.Linear(prot_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(512 + 128, emb), nn.ReLU())

    def forward(self, fcgr, prot):
        s = self.seq(fcgr)
        p = self.prot(prot)
        return self.fuse(torch.cat([s, p], dim=1))


def permute_representation(matrices, seed):
    """Randomly permute the 4096 k-mer frequencies of each sample.

    Same permutation is applied to all samples so the mapping is consistent;
    identical frequency content, no FCGR topology.
    """
    n = matrices.shape[0]
    flat = matrices.reshape(n, -1)
    g = torch.gen.Generator().manual_seed(seed) if False else None
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.shape[1])
    out = flat[:, perm]
    return out.reshape(matrices.shape)
