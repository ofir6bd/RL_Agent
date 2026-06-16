"""3-D CNN encoder: (C, G, G, G) -> (latent_dim,)."""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNEncoder3D(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),           # 16
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),           #  8
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),          #  4
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        convolution_output = self.conv(state).flatten(1)
        return self.proj(convolution_output)
