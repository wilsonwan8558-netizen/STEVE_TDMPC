"""Compact nominal-N force head attached to frozen TD-MPC2 latent/action pairs."""

from __future__ import annotations

import torch
from torch import nn


class ForceHead(nn.Module):
    """Predict nonnegative next-interval max distal contact force, in nominal N."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.shape != (258,) or std.shape != (258,) or torch.any(std <= 0):
            raise ValueError("G12 force-head normalization must have 258 positive-scale features")
        self.register_buffer("input_mean", mean.detach().clone().float())
        self.register_buffer("input_std", std.detach().clone().float())
        self.network = nn.Sequential(
            nn.Linear(258, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus(),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if latent.shape[:-1] != action.shape[:-1] or latent.shape[-1] != 256 or action.shape[-1] != 2:
            raise ValueError("G12 force-head input must be aligned 256-D latent and 2-D action")
        features = torch.cat((latent, action), dim=-1)
        return self.network((features - self.input_mean) / self.input_std)
