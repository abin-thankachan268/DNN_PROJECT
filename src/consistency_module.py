"""
GAN-style loss helpers.
"""
from __future__ import annotations

import torch


def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return (-d_fake).mean()
