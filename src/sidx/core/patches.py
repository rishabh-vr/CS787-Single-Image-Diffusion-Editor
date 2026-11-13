"""Patch extraction and Gaussian-weighted reassembly (the paper's R_rho, §3.3).

A patch vector is flattened channel-major, (C, p, p) -> C*p*p, the same layout as the official
``extract_patches(...).flatten(1)``. Patches are ordered row-major over their top-left positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def extract_patches(image: torch.Tensor, patch_size: int, stride: int = 1) -> torch.Tensor:
    """(C, H, W) -> (N, C*p*p), N = ((H-p)//stride + 1) * ((W-p)//stride + 1)."""
    assert image.dim() == 3
    C, H, W = image.shape
    if min(H, W) < patch_size:
        raise ValueError(f"image {H}x{W} is smaller than patch_size={patch_size}")
    return F.unfold(image.unsqueeze(0), kernel_size=patch_size, stride=stride)[0].T.contiguous()


def gaussian_window(patch_size: int, rho: float, channels: int,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """Separable Gaussian over the patch's normalised [-1, 1] coordinates, std ``rho`` (notes Q1:
    1.4 px for p = 15, rho = 0.2). ``rho = inf`` gives uniform weights. Shape (C*p*p,)."""
    g = torch.linspace(-1, 1, patch_size, dtype=torch.float32)
    g = torch.ones_like(g) if math.isinf(rho) else torch.exp(-0.5 * (g / rho) ** 2)
    w2d = torch.outer(g, g)
    return w2d.repeat(channels, 1, 1).reshape(-1).to(device=device, dtype=dtype)


def fold_weighted(patches: torch.Tensor, window: torch.Tensor, hw, patch_size: int,
                  stride: int = 1) -> torch.Tensor:
    """Weighted average of overlapping patches: fold(patches * w) / fold(w). Returns (1, C, H, W).

    With stride > 1 some border pixels may be covered by no patch; they come back as 0.
    """
    H, W = int(hw[0]), int(hw[1])
    num = F.fold((patches * window).T.unsqueeze(0), (H, W), kernel_size=patch_size, stride=stride)
    den = F.fold(window.unsqueeze(1).expand(-1, patches.shape[0]).unsqueeze(0), (H, W),
                 kernel_size=patch_size, stride=stride)
    return torch.nan_to_num(num / den)
