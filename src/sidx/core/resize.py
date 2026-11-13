"""Antialiased cubic resampling, written as one (out x in) matrix per spatial axis.

This is the same resampler the official code uses (its vendored ResizeRight, `resize_right.py`, via
GPNN's copy), re-expressed as dense matrices so it is short and easy to check:

* kernel: Keys cubic (a = -0.5), nominal width 6 (the official default; the cubic's support is 4,
  so the extra taps get zero weight);
* antialiasing when downscaling: the kernel is stretched by 1/scale and scaled by `scale`;
* sample positions: ResizeRight's centre-aligned mapping, with scale = out_len / in_len exactly
  (not 0.5 when the pyramid rounds sizes up);
* weights normalised to sum to 1 per output pixel;
* borders: symmetric (half-sample) mirroring of the input.

Agreement with the official resampler is tested in tests/test_core_equivalence.py (float32
rounding only).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch


def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((1 < ax) & (ax <= 2)))


@lru_cache(maxsize=256)
def _resize_matrix_np(in_len: int, out_len: int, antialias: bool, kernel_width: float) -> np.ndarray:
    if in_len == out_len:
        return np.eye(in_len)
    scale = out_len / in_len
    aa = antialias and scale < 1.0
    kernel = (lambda a: scale * _cubic(scale * a)) if aa else _cubic
    width = kernel_width / scale if aa else kernel_width

    out_coords = np.arange(1, out_len + 1)
    match = (out_coords - (out_len - in_len * scale) / 2) / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(match - width / 2)
    taps = int(np.ceil(width)) + 2
    fov = (left[:, None] + np.arange(taps) - 1).astype(np.int64)        # 0-based input indices
    w = kernel(match[:, None] - fov - 1)                                  # pixel j has centre j + 1
    s = w.sum(axis=1)
    s[s == 0] = 1.0
    w = w / s[:, None]

    mirror = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    idx = mirror[np.mod(fov, 2 * in_len)]
    m = np.zeros((out_len, in_len))
    np.add.at(m, (np.repeat(np.arange(out_len), taps), idx.ravel()), w.ravel())
    return m


def resize_matrix(in_len: int, out_len: int, antialias: bool = True, kernel_width: float = 6.0,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    """The (out_len, in_len) matrix that resamples one axis."""
    m = _resize_matrix_np(int(in_len), int(out_len), bool(antialias), float(kernel_width))
    return torch.as_tensor(m, dtype=dtype, device=device)


def resize(x: torch.Tensor, out_hw: Sequence[int], antialias: bool = True) -> torch.Tensor:
    """Resample the last two axes of ``x`` ([..., H, W]) to ``out_hw``.

    Computed in float32 and cast back to ``x.dtype``. The axis with the smaller scale factor is
    resampled first (as ResizeRight does), which only affects rounding.
    """
    H, W = x.shape[-2:]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    dtype = x.dtype
    y = x.float()
    order = sorted([(oh / H, -2), (ow / W, -1)], key=lambda t: t[0])
    for _, axis in order:
        n_in, n_out = (H, oh) if axis == -2 else (W, ow)
        if n_in == n_out:
            continue
        m = resize_matrix(n_in, n_out, antialias, device=y.device)
        if axis == -2:
            y = torch.einsum('oh,...hw->...ow', m, y)
        else:
            y = torch.einsum('ow,...hw->...ho', m, y)
    return y.to(dtype)
