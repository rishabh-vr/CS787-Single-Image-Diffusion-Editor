"""Verbatim copying (plan §8, G3): how much of a sample is a large block copied from the input.

At the last sampling step every output patch copies one input patch (provenance, paper S5), so a
copied *region* shows up as a set of neighbouring output patches whose nearest input patch sits at
the *same offset*. We compute the nearest-input-patch offset field (exact, 7x7 patches, at a
working resolution) and find connected regions of constant offset.

block_copy_ratio   fraction of output pixels inside constant-offset regions at least
                   `min_block` x `min_block` input pixels in area (default 48 x 48, plan §8)
mean_region_px     mean area of constant-offset regions (in input pixels): a continuous measure
                   of how big the copied pieces are
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from sidx.core.resize import resize


def copy_regions(reference: torch.Tensor, sample: torch.Tensor, p: int = 7, long_side: int = 128,
                 min_block: int = 48, offset_tol: int = 1) -> Dict[str, float]:
    from scipy import ndimage

    H, W = reference.shape[-2:]
    s = min(1.0, long_side / max(H, W))
    hw = (max(p, round(H * s)), max(p, round(W * s)))
    ref = resize(reference.float(), hw) if s < 1 else reference.float()
    smp = resize(sample.float(), hw) if sample.shape[-2:] != hw else sample.float()
    nh, nw = hw[0] - p + 1, hw[1] - p + 1
    a = F.unfold(ref[None], p)[0].T
    b = F.unfold(smp[None], p)[0].T
    idx = torch.cdist(b, a).argmin(1).numpy()
    sy, sx = idx // nw, idx % nw                                    # source position per output patch
    oy, ox = np.mgrid[0:nh, 0:nw]
    dy = (sy.reshape(nh, nw) - oy)
    dx = (sx.reshape(nh, nw) - ox)
    # regions: 4-connected output positions whose offsets agree with a neighbour within offset_tol
    same_r = (np.abs(np.diff(dy, axis=1)) <= offset_tol) & (np.abs(np.diff(dx, axis=1)) <= offset_tol)
    same_d = (np.abs(np.diff(dy, axis=0)) <= offset_tol) & (np.abs(np.diff(dx, axis=0)) <= offset_tol)
    # union-find via labelling a graph: build an adjacency image at 2x resolution
    g = np.zeros((2 * nh - 1, 2 * nw - 1), dtype=bool)
    g[::2, ::2] = True
    g[::2, 1::2] = same_r
    g[1::2, ::2] = same_d
    labels, n = ndimage.label(g)
    lab = labels[::2, ::2]
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    min_area = (min_block * s) ** 2
    big = np.isin(lab, np.nonzero(sizes >= min_area)[0] + 1)
    px_per_patch = (1 / s) ** 2
    return {"block_copy_ratio": float(big.mean()),
            "mean_region_px": float(sizes.mean() * px_per_patch) if sizes.size else 0.0}
