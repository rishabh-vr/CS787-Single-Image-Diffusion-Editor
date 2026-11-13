"""Per-image complexity descriptors (plan §4.3).

They describe *how* structured an image is, so failures can later be regressed on image properties
(plan F1), and natural images give the in-distribution reference values. All are computed on the
luminance image in [0, 1], deterministically, on CPU.

    edge_density         fraction of Canny edge pixels (sigma = 1)
    ink_fraction         fraction of pixels in the minority Otsu class ("ink" for line art)
    line_width_px        median stroke width of the ink: 2 x distance-to-background at skeleton
                         pixels (meaningful for line art; reported for every image)
    periodicity_score    lattice-ness of the high-passed image (minus a sigma-2 Gaussian blur): the
                         strongest autocorrelation at a vertical shift >= 4 px and the strongest at a
                         horizontal shift >= 4 px, the smaller of the two, relative to the zero-lag
                         value; in [0, 1] (1 = periodic along both axes)
    patch_code_entropy   normalised Shannon entropy of 3x3 census codes (8-bit local-structure
                         codes): local structural variety, in [0, 1]
    rare_patch_fraction  on the image downscaled to <= 64 px, the fraction of 7x7 patches whose
                         nearest non-overlapping neighbour is > 2x the median such distance:
                         how much of the image is made of patches that occur only once
                         (the rare-detail risk of patch recombination, plan F2)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def luminance(img_chw01: torch.Tensor) -> np.ndarray:
    x = img_chw01.float()
    if x.shape[0] == 1:
        return x[0].numpy()
    return (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).numpy()


def edge_density(gray: np.ndarray) -> float:
    from skimage.feature import canny

    return float(canny(gray, sigma=1.0).mean())


def ink_mask(gray: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_otsu

    if gray.max() - gray.min() < 1e-6:
        return np.zeros_like(gray, dtype=bool)
    dark = gray < threshold_otsu(gray)
    return dark if dark.mean() <= 0.5 else ~dark


def line_width_px(ink: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if ink.sum() == 0:
        return 0.0
    skel = skeletonize(ink)
    if skel.sum() == 0:
        return 0.0
    return float(np.median(2.0 * distance_transform_edt(ink)[skel]))


def periodicity_score(gray: np.ndarray, min_lag: int = 4, highpass_sigma: float = 2.0) -> float:
    from scipy.ndimage import gaussian_filter

    # high-pass first: smooth gradients (skies) correlate with themselves at any small shift and
    # would otherwise look "periodic"; only repeating structure should count
    g = gray - gaussian_filter(gray, highpass_sigma, mode="reflect")
    g = g - g.mean()
    if np.allclose(g, 0, atol=1e-7):
        return 0.0
    H, W = g.shape
    f = np.fft.rfft2(g, s=(2 * H, 2 * W))                    # zero-padded: linear autocorrelation
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(f) ** 2, s=(2 * H, 2 * W)))   # zero lag at [H, W]
    yy, xx = np.meshgrid(np.arange(2 * H) - H, np.arange(2 * W) - W, indexing="ij")
    # per-lag mean over the overlap, so long lags are not penalised for overlapping less
    overlap = np.clip(H - np.abs(yy), 0, None) * np.clip(W - np.abs(xx), 0, None)
    ac = ac / np.maximum(overlap, 1)
    window = (np.abs(yy) <= H // 3) & (np.abs(xx) <= W // 3)
    # repetition must exist along BOTH axes (a lattice); a single straight line repeats under any
    # shift along itself and would otherwise score as periodic
    vert = ac[window & (np.abs(yy) >= min_lag)].max()
    horiz = ac[window & (np.abs(xx) >= min_lag)].max()
    return float(np.clip(min(vert, horiz) / ac[H, W], 0.0, 1.0))


def patch_code_entropy(gray: np.ndarray) -> float:
    g = torch.from_numpy(gray)[None, None].float()
    patches = F.unfold(g, 3)[0]                               # (9, L)
    centre = patches[4]
    neigh = torch.cat([patches[:4], patches[5:]])             # 8 neighbours
    codes = ((neigh > centre).long() * (2 ** torch.arange(8))[:, None]).sum(0)
    p = torch.bincount(codes, minlength=256).float()
    p = p[p > 0] / p.sum()
    return max(0.0, float(-(p * p.log2()).sum() / 8.0))


def rare_patch_fraction(gray: np.ndarray, max_side: int = 64, p: int = 7) -> float:
    from sidx.core.resize import resize

    g = torch.from_numpy(gray)[None].float()
    H, W = g.shape[-2:]
    s = min(1.0, max_side / max(H, W))
    if s < 1.0:
        g = resize(g, (max(p, round(H * s)), max(p, round(W * s))))
    h, w = g.shape[-2:]
    patches = F.unfold(g[None], p)[0].T                       # (N, p*p)
    nh, nw = h - p + 1, w - p + 1
    d = torch.cdist(patches, patches, compute_mode="donot_use_mm_for_euclid_dist")  # exact zeros
    yy, xx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
    yy, xx = yy.reshape(-1), xx.reshape(-1)
    overlapping = ((yy[:, None] - yy[None]).abs() < p) & ((xx[:, None] - xx[None]).abs() < p)
    d[overlapping] = float("inf")
    nn = d.min(1).values
    nn = nn[torch.isfinite(nn)]
    if nn.numel() == 0:
        return float("nan")
    med = nn.median()
    if med <= 1e-6:
        return float((nn > 1e-6).float().mean())
    return float((nn > 2 * med).float().mean())


def compute_descriptors(img_chw01: torch.Tensor) -> Dict[str, float]:
    gray = luminance(img_chw01)
    ink = ink_mask(gray)
    return {
        "edge_density": round(edge_density(gray), 5),
        "ink_fraction": round(float(ink.mean()), 5),
        "line_width_px": round(line_width_px(ink), 3),
        "periodicity_score": round(periodicity_score(gray), 5),
        "patch_code_entropy": round(patch_code_entropy(gray), 5),
        "rare_patch_fraction": round(rare_patch_fraction(gray), 5),
    }
