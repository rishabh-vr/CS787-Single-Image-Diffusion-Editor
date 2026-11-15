"""Structural validity for line art and patterns (plan §8, G2). Reference-free in the sense that a
sample is compared with the input's *statistics*, never pixel by pixel.

All on luminance in [0, 1]. "Ink" = minority Otsu class, with the threshold taken from the INPUT
and reused for its samples, so a sample cannot change its own binarisation.

endpoint_excess      (skeleton endpoints per 1000 skeleton px)_sample / (same)_input  — >1 means
                     broken lines / dangling stubs
junction_hist_l1     L1 distance between junction-degree histograms (3, 4, >=5 branches)
junctions_per_kpx    sample's junctions per 1000 skeleton px (input value reported separately)
seg_len_w1_rel       Wasserstein-1 distance between LSD segment-length distributions, divided by the
                     input's mean segment length (scale-free)
seg_angle_w1_deg     Wasserstein-1 distance between segment-orientation distributions (degrees,
                     length-weighted, on the half-circle)
mean_seg_len_ratio   mean LSD segment length, sample / input — <1 means fragmentation
grey_ratio           fraction of "intermediate grey" pixels (0.2 < I < 0.8), sample / input — >1
                     means ghosting / blur from averaging misaligned patches
edge_density_ratio   Canny edge density, sample / input
grad_mag_ratio       mean gradient magnitude, sample / input (sharpness)
periodicity_ratio    lattice-ness (sidx.data.descriptors.periodicity_score), sample / input
spectral_peak_recall fraction of the input's 8 strongest non-DC power-spectrum peaks that are
                     among the sample's 16 strongest (within +-1 frequency bin) — lattice phase /
                     period preserved
orientation_conformity  (synthetic only) fraction of segment length within +-2 deg of the allowed
                     generator orientations
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from sidx.data.descriptors import luminance, periodicity_score


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return float("nan") if a == 0 else float("inf")
    return a / b


def ink_threshold(gray_ref: np.ndarray) -> tuple:
    from skimage.filters import threshold_otsu

    t = threshold_otsu(gray_ref) if gray_ref.max() - gray_ref.min() > 1e-6 else 0.5
    dark_is_ink = (gray_ref < t).mean() <= 0.5
    return t, dark_is_ink


def ink(gray: np.ndarray, thr: tuple) -> np.ndarray:
    t, dark_is_ink = thr
    return gray < t if dark_is_ink else gray >= t


# --- skeleton topology --------------------------------------------------------------------------

def skeleton_stats(mask: np.ndarray) -> Dict[str, float]:
    from scipy import ndimage
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    n = int(skel.sum())
    if n == 0:
        return {"skeleton_px": 0, "endpoints_per_kpx": 0.0, "junctions_per_kpx": 0.0,
                "jdeg3": 0.0, "jdeg4": 0.0, "jdeg5p": 0.0}
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    nb = ndimage.convolve(skel.astype(int), k, mode="constant") * skel
    endpoints = int((nb == 1).sum())
    jmask = nb >= 3
    labels, nj = ndimage.label(jmask, structure=np.ones((3, 3)))
    degs = []
    for j in range(1, nj + 1):
        # branches leaving a junction cluster = skeleton pixels adjacent to it but not in it
        cl = labels == j
        ring = ndimage.binary_dilation(cl, structure=np.ones((3, 3))) & skel & ~cl
        rl, nb_ = ndimage.label(ring, structure=np.ones((3, 3)))
        degs.append(nb_)
    degs = np.array(degs) if degs else np.zeros(0)
    tot = max(1, len(degs))
    return {"skeleton_px": n, "endpoints_per_kpx": 1000 * endpoints / n,
            "junctions_per_kpx": 1000 * nj / n,
            "jdeg3": float((degs == 3).sum() / tot), "jdeg4": float((degs == 4).sum() / tot),
            "jdeg5p": float((degs >= 5).sum() / tot)}


# --- line segments ------------------------------------------------------------------------------

def line_segments(gray: np.ndarray, min_len: float = 4.0) -> np.ndarray:
    """LSD segments as (length, angle_deg in [0, 180)). Fixed OpenCV LSD parameters."""
    import cv2

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect((np.clip(gray, 0, 1) * 255).astype(np.uint8))[0]
    if lines is None:
        return np.zeros((0, 2))
    l = lines.reshape(-1, 4)
    dx, dy = l[:, 2] - l[:, 0], l[:, 3] - l[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    keep = length >= min_len
    return np.stack([length[keep], ang[keep]], 1)


def _w1(a: np.ndarray, b: np.ndarray, wa=None, wb=None) -> float:
    from scipy.stats import wasserstein_distance

    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b, wa, wb))


def _angle_w1(a, b, wa, wb) -> float:
    """W1 on the half-circle [0, 180): minimum over rotations of the cut point (approximation by
    trying 12 cut points)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return min(_w1((a + c) % 180, (b + c) % 180, wa, wb) for c in range(0, 180, 15))


# --- spectrum -----------------------------------------------------------------------------------

def _spectrum_peaks(gray: np.ndarray, k: int) -> np.ndarray:
    g = gray - gray.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    H, W = P.shape
    cy, cx = H // 2, W // 2
    P[cy - 1:cy + 2, cx - 1:cx + 2] = 0                       # DC
    P[:, :cx] = 0                                             # half-plane (spectrum is symmetric)
    flat = np.argsort(P.ravel())[::-1][:k]
    return np.stack([flat // W - cy, flat % W - cx], 1)


def spectral_peak_recall(gray_ref: np.ndarray, gray_smp: np.ndarray, k_ref: int = 8, k_smp: int = 16) -> float:
    if gray_ref.shape != gray_smp.shape:
        return float("nan")
    a, b = _spectrum_peaks(gray_ref, k_ref), _spectrum_peaks(gray_smp, k_smp)
    hit = (np.abs(a[:, None, :] - b[None]).max(-1) <= 1).any(1)
    return float(hit.mean())


# --- per-input reference cache + per-sample metrics ----------------------------------------------

class StructureReference:
    def __init__(self, reference: torch.Tensor):
        from skimage.feature import canny

        self.gray = luminance(reference)
        self.thr = ink_threshold(self.gray)
        self.skel = skeleton_stats(ink(self.gray, self.thr))
        self.segs = line_segments(self.gray)
        self.grey = float(((self.gray > 0.2) & (self.gray < 0.8)).mean())
        self.edge = float(canny(self.gray, sigma=1.0).mean())
        gy, gx = np.gradient(self.gray)
        self.grad = float(np.hypot(gx, gy).mean())
        self.period = periodicity_score(self.gray)

    def info(self) -> Dict[str, float]:
        return {"ref_" + k: v for k, v in {**self.skel, "grey": self.grey, "edge_density": self.edge,
                                           "periodicity": self.period,
                                           "num_segments": len(self.segs)}.items()}

    def compare(self, sample: torch.Tensor, allowed_angles: Optional[Sequence[float]] = None) -> Dict[str, float]:
        from skimage.feature import canny

        g = luminance(sample)
        sk = skeleton_stats(ink(g, self.thr))
        segs = line_segments(g)
        r = self.segs
        out = {
            "endpoint_excess": _safe_ratio(sk["endpoints_per_kpx"], self.skel["endpoints_per_kpx"]),
            "junction_hist_l1": float(sum(abs(sk[k] - self.skel[k]) for k in ("jdeg3", "jdeg4", "jdeg5p"))),
            "junctions_per_kpx": sk["junctions_per_kpx"],
            "seg_len_w1_rel": _safe_ratio(_w1(segs[:, 0], r[:, 0]), float(r[:, 0].mean()) if len(r) else 0.0),
            "seg_angle_w1_deg": _angle_w1(segs[:, 1], r[:, 1], segs[:, 0], r[:, 0]) if len(segs) and len(r) else float("nan"),
            "mean_seg_len_ratio": _safe_ratio(float(segs[:, 0].mean()) if len(segs) else 0.0,
                                              float(r[:, 0].mean()) if len(r) else 0.0),
            "grey_ratio": _safe_ratio(float(((g > 0.2) & (g < 0.8)).mean()), self.grey),
            "edge_density_ratio": _safe_ratio(float(canny(g, sigma=1.0).mean()), self.edge),
            "grad_mag_ratio": _safe_ratio(float(np.hypot(*np.gradient(g)).mean()), self.grad),
            "periodicity_ratio": _safe_ratio(periodicity_score(g), self.period),
            "spectral_peak_recall": spectral_peak_recall(self.gray, g),
        }
        if allowed_angles is not None and len(segs):
            d = np.abs((segs[:, 1][:, None] - np.asarray(allowed_angles)[None] + 90) % 180 - 90).min(1)
            out["orientation_conformity"] = float(segs[d <= 2.0, 0].sum() / segs[:, 0].sum())
        return out
