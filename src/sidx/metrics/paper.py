"""The paper's Table-1 metrics, with the variants needed for calibration (notes §5.3).

SIFID         SinGAN's definition: InceptionV3 features at the first max-pool (64 channels,
              pytorch-fid block 0, input resized to 299 and normalised as pytorch-fid does);
              Fréchet distance between the spatial feature sets of the input and one sample.
pixel div.    per-pixel std across samples, mean over pixels and channels; `pixel_div` divides by
              the input's pixel std (SinGAN), `pixel_div_raw` does not (values in [0, 1]).
LPIPS div.    mean LPIPS over all sample pairs (AlexNet; VGG as a variant), inputs in [-1, 1].
NIQE/NIMA/MUSIQ  pyiqa defaults (optional dependency; NaN if pyiqa is missing).
near-dup      Table S1: a sample is a near-duplicate at level r if >= r% of its pixels have a
              mean-over-RGB absolute difference < 15/255 from the input.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sidx.core.resize import resize

#: Paper Table 1 (mean values), for side-by-side reports.
PAPER_TABLE1 = {
    "proposed (T=10, eta=0)": {"sifid": 0.29, "niqe": 8.08, "nima": 4.53, "musiq": 55.41,
                               "pixel_div": 0.15, "lpips_div": 0.49},
    "proposed (T=40, eta=1)": {"sifid": 0.21, "niqe": 8.18, "nima": 4.47, "musiq": 55.81,
                               "pixel_div": 0.13, "lpips_div": 0.39},
    "proposed (T=10, eta=0, k=5)": {"sifid": 0.38, "niqe": 8.10, "nima": 4.52, "musiq": 55.13,
                                    "pixel_div": 0.15, "lpips_div": 0.50},
    "SinDDM": {"sifid": 0.48, "niqe": 7.69, "nima": 4.30, "musiq": 50.74, "pixel_div": 0.10,
               "lpips_div": 0.36},
}


# --------------------------------------------------------------------------------------------
# SIFID
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _inception_block0(device: str = "cpu"):
    from pytorch_fid.inception import InceptionV3

    return InceptionV3([0]).eval().to(device)          # resize_input / normalize_input: defaults


@torch.no_grad()
def _sifid_features(img: torch.Tensor, device: str = "cpu") -> np.ndarray:
    feat = _inception_block0(device)(img[None].to(device))[0][0]           # (64, h, w)
    return feat.reshape(feat.shape[0], -1).T.double().cpu().numpy()         # (h*w, 64)


def _frechet(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (rows = samples).

    Own implementation: pytorch-fid's version calls scipy.linalg.sqrtm(..., disp=False), which
    current SciPy no longer accepts. Same formula and the same eps regularisation as pytorch-fid:
    |mu1-mu2|^2 + Tr(S1) + Tr(S2) - 2 Tr(sqrt(S1 S2)).
    """
    from scipy import linalg

    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def sifid(reference: torch.Tensor, samples: Sequence[torch.Tensor], device: str = "cpu") -> List[float]:
    """One SIFID value per sample (each sample vs the input)."""
    ref = _sifid_features(reference, device)
    return [_frechet(ref, _sifid_features(s, device)) for s in samples]


# --------------------------------------------------------------------------------------------
# diversity
# --------------------------------------------------------------------------------------------

def pixel_diversity(reference: torch.Tensor, samples: Sequence[torch.Tensor]) -> Dict[str, float]:
    x = torch.stack([s.float() for s in samples])           # (N, C, H, W)
    raw = float(x.std(0, unbiased=False).mean())
    return {"pixel_div": raw / float(reference.float().std(unbiased=False)), "pixel_div_raw": raw}


@lru_cache(maxsize=2)
def _lpips(net: str = "alex", device: str = "cpu"):
    import lpips

    return lpips.LPIPS(net=net, verbose=False).eval().to(device)


@torch.no_grad()
def lpips_diversity(samples: Sequence[torch.Tensor], net: str = "alex", max_pairs: int = 1225,
                    device: str = "cpu") -> float:
    pairs = list(itertools.combinations(range(len(samples)), 2))
    if not pairs:
        return float("nan")
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    model = _lpips(net, device)
    x = torch.stack([s.float() for s in samples]).to(device) * 2 - 1
    vals = []
    for k in range(0, len(pairs), 64):
        i, j = zip(*pairs[k:k + 64])
        vals.append(model(x[list(i)], x[list(j)]).flatten().cpu())
    return float(torch.cat(vals).mean())


# --------------------------------------------------------------------------------------------
# near-duplicates (Table S1)
# --------------------------------------------------------------------------------------------

def near_duplicate_fraction(reference: torch.Tensor, sample: torch.Tensor, tol: float = 15 / 255) -> float:
    """Fraction of pixels whose mean-over-RGB absolute difference from the input is < tol."""
    if sample.shape != reference.shape:
        sample = resize(sample, reference.shape[-2:])
    return float(((sample - reference).abs().mean(0) < tol).float().mean())


def near_duplicate_rate(fractions: Sequence[float], levels=(0.5, 0.7, 0.85, 0.95)) -> Dict[str, float]:
    f = np.asarray(fractions)
    return {f"near_dup_r{int(r * 100)}": float((f >= r).mean()) for r in levels}


# --------------------------------------------------------------------------------------------
# no-reference IQA (pyiqa, optional)
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _iqa(name: str, device: str = "cpu"):
    import pyiqa

    return pyiqa.create_metric(name, device=device)


@torch.no_grad()
def iqa_scores(img: torch.Tensor, names=("niqe", "nima", "musiq"), device: str = "cpu") -> Dict[str, float]:
    out = {}
    for n in names:
        try:
            out[n] = float(_iqa(n, device)(img[None].clamp(0, 1).to(device)))
        except Exception:                       # pyiqa missing, or a weight download failed
            out[n] = float("nan")
    return out
