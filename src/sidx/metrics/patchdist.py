"""Domain-agnostic patch-distribution metrics (plan §8, G1).

patch_swd        sliced Wasserstein distance between the input's and a sample's raw RGB patch sets,
                 averaged over patch sizes and pyramid scales. Same projections for every image.
patch_precision  fraction of a sample's patches whose nearest input patch is closer than tau:
                 "no invented content". tau = 95th percentile of the input's own nearest
                 *non-overlapping* neighbour distances (so tau adapts to how repetitive the input is).
patch_recall     over all samples of an image: fraction of the input's patches that are the nearest
                 neighbour of at least one sample patch: "no dropped details".

Precision/recall run at a working resolution (long side <= 128 px, 7x7 patches) to stay cheap.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from sidx.core.resize import resize


def _patches(img: torch.Tensor, p: int, stride: int = 1) -> torch.Tensor:
    return F.unfold(img[None].float(), p, stride=stride)[0].T            # (N, C*p*p)


def _working(img: torch.Tensor, long_side: int) -> torch.Tensor:
    H, W = img.shape[-2:]
    s = min(1.0, long_side / max(H, W))
    return img if s >= 1 else resize(img, (max(8, round(H * s)), max(8, round(W * s))))


def patch_swd(reference: torch.Tensor, sample: torch.Tensor, patch_sizes=(7, 15), num_scales: int = 3,
              n_proj: int = 128, max_patches: int = 8000, seed: int = 0) -> float:
    vals = []
    g = torch.Generator().manual_seed(seed)
    ref, smp = reference.float(), sample.float()
    for s in range(num_scales):
        if s:
            ref = resize(ref, ((ref.shape[-2] + 1) // 2, (ref.shape[-1] + 1) // 2))
            smp = resize(smp, ((smp.shape[-2] + 1) // 2, (smp.shape[-1] + 1) // 2))
        for p in patch_sizes:
            if min(ref.shape[-2:]) < p or min(smp.shape[-2:]) < p:
                continue
            a, b = _patches(ref, p), _patches(smp, p)
            proj = torch.randn(a.shape[1], n_proj, generator=g)
            proj = proj / proj.norm(dim=0, keepdim=True)
            pa, pb = a @ proj, b @ proj
            # compare quantile functions (handles different set sizes)
            q = torch.linspace(0, 1, min(max_patches, pa.shape[0], pb.shape[0]))
            qa, qb = torch.quantile(pa, q, dim=0), torch.quantile(pb, q, dim=0)
            vals.append(float((qa - qb).abs().mean()))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _nn_dist_excluding_overlap(x: torch.Tensor, nh: int, nw: int, p: int) -> torch.Tensor:
    d = torch.cdist(x, x)
    yy, xx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
    yy, xx = yy.reshape(-1), xx.reshape(-1)
    ov = ((yy[:, None] - yy[None]).abs() < p) & ((xx[:, None] - xx[None]).abs() < p)
    d[ov] = float("inf")
    return d.min(1).values


class PatchPR:
    """Precision per sample, recall pooled over the samples of one input."""

    def __init__(self, reference: torch.Tensor, p: int = 7, long_side: int = 128, quantile: float = 0.95):
        self.p, self.long_side = p, long_side
        ref = _working(reference, long_side)
        self.ref_hw = ref.shape[-2:]
        self.ref = _patches(ref, p)
        nh, nw = ref.shape[-2] - p + 1, ref.shape[-1] - p + 1
        nn = _nn_dist_excluding_overlap(self.ref, nh, nw, p)
        nn = nn[torch.isfinite(nn)]
        self.tau = float(torch.quantile(nn, quantile)) if nn.numel() else 0.0
        self.hit = torch.zeros(self.ref.shape[0], dtype=torch.bool)

    def add(self, sample: torch.Tensor) -> float:
        smp = resize(sample.float(), self.ref_hw) if sample.shape[-2:] != self.ref_hw else sample.float()
        q = _patches(smp, self.p)
        d = torch.cdist(q, self.ref)
        best, idx = d.min(1)
        self.hit[idx] = True
        return float((best <= self.tau + 1e-6).float().mean())

    def recall(self) -> float:
        return float(self.hit.float().mean())
