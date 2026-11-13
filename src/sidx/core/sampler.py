"""Coarse-to-fine unconditional sampling (paper Alg. 1 + Alg. 2), our implementation.

Per scale s, from the coarsest to the finest, starting from pure noise:

    for t = T .. 1:
        x_hat = fold_rho( D( unfold(x_t) ) )                    # closed-form patch denoiser
        eps   = (x_t - alpha_t x_hat) / sigma_t                 # before the blend
        if s is not the coarsest and t != 1:
            x_hat = x_hat - up(down(x_hat)) + up(x_{s+1})       # two-scale Laplacian blend
        x_{t-1} = DDIM/eta step(x_hat, eps)

Random numbers are drawn in the same order as the official code (all scales' initial noise
up front, finest scale first; then fresh step noise for eta > 0), so with the same seed and the
same settings the two implementations produce the same image up to float rounding.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from sidx.core.denoiser import PatchDenoiser
from sidx.core.patches import extract_patches, fold_weighted, gaussian_window
from sidx.core.pyramid import build_pyramid, pyramid_shapes, two_scale_blend
from sidx.core.schedule import FlowLinearSchedule


@dataclass
class CoreConfig:
    """Defaults = the paper's Table-1 preset (`configs/uncond/paper_sinddm15.yaml`), in fp32."""
    patch_size: int = 15
    fold_rho: float = 0.2
    dataset_stride: int = 1
    query_stride: int = 1
    num_scales: int = 4
    relative_scale: float = 0.5
    min_short_side: Optional[int] = None
    num_steps: int = 10
    sigma_min: float = 1e-4
    eta: float = 0.0
    backend: str = "dense"          # dense | chunked | sdpa
    query_chunks: int = 1
    blend: bool = True              # False = ablation: no two-scale blend (plan F4)
    dtype: str = "float32"


def load_image(path: str) -> torch.Tensor:
    """RGB image file -> (3, H, W) float32 in [0, 1] (same as the official loader)."""
    arr = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div(255)


class CoreSampler:
    """Build once per input image, then ``sample(seed)`` as often as needed.

    image : (C, H, W) in [0, 1]
    """

    def __init__(self, image: torch.Tensor, config: Optional[CoreConfig] = None,
                 device: Optional[str] = None) -> None:
        self.cfg = cfg = config or CoreConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = getattr(torch, cfg.dtype)
        x = (image.to(self.device, self.dtype) * 2 - 1)
        self.C = x.shape[0]
        self.shapes = pyramid_shapes(x.shape[-2:], cfg.num_scales, cfg.relative_scale, cfg.min_short_side)
        self.pyramid = [lvl.to(self.dtype) for lvl in build_pyramid(x, self.shapes)]
        for s, (h, w) in enumerate(self.shapes):
            if min(h, w) < cfg.patch_size:
                raise ValueError(f"scale {s} is {h}x{w}, smaller than patch_size={cfg.patch_size}; "
                                 f"use fewer scales or a smaller patch")
        self.denoisers = [PatchDenoiser(extract_patches(lvl, cfg.patch_size, cfg.dataset_stride),
                                        backend=cfg.backend, chunks=cfg.query_chunks)
                          for lvl in self.pyramid]
        self.window = gaussian_window(cfg.patch_size, cfg.fold_rho, self.C, self.device, self.dtype)
        self.schedule = FlowLinearSchedule(cfg.num_steps, cfg.sigma_min)

    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {"config": asdict(self.cfg), "device": str(self.device),
                "pyramid_hw": [list(s) for s in self.shapes],
                "num_patches_per_scale": [d.N for d in self.denoisers]}

    def denoise(self, xt: torch.Tensor, s: int, t: int, argmax: bool = False):
        """One image-level denoiser call at scale s, step t: unfold -> closed form -> fold."""
        cfg = self.cfg
        q = extract_patches(xt[0], cfg.patch_size, cfg.query_stride)
        # alpha/sigma stay 0-dim float32 tensors (as in the official code), so the kernel scale
        # -1/(2 sigma^2) is rounded identically.
        out, idx = self.denoisers[s](q, self.schedule.alphas[t].to(self.device),
                                     self.schedule.sigmas[t].to(self.device), argmax=argmax)
        x_hat = fold_weighted(out, self.window, xt.shape[-2:], cfg.patch_size, cfg.query_stride)
        return x_hat.to(self.dtype), idx

    @torch.inference_mode()
    def sample(self, seed: Optional[int] = None, out_shapes: Optional[Sequence[Tuple[int, int]]] = None,
               hook: Optional[Callable[[Dict[str, Any]], None]] = None,
               record_argmax: bool = False) -> torch.Tensor:
        """Draw one sample. Returns (C, H, W) in [0, 1].

        seed : seeds torch's global RNG first (as the official script does per sample).
        out_shapes : per-scale output sizes (finest first) to generate at another size; defaults to
            the input's pyramid.
        hook : called with step/scale events (see sidx.core.hooks).
        """
        cfg, sch = self.cfg, self.schedule
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        shapes = list(out_shapes or self.shapes)
        S = len(shapes)
        xt = [sch.sigma_max * torch.randn(1, self.C, h, w, device=self.device, dtype=self.dtype)
              for (h, w) in shapes]
        for s in reversed(range(S)):
            for t in reversed(range(1, sch.num_steps + 1)):
                x_hat, idx = self.denoise(xt[s], s, t, argmax=record_argmax)
                eps = sch.eps_from_x_hat(xt[s], x_hat, t)
                if cfg.blend and s != S - 1 and t != 1:
                    x_hat = two_scale_blend(x_hat, xt[s + 1], cfg.relative_scale).to(self.dtype)
                if hook is not None:
                    hook({"kind": "step", "scale": s, "t": t, "sigma": float(sch.sigmas[t]),
                          "x_hat": x_hat, "argmax": idx})
                xt[s] = sch.step(x_hat, eps, t, cfg.eta)
            if hook is not None:
                hook({"kind": "scale", "scale": s, "x": xt[s]})
        return ((xt[0][0] + 1) * 0.5).float()
