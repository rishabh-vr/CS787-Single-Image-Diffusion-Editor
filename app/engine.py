"""Engine for the interactive demo: the project's own sampler (sidx.core) plus what the UI needs.

Additions over ``CoreSampler`` (the math of the method is unchanged):

* ``stride`` - evaluate the closed-form denoiser only on a grid of query patches (and keep only a
  grid of dataset patches). The grid always includes the last row and column, so every pixel is
  covered (plain strided unfolding can leave the border uncovered). stride = 1 is the paper's
  exact preset; stride 2 is ~16x less work and is what makes CPU use practical.
* retargeting - generate at a different output size (the patch datasets still come from the input).
* provenance - at the last step every output patch is essentially one input patch; we record which
  one (the argmax of the softmax weights) and paint it as a map.
* timing and per-step progress for the UI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sidx.core import CoreConfig, CoreSampler
from sidx.core.patches import extract_patches
from sidx.core.pyramid import pyramid_shapes
from sidx.core.resize import resize


# ----------------------------------------------------------------------------- image helpers
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def fit_long_side(x: torch.Tensor, long_side: int) -> torch.Tensor:
    """Antialiased resize so the long side is ``long_side`` (never upsamples more than needed)."""
    H, W = x.shape[-2:]
    s = long_side / max(H, W)
    hw = (max(8, round(H * s)), max(8, round(W * s)))
    return x if hw == (H, W) else resize(x, hw).clamp(0, 1)


def auto_num_scales(hw: Tuple[int, int], patch_size: int, max_scales: int = 4) -> int:
    """Most scales (<= max_scales) whose coarsest level is still >= 1.5 patches on its short side."""
    best = 1
    for s in range(1, max_scales + 1):
        if min(pyramid_shapes(hw, s)[-1]) >= 1.5 * patch_size:
            best = s
    return best


def _grid(n: int, p: int, stride: int) -> torch.Tensor:
    """Patch start positions 0, stride, 2*stride, ... plus the last valid one (n - p)."""
    pos = list(range(0, n - p + 1, stride))
    if pos[-1] != n - p:
        pos.append(n - p)
    return torch.tensor(pos)


# ----------------------------------------------------------------------------- sampler
class DemoSampler(CoreSampler):
    def __init__(self, image: torch.Tensor, config: CoreConfig, device: str, stride: int = 1) -> None:
        self.stride = int(stride)
        self.want_provenance = True
        config.query_stride = 1                      # query striding is done here, with full coverage
        config.dataset_stride = self.stride
        super().__init__(image, config, device)
        # dataset patch index -> (y, x) position in its pyramid level (for the provenance map)
        p = config.patch_size
        self._ds_cols = [(w - p) // self.stride + 1 for (_, w) in self.shapes]

    def denoise(self, xt: torch.Tensor, s: int, t: int, argmax: bool = False):
        cfg, p = self.cfg, self.cfg.patch_size
        H, W = xt.shape[-2:]
        # provenance needs the winners only at the very last step of the finest scale
        argmax = bool(self.want_provenance and s == 0 and t == 1)
        full = extract_patches(xt[0], p, 1)                                   # (L, D)
        ys, xs = _grid(H, p, self.stride), _grid(W, p, self.stride)
        sel = (ys[:, None] * (W - p + 1) + xs[None]).reshape(-1).to(full.device)
        out, idx = self.denoisers[s](full[sel], self.schedule.alphas[t].to(self.device),
                                     self.schedule.sigmas[t].to(self.device), argmax=argmax)
        buf = torch.zeros_like(full)
        buf[sel] = out * self.window
        cov = torch.zeros_like(full)
        cov[sel] = self.window.expand(len(sel), -1)
        num = F.fold(buf.T.unsqueeze(0), (H, W), kernel_size=p)
        den = F.fold(cov.T.unsqueeze(0), (H, W), kernel_size=p)
        x_hat = torch.nan_to_num(num / den).to(self.dtype)
        if argmax and idx is not None:
            self._last_winners = (s, ys, xs, idx.cpu())
        return x_hat, idx

    def source_position(self, s: int, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cols = self._ds_cols[s]
        return (idx // cols) * self.stride, (idx % cols) * self.stride


@dataclass
class Result:
    image: torch.Tensor                          # (3, H, W) in [0, 1]
    seed: int
    seconds: float
    trace: Dict[int, torch.Tensor] = field(default_factory=dict)   # scale -> finished image
    provenance: Optional[np.ndarray] = None      # (H, W, 3) uint8: colour = where the patch came from


def position_colours(h: int, w: int) -> np.ndarray:
    """2-D colour key: every position gets its own colour (red = x, green = y, blue = 1 - x)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / max(w - 1, 1), yy / max(h - 1, 1)
    rgb = np.stack([0.15 + 0.85 * u, 0.15 + 0.85 * v, 1.0 - 0.85 * u], -1)
    return (rgb * 255).astype(np.uint8)


def provenance_map(sampler: DemoSampler, out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Paint each output patch centre with the colour of the input position it was taken from."""
    if getattr(sampler, "_last_winners", None) is None:
        return None
    s, ys, xs, idx = sampler._last_winners
    if s != 0:
        return None
    p = sampler.cfg.patch_size
    H, W = out_hw
    h0, w0 = sampler.shapes[0]
    sy, sx = sampler.source_position(0, idx)
    key = position_colours(h0, w0)
    cy = (sy + p // 2).clamp(max=h0 - 1).numpy()
    cx = (sx + p // 2).clamp(max=w0 - 1).numpy()
    grid = key[cy, cx].reshape(len(ys), len(xs), 3)
    # nearest-neighbour fill of the patch-centre grid back to full size
    out = np.zeros((H, W, 3), np.uint8)
    centres_y = (ys + p // 2).numpy()
    centres_x = (xs + p // 2).numpy()
    iy = np.abs(np.arange(H)[:, None] - centres_y[None]).argmin(1)
    ix = np.abs(np.arange(W)[:, None] - centres_x[None]).argmin(1)
    out[:] = grid[iy][:, ix]
    return out


def generate(sampler: DemoSampler, seed: int, out_hw: Optional[Tuple[int, int]] = None,
             on_step: Optional[Callable[[int, int], None]] = None) -> Result:
    """One sample. ``out_hw`` = output size (retargeting); ``on_step(scale, t)`` for progress."""
    trace: Dict[int, torch.Tensor] = {}

    def hook(ev):
        if ev["kind"] == "step" and on_step is not None:
            on_step(ev["scale"], ev["t"])
        elif ev["kind"] == "scale":
            trace[ev["scale"]] = ((ev["x"][0].float().cpu() + 1) * 0.5).clamp(0, 1)

    shapes = None
    if out_hw is not None and tuple(out_hw) != tuple(sampler.shapes[0]):
        shapes = pyramid_shapes(out_hw, len(sampler.shapes), sampler.cfg.relative_scale)
        if min(shapes[-1]) < sampler.cfg.patch_size:
            raise ValueError("output too small for this many scales")
    sampler._last_winners = None
    t0 = time.perf_counter()
    img = sampler.sample(seed, out_shapes=shapes, hook=hook).clamp(0, 1).cpu()
    secs = time.perf_counter() - t0
    return Result(img, seed, secs, trace, provenance_map(sampler, tuple(img.shape[-2:])))


# ----------------------------------------------------------------------------- copying metrics
def copying_metrics(reference: torch.Tensor, samples: List[torch.Tensor]) -> Dict[str, object]:
    """Our project metrics, computed exactly as in evaluation (sidx.metrics):
    patch precision per sample, patch recall pooled over the samples, copied-block ratio per sample."""
    from sidx.metrics.copying import copy_regions
    from sidx.metrics.patchdist import PatchPR

    pr = PatchPR(reference)
    prec = [pr.add(x) for x in samples]
    same_size = [x.shape[-2:] == reference.shape[-2:] for x in samples]
    copied = [copy_regions(reference, x)["block_copy_ratio"] if ok else float("nan")
              for x, ok in zip(samples, same_size)]
    return {"precision": prec, "recall": pr.recall(), "copied": copied}
