"""Run the OFFICIAL efficient-SID sampler on any device (including CPU), for equivalence checks.

The official ``build_image_denoiser`` hard-codes ``.cuda()`` (notes Q10), so this assembles the
official components directly: their scheduler, pyramid processor, exact closed-form denoiser,
image denoiser and ``sample_uncond``. Nothing of ours is used in the official path; our core is
only compared against it. Only the exact backend in fp32 is supported here.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import torch

from sidx.core.sampler import CoreConfig


def import_official(sid_root: Optional[str] = None):
    try:
        import efficient_sid  # noqa: F401
    except ImportError:
        root = sid_root or str(Path(__file__).resolve().parents[3] / "external" / "efficient-SID-code")
        sys.path.insert(0, root)
    warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
    from efficient_sid.applications.uncond import sample_uncond
    from efficient_sid.closed_form_denoiser import ClosedFormDenoiserConfig, ExactClosedFormDenoiser
    from efficient_sid.image_denoiser import ImageDenoiser, ImageDenoiserConfig, extract_patches
    from efficient_sid.pyramid import PyramidConfig, PyramidProcessor
    from efficient_sid.scheduler import FlowLinearConfig, FlowLinearScheduler
    return dict(sample_uncond=sample_uncond, ClosedFormDenoiserConfig=ClosedFormDenoiserConfig,
                ExactClosedFormDenoiser=ExactClosedFormDenoiser, ImageDenoiser=ImageDenoiser,
                ImageDenoiserConfig=ImageDenoiserConfig, extract_patches=extract_patches,
                PyramidConfig=PyramidConfig, PyramidProcessor=PyramidProcessor,
                FlowLinearConfig=FlowLinearConfig, FlowLinearScheduler=FlowLinearScheduler)


class OfficialSampler:
    """The official pipeline for ``image`` ((C,H,W) in [0,1]) under the settings of ``cfg``."""

    def __init__(self, image: torch.Tensor, cfg: CoreConfig, device: str = "cpu",
                 sid_root: Optional[str] = None, contiguous_step_noise: bool = True) -> None:
        assert cfg.dtype == "float32" and cfg.backend in ("dense", "chunked")
        o = self.o = import_official(sid_root)
        self.device = torch.device(device)
        x = image.to(self.device, torch.float32) * 2 - 1
        self.processor = o["PyramidProcessor"](o["PyramidConfig"](
            relative_scale=cfg.relative_scale, num_scales=cfg.num_scales, min_short_side=cfg.min_short_side))
        pyramid = self.processor.build_gaussian_pyramid(x)
        self.schedulers = [o["FlowLinearScheduler"](o["FlowLinearConfig"](
            num_steps=cfg.num_steps, sigma_min=cfg.sigma_min, eta=cfg.eta), self.device)
            for _ in pyramid]
        self.denoisers = []
        for lvl in pyramid:
            pd_cfg = o["ClosedFormDenoiserConfig"](type="exact", bottleneck_dtype="float32",
                                                   query_chunks=cfg.query_chunks)
            data = o["extract_patches"](lvl, cfg.patch_size, cfg.dataset_stride)
            pd = o["ExactClosedFormDenoiser"](pd_cfg, data)
            id_cfg = o["ImageDenoiserConfig"](patch_size=cfg.patch_size, dataset_stride=cfg.dataset_stride,
                                              query_stride=cfg.query_stride, fold_rho=cfg.fold_rho,
                                              patch_denoiser=pd_cfg)
            self.denoisers.append(o["ImageDenoiser"](id_cfg, pd, dtype=torch.float32).to(self.device))
        self.shapes = self.processor.gaussian_pyramid_shapes(x.shape)
        self.eta = cfg.eta
        if contiguous_step_noise:
            # The official blend returns a NON-contiguous tensor, and Scheduler.step draws its fresh
            # noise with randn_like(x_hat), which keeps that layout; on CPU a strided normal_ draws a
            # different random stream than a contiguous one (notes D14). Making x_hat contiguous
            # changes no arithmetic, only which random numbers an eta > 0 step receives, so that
            # the official and our sampler see the same noise.
            for sch in self.schedulers:
                step = sch.step
                sch.step = (lambda x_hat, eps, t, eta=0.0, target_t=None, _step=step:
                            _step(x_hat.contiguous(), eps, t, eta=eta, target_t=target_t))

    def sample(self, seed: int) -> torch.Tensor:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        return self.o["sample_uncond"](denoisers=self.denoisers, schedulers=self.schedulers,
                                       pyramid_processor=self.processor,
                                       sample_pyramid_shapes=self.shapes, eta=self.eta,
                                       vae=None, device=self.device, dtype=torch.float32)
