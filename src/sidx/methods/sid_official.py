"""Wrapper around the official efficient-SID code (unconditional generation).

Mirrors ``scripts/sample_uncond.py`` of the official repo (audited at commit 85381e3; see
docs/technical_notes.md §2). The difference: the per-image build (pyramid, patch datasets,
denoisers) happens once in ``setup``, and ``sample(seed)`` draws one sample. The runner can then
skip seeds that already exist on Drive and resume an interrupted session.

Determinism matches the official script: the script seeds with ``seed + i`` before sample i, and
here we seed with ``seed`` before each sample. So ``sample(10 + i)`` reproduces the script's
sample i at ``seed=10``.

CUDA only — the official code hard-codes ``.cuda()`` (notes Q10).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sidx.methods.base import Method
from sidx.utils.runtime import GpuTimer


def _import_efficient_sid(sid_root: Optional[str]):
    """Import the official package, from an installed copy or from ``sid_root``."""
    try:
        import efficient_sid  # noqa: F401
    except ImportError:
        if not sid_root:
            raise
        sys.path.insert(0, str(sid_root))
    from efficient_sid import build
    from efficient_sid.applications.uncond import sample_uncond
    from efficient_sid.config import UncondAppConfig, load
    from efficient_sid.latent import maybe_encode, maybe_load_vae
    from efficient_sid.utils import (configure_matmul_precision, load_image_as_tensor,
                                     normalize_to_neg_one_to_one, seed_everything, torch_dtype,
                                     torch_resize)
    return dict(build=build, sample_uncond=sample_uncond, UncondAppConfig=UncondAppConfig,
                load=load, maybe_encode=maybe_encode, maybe_load_vae=maybe_load_vae,
                configure_matmul_precision=configure_matmul_precision,
                load_image_as_tensor=load_image_as_tensor,
                normalize_to_neg_one_to_one=normalize_to_neg_one_to_one,
                seed_everything=seed_everything, torch_dtype=torch_dtype, torch_resize=torch_resize)


class SidOfficial(Method):
    """Parameters
    ----------
    config : str
        Preset YAML, absolute or relative to ``sid_root`` (e.g. "configs/uncond/paper_sinddm15.yaml").
    overrides : list[str]
        OmegaConf dotlist overrides, exactly as on the official CLI (e.g. "precision.dtype=float32").
    sid_root : str
        Path of the official repo checkout. Defaults to $SID_ROOT.
    """

    name = "sid_official"

    def __init__(self, config: str, overrides: Optional[List[str]] = None,
                 sid_root: Optional[str] = None, **params: Any) -> None:
        sid_root = sid_root or os.environ.get("SID_ROOT")
        super().__init__(config=config, overrides=list(overrides or []), sid_root=sid_root, **params)
        self.sid_root = sid_root
        cfg_path = Path(config)
        if not cfg_path.is_absolute() and sid_root:
            cfg_path = Path(sid_root) / cfg_path
        self.config_path = cfg_path
        self.overrides = list(overrides or [])
        self._m = None
        self._state: Dict[str, Any] = {}
        self._resolved: Any = None

    # ------------------------------------------------------------------
    def setup(self, image_path: str) -> Dict[str, Any]:
        import torch
        from omegaconf import OmegaConf

        m = self._m = _import_efficient_sid(self.sid_root)
        if not torch.cuda.is_available():
            raise RuntimeError("sid_official needs a CUDA GPU (the official code hard-codes .cuda()).")

        config = m["load"](m["UncondAppConfig"], self.config_path,
                           [*self.overrides, f"image_path={image_path}"])
        m["seed_everything"](config.seed)
        m["configure_matmul_precision"](config.precision.get("matmul"), config.precision.dtype,
                                        config.image_denoiser.patch_denoiser.bottleneck_dtype)
        device = torch.device("cuda")
        dtype = m["torch_dtype"](config.precision.dtype)
        build = m["build"]

        timer = GpuTimer()
        with timer.measure():
            vae = m["maybe_load_vae"](config.latent, device)
            image = m["load_image_as_tensor"](config.image_path).to(dtype).to(device)
            if config.get("input_resize_to", None) is not None:
                image = m["torch_resize"](image, build.parse_size(config.input_resize_to, "input_resize_to"))
            input_hw = list(image.shape[-2:])
            image = m["normalize_to_neg_one_to_one"](image)
            image = m["maybe_encode"](image, vae)
            sample_size = build.resolve_sample_size(config, image.shape[-2:], vae)
            config = build.resolve(config, image.shape[-2:], sample_size, auto_fields=build.AUTO_FIELDS)
            schedulers = build.build_schedulers(config.scheduler, config.pyramid.num_scales, device)
            pyramid_processor, pyramid = build.build_pyramid(config.pyramid, image, dtype)
            denoisers = build.build_image_denoisers(config.image_denoiser, pyramid, dtype=dtype)
            sample_pyramid_shapes = pyramid_processor.gaussian_pyramid_shapes((image.shape[0], *sample_size))

        self._state = dict(config=config, device=device, dtype=dtype, vae=vae, schedulers=schedulers,
                           pyramid_processor=pyramid_processor, denoisers=denoisers,
                           sample_pyramid_shapes=sample_pyramid_shapes)
        self._resolved = OmegaConf.to_container(config, resolve=True)
        return {
            "input_hw": input_hw,
            "pyramid_hw": [list(p.shape[-2:]) for p in pyramid],
            "num_patches_per_scale": [d.patch_denoiser.N if d is not None else None for d in denoisers],
            "setup_seconds": round(timer.seconds, 3),
            "setup_peak_vram_gb": timer.peak_vram_gb,
        }

    def sample(self, seed: int):
        s, m = self._state, self._m
        assert s, "call setup() first"
        m["seed_everything"](int(seed))
        return m["sample_uncond"](
            denoisers=s["denoisers"],
            schedulers=s["schedulers"],
            pyramid_processor=s["pyramid_processor"],
            sample_pyramid_shapes=s["sample_pyramid_shapes"],
            eta=s["config"].scheduler.eta,
            vae=s["vae"],
            device=s["device"],
            dtype=s["dtype"],
            intermediate_output_dir=None,
        )

    def teardown(self) -> None:
        import gc

        import torch

        for d in self._state.get("denoisers", []) or []:
            if d is not None and hasattr(d.patch_denoiser, "free_gpu"):
                d.patch_denoiser.free_gpu()
        self._state = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "config_path": str(self.config_path), "overrides": self.overrides,
                "resolved_config": self._resolved}
