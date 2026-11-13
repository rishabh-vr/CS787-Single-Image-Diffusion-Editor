"""Our re-implementation (sidx.core) behind the common Method interface.

Unlike ``sid_official`` this runs on CPU as well as CUDA, so small jobs (synthetic 64-128 px sweeps,
tests) can run on the laptop. Seeding matches ``sid_official``: ``sample(seed)`` seeds torch's
global RNG with ``seed`` and then draws, so the same seed gives the same sample in both
implementations (up to float rounding; see reports/equivalence/).

Job-file params: any CoreConfig field (patch_size, fold_rho, num_scales, num_steps, eta, backend,
query_chunks, blend, ...), plus ``device`` ("auto" | "cpu" | "cuda") and ``resize_to`` ([H, W],
optional; resampled with our antialiased cubic resize before anything else).
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Dict, Optional

from sidx.methods.base import Method
from sidx.utils.runtime import GpuTimer


class SidCore(Method):
    name = "sid_core"

    def __init__(self, device: str = "auto", resize_to: Optional[list] = None, **params: Any) -> None:
        from sidx.core import CoreConfig

        known = {f.name for f in fields(CoreConfig)}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"unknown sid_core params: {sorted(unknown)}; known: {sorted(known)}")
        super().__init__(device=device, resize_to=resize_to, **params)
        self.config = CoreConfig(**params)
        self.device = device
        self.resize_to = resize_to
        self._sampler = None

    def setup(self, image_path: str) -> Dict[str, Any]:
        import torch

        from sidx.core import CoreSampler
        from sidx.core.resize import resize
        from sidx.core.sampler import load_image

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        image = load_image(image_path)
        input_hw = list(image.shape[-2:])
        if self.resize_to:
            image = resize(image, self.resize_to).clamp(0, 1)
        timer = GpuTimer()
        with timer.measure():
            self._sampler = CoreSampler(image, self.config, dev)
        info = self._sampler.info()
        info.update(input_hw=input_hw, setup_seconds=round(timer.seconds, 3),
                    setup_peak_vram_gb=timer.peak_vram_gb)
        return info

    def sample(self, seed: int):
        assert self._sampler is not None, "call setup() first"
        return self._sampler.sample(int(seed))

    def teardown(self) -> None:
        import gc

        import torch

        self._sampler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "device": self.device, "resize_to": self.resize_to,
                "config": asdict(self.config)}
