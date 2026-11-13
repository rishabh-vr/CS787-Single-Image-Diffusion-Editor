"""The one interface every generator implements.

Every method — the official efficient-SID code, our re-implementation, SinDDM, LoRA — is driven the
same way by the job runner, and writes the same output layout, so evaluation never needs to know
which method produced a folder.

Lifecycle per (job, image):
    m = make_method(name, **params)
    info = m.setup(image_path)        # build per-image state once (patch datasets, models, ...)
    for seed in seeds:
        img = m.sample(seed)          # (C, H, W) float tensor in [0, 1], on any device
    m.teardown()                      # free GPU memory before the next image
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict


class Method(ABC):
    #: short identifier used in output paths
    name: str = "method"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def setup(self, image_path: str) -> Dict[str, Any]:
        """Prepare everything that depends on the input image. Returns facts worth logging."""

    @abstractmethod
    def sample(self, seed: int):
        """Draw one sample for ``seed``. Must be deterministic given (setup image, params, seed)."""

    def teardown(self) -> None:
        """Release per-image resources (GPU memory)."""

    def describe(self) -> Dict[str, Any]:
        """The fully resolved configuration, for meta.json. Valid after ``setup``."""
        return {"name": self.name, "params": self.params}


#: method name -> "module:Class", imported lazily so CPU-only machines never import CUDA-only code.
METHOD_REGISTRY: Dict[str, str] = {
    "dummy_shuffle": "sidx.methods.dummy:DummyShuffle",
    "sid_official": "sidx.methods.sid_official:SidOfficial",
}


def make_method(name: str, **params: Any) -> Method:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method {name!r}; known: {', '.join(METHOD_REGISTRY)}")
    module_name, cls_name = METHOD_REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**params)
