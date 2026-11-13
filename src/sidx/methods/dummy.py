"""A CPU-only stand-in generator for testing the runner and the evaluation pipeline.

It cuts the input into square blocks and shuffles them with the seed. That is a crude
"same pieces, new arrangement" image — useful as a sanity baseline for patch-distribution metrics,
not as a real method.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from sidx.methods.base import Method


class DummyShuffle(Method):
    name = "dummy_shuffle"

    def __init__(self, block: int = 16, sleep_s: float = 0.0, **params: Any) -> None:
        super().__init__(block=block, sleep_s=sleep_s, **params)
        self.block = int(block)
        self.sleep_s = float(sleep_s)  # lets tests emulate a slow method for time-budget checks
        self.image: torch.Tensor | None = None

    def setup(self, image_path: str) -> Dict[str, Any]:
        arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        self.image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        _, h, w = self.image.shape
        return {"input_hw": [h, w]}

    def sample(self, seed: int) -> torch.Tensor:
        assert self.image is not None, "call setup() first"
        if self.sleep_s:
            time.sleep(self.sleep_s)
        c, h, w = self.image.shape
        b = self.block
        hb, wb = h // b, w // b
        crop = self.image[:, : hb * b, : wb * b]
        blocks = crop.reshape(c, hb, b, wb, b).permute(1, 3, 0, 2, 4).reshape(hb * wb, c, b, b)
        g = torch.Generator().manual_seed(int(seed))
        blocks = blocks[torch.randperm(hb * wb, generator=g)]
        return blocks.reshape(hb, wb, c, b, b).permute(2, 0, 3, 1, 4).reshape(c, hb * b, wb * b)

    def teardown(self) -> None:
        self.image = None
