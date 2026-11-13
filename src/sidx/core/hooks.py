"""Instrumentation for the core sampler.

The sampler calls ``hook(event)`` with a dict. Events:

    {"kind": "step",  "scale": s, "t": t, "sigma": float, "x_hat": (1,C,H,W), "argmax": (M,) or None}
    {"kind": "scale", "scale": s, "x": (1,C,H,W)}          # the finished scale

Tensors are passed as-is (no copies); a hook that keeps them must clone them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class ScaleRecorder:
    """Keeps every finished scale, in [0, 1] (the coarse-to-fine trace, plan F3)."""

    def __init__(self) -> None:
        self.scales: Dict[int, torch.Tensor] = {}

    def __call__(self, event: Dict[str, Any]) -> None:
        if event["kind"] == "scale":
            self.scales[event["scale"]] = ((event["x"][0].float().cpu() + 1) * 0.5).clamp(0, 1)


class WinnerRecorder:
    """Keeps the index of the winning dataset patch per query location at the chosen steps
    (default: the last step, t = 1, where the output copies that patch). Plan F2.

    Requires the sampler to run with ``record_argmax=True``.
    """

    def __init__(self, steps: Optional[List[int]] = None) -> None:
        self.steps = steps or [1]
        self.winners: Dict[tuple, torch.Tensor] = {}

    def __call__(self, event: Dict[str, Any]) -> None:
        if event["kind"] == "step" and event["t"] in self.steps and event["argmax"] is not None:
            self.winners[(event["scale"], event["t"])] = event["argmax"].cpu().clone()


class Hooks:
    """Fan one event out to several hooks."""

    def __init__(self, *hooks) -> None:
        self.hooks = [h for h in hooks if h is not None]

    def __call__(self, event: Dict[str, Any]) -> None:
        for h in self.hooks:
            h(event)
