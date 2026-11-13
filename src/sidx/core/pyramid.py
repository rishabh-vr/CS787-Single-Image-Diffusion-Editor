"""Image pyramid geometry, pyramid building, and the two-scale Laplacian blend (paper Alg. 2).

Conventions (as in the official code, notes Q2/Q5): scale 0 is the input, each coarser scale is
``ceil(size * relative_scale**s)``, ``num_scales`` counts all levels, and every level is resampled
from the level above it.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from sidx.core.resize import resize


def pyramid_shapes(hw: Sequence[int], num_scales: int, relative_scale: float = 0.5,
                   min_short_side: Optional[int] = None) -> List[Tuple[int, int]]:
    """(H, W) of every scale, finest first.

    ``min_short_side`` (optional) enlarges only the coarsest scale so its short side is at least
    that, clamped to the size of the scale above it (the official rule, `pyramid.py`).
    """
    H0, W0 = int(hw[0]), int(hw[1])
    shapes = [(H0, W0)]
    for s in range(1, num_scales):
        shapes.append((math.ceil(H0 * relative_scale ** s), math.ceil(W0 * relative_scale ** s)))
    if min_short_side is not None and num_scales > 1:
        prev_h, prev_w = shapes[-2]
        if min(prev_h, prev_w) < min_short_side:
            raise ValueError(f"num_scales={num_scales} too large for min_short_side={min_short_side}")
        h, w = shapes[-1]
        if min(h, w) < min_short_side:
            short0 = min(H0, W0)
            shapes[-1] = (min(-(-H0 * min_short_side // short0), prev_h),
                          min(-(-W0 * min_short_side // short0), prev_w))
    return shapes


def build_pyramid(image: torch.Tensor, shapes: Sequence[Tuple[int, int]]) -> List[torch.Tensor]:
    """Gaussian pyramid of ``image`` ([..., H, W]) at ``shapes`` (finest first, = image size)."""
    assert tuple(image.shape[-2:]) == tuple(shapes[0])
    levels = [image]
    for hw in shapes[1:]:
        levels.append(resize(levels[-1], hw))
    return levels


def two_scale_blend(fine: torch.Tensor, coarser: torch.Tensor, relative_scale: float = 0.5) -> torch.Tensor:
    """Keep ``fine``'s detail band, take its low frequencies from the finished ``coarser`` scale:

        out = fine - up(down(fine)) + up(coarser)

    ``down`` goes to the next pyramid size of ``fine``; both ``up``s go back to ``fine``'s size.
    """
    H, W = fine.shape[-2:]
    low_hw = pyramid_shapes((H, W), 2, relative_scale)[1]
    detail = fine - resize(resize(fine, low_hw), (H, W))
    return resize(coarser, (H, W)) + detail
