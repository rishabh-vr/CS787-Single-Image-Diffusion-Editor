"""Set B — procedurally generated line drawings with one factor varied at a time (plan §4.3).

Every image is black ink on white, rendered at 4x and box-downsampled (so lines are anti-aliased,
like rasterised CAD drawings). Lines are axis-aligned, so the allowed orientations are exactly
0 and 90 degrees, and structural metrics have a known ground truth.

Base configuration (B0): 128 x 128 px, line thickness 2 px, 5 lines per 100 px (spacing 20 px),
long lines spanning the image, perfect lattice, X junctions (a full grid), no text.

Factors (each varied alone around B0):
    thickness    1, 2, 4, 8 px
    density      2, 5, 10 lines per 100 px
    length       long (spanning) | short (dashes of 10 px, shorter than the 15-px patch)
    periodicity  perfect | jitter (line positions + N(0, 2 px)) | random (uniform positions)
    junctions    X (grid) | T (verticals stop at alternate horizontals) | none (horizontals only)
    text         none | sparse (a few short labels in grid cells)
"""

from __future__ import annotations

import string
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 4      # supersampling factor


@dataclass(frozen=True)
class SynthSpec:
    size: int = 128
    thickness: float = 2.0
    density: float = 5.0             # lines per 100 px
    length: str = "long"             # long | short
    periodicity: str = "perfect"     # perfect | jitter | random
    junctions: str = "X"             # X | T | none
    text: str = "none"               # none | sparse
    seed: int = 0

    @property
    def spacing(self) -> float:
        return 100.0 / self.density


BASE = SynthSpec()

FACTORS: Dict[str, List] = {
    "thickness": [1.0, 2.0, 4.0, 8.0],
    "density": [2.0, 5.0, 10.0],
    "length": ["long", "short"],
    "periodicity": ["perfect", "jitter", "random"],
    "junctions": ["X", "T", "none"],
    "text": ["none", "sparse"],
}


def _positions(n_px: int, spec: SynthSpec, rng: np.random.Generator) -> np.ndarray:
    sp = spec.spacing
    n = max(1, int(round(n_px / sp)))
    if spec.periodicity == "random":
        pos = np.sort(rng.uniform(sp / 2, n_px - sp / 2, n))
    else:
        pos = sp / 2 + sp * np.arange(n)
        if spec.periodicity == "jitter":
            pos = pos + rng.normal(0, 2.0, n)
    return pos[(pos > 1) & (pos < n_px - 1)]


def render(spec: SynthSpec) -> Image.Image:
    rng = np.random.default_rng(spec.seed)
    S = spec.size * SS
    img = Image.new("L", (S, S), 255)
    d = ImageDraw.Draw(img)
    w = max(1, int(round(spec.thickness * SS)))
    ys = _positions(spec.size, spec, rng) * SS
    xs = _positions(spec.size, spec, rng) * SS

    def hline(y, x0, x1):
        if spec.length == "short":
            x = x0 + rng.uniform(0, 10 * SS)
            while x < x1:
                d.line([(x, y), (min(x + 10 * SS, x1), y)], fill=0, width=w)
                x += 20 * SS
        else:
            d.line([(x0, y), (x1, y)], fill=0, width=w)

    def vline(x, y0, y1):
        if spec.length == "short":
            y = y0 + rng.uniform(0, 10 * SS)
            while y < y1:
                d.line([(x, y), (x, min(y + 10 * SS, y1))], fill=0, width=w)
                y += 20 * SS
        else:
            d.line([(x, y0), (x, y1)], fill=0, width=w)

    for y in ys:
        hline(y, 0, S)
    if spec.junctions == "X":
        for x in xs:
            vline(x, 0, S)
    elif spec.junctions == "T" and len(ys) >= 2:
        # each vertical runs between two consecutive horizontals, alternating rows -> T junctions
        for i, x in enumerate(xs):
            for k in range(i % 2, len(ys) - 1, 2):
                vline(x, ys[k], ys[k + 1])
    if spec.text == "sparse":
        font = ImageFont.load_default(size=9 * SS) if hasattr(ImageFont, "load_default") else None
        n_labels = 4
        for _ in range(n_labels):
            label = "".join(rng.choice(list(string.ascii_uppercase + string.digits), 2))
            x, y = rng.uniform(0.1, 0.8) * S, rng.uniform(0.1, 0.8) * S
            d.text((x, y), label, fill=0, font=font)
    return img.resize((spec.size, spec.size), Image.BOX).convert("RGB")


def design(size: int = 128) -> List[Tuple[str, str, SynthSpec]]:
    """One-factor-at-a-time design: (image_id, factor, spec). Every factor level appears once per
    layout seed; the base B0 is shared by all factors (listed once, factor "base").
    ``size`` is the image side in px (128 = Set B for the GPU; 64 = the laptop pilot, Set B64)."""
    out = []
    for seed in (0, 1):
        base = replace(BASE, seed=seed, size=size)
        out.append((f"b0_s{seed}", "base", base))
        for factor, levels in FACTORS.items():
            for lv in levels:
                spec = replace(base, **{factor: lv})
                if spec == base:
                    continue
                lv_s = str(lv).replace(".0", "")
                out.append((f"{factor}_{lv_s}_s{seed}", factor, spec))
    return out


def spec_dict(spec: SynthSpec) -> dict:
    return asdict(spec)
