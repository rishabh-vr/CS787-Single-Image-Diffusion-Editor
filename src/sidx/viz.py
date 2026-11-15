"""Qualitative figures (plan §8 'Qualitative protocol'): input + 6 FIXED seeds (the first six, no cherry-picking)
+ the best and worst sample by the composite structural deviation (lower = better)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw


def _fit(im: Image.Image, size: int) -> Image.Image:
    s = size / max(im.size)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)


def sample_grid(input_path: str, sample_paths: Sequence[str], out_path: str, labels: Optional[Sequence[str]] = None,
                cell: int = 192, title: str = "") -> None:
    """One row: input first, then the samples, each fitted into a square cell with a small caption."""
    paths = [input_path, *sample_paths]
    labels = ["input", *(labels or [Path(p).stem.replace("seed_", "seed ") for p in sample_paths])]
    head = 18 if title else 0
    W = Image.new("RGB", (cell * len(paths), cell + 16 + head), "white")
    d = ImageDraw.Draw(W)
    if title:
        d.text((4, 3), title, fill=(0, 0, 0))
    for i, (p, lab) in enumerate(zip(paths, labels)):
        im = _fit(Image.open(p).convert("RGB"), cell - 4)
        W.paste(im, (i * cell + (cell - im.width) // 2, head + 2 + (cell - im.height) // 2))
        d.text((i * cell + 4, head + cell), lab, fill=(200, 0, 0) if i == 0 else (60, 60, 60))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    W.save(out_path)


def grids_for_job(job_dir: Path, per_sample_csv: Path, input_paths: dict, out_dir: Path, n_fixed: int = 6,
                  score: str = "struct_dev") -> List[Path]:
    """Write one grid per image of a job: input | first `n_fixed` seeds | best | worst by `score`."""
    import pandas as pd

    ps = pd.read_csv(per_sample_csv) if per_sample_csv and Path(per_sample_csv).exists() else None
    written = []
    for img_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        files = sorted(img_dir.glob("seed_*.png"))
        if not files or img_dir.name not in input_paths:
            continue
        fixed = files[:n_fixed]
        extra, labels = [], [Path(f).stem.replace("seed_", "seed ") for f in fixed]
        if ps is not None and score in ps:
            g = ps[ps["image_id"] == img_dir.name].dropna(subset=[score])
            if len(g):
                best, worst = g.loc[g[score].idxmin()], g.loc[g[score].idxmax()]
                for tag, row in (("best", best), ("worst", worst)):
                    f = img_dir / f"seed_{int(row['seed']):04d}.png"
                    if f.exists():
                        extra.append(f)
                        labels.append(f"{tag} ({int(row['seed'])})")
        out = out_dir / f"{job_dir.name.split('__')[0]}__{img_dir.name}.png"
        sample_grid(input_paths[img_dir.name], [*fixed, *extra], str(out), labels, title=f"{job_dir.name.split('__')[0]} / {img_dir.name}")
        written.append(out)
    return written
