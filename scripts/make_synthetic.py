"""Phase 6: render Set B (synthetic line drawings) and write its manifest.

    python scripts/make_synthetic.py            # -> data/synthetic/setB/*.png + data/manifests/synthetic_setB.csv

Deterministic (seeded); the images are ours, so they are committed with the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidx.core.sampler import load_image  # noqa: E402
from sidx.data.descriptors import compute_descriptors  # noqa: E402
from sidx.data.synthetic import design, render, spec_dict  # noqa: E402


def main() -> None:
    out_dir = ROOT / "data" / "synthetic" / "setB"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data" / "manifests" / "synthetic_setB.csv"
    rows = []
    for image_id, factor, spec in design():
        p = out_dir / f"{image_id}.png"
        render(spec).save(p)
        row = {"id": image_id, "set": "synthetic_setB", "category": "synthetic", "factor": factor,
               "path": Path(os.path.relpath(p, manifest.parent)).as_posix(),
               "height": spec.size, "width": spec.size,
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "source": "procedural (src/sidx/data/synthetic.py)", "licence": "ours (MIT, this repo)",
               "allowed_angles": "0;90", **spec_dict(spec)}
        row.update(compute_descriptors(load_image(str(p))))
        rows.append(row)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images -> {out_dir.relative_to(ROOT)}; manifest {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
