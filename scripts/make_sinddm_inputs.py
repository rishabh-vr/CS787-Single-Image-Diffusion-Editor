"""Phase 8: the 3 OOD images SinDDM is trained on (plan §9), at 128 px, plus their manifest.

SinDDM at 128 px costs ~712 s per 1000 steps on a free T4 (notebook 05), so a 20k-step reduced run is
~4 h per image. The paper's method is run on the SAME 128-px files, so both methods see identical inputs.

    python scripts/make_sinddm_inputs.py   ->  data/sinddm/*.png, data/manifests/sinddm_ood3.csv
"""
import csv, hashlib, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sidx.core.resize import resize  # noqa: E402
from sidx.core.sampler import load_image  # noqa: E402

PICK = [("sd_b0", "data/synthetic/setB/b0_s0.png", "synthetic", "Set B base grid (128 px)"),
        ("sd_fp_house2", "data/ood/setA/fp_house2.png", "floorplan", "Set A floor plan, long side -> 128"),
        ("sd_pat_p4", "data/ood/setA/pat_p4.png", "pattern", "Set A p4 wallpaper pattern, 256 -> 128")]
out = ROOT / "data" / "sinddm"
out.mkdir(parents=True, exist_ok=True)
manifest = ROOT / "data" / "manifests" / "sinddm_ood3.csv"
rows = []
for image_id, src, cat, note in PICK:
    img = load_image(str(ROOT / src))
    H, W = img.shape[-2:]
    s = 128 / max(H, W)
    small = img if s == 1 else resize(img, (round(H * s), round(W * s))).clamp(0, 1)
    p = out / f"{image_id}.png"
    Image.fromarray((small.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)).save(p)
    rows.append({"id": image_id, "set": "sinddm_ood3", "category": cat,
                 "path": Path(os.path.relpath(p, manifest.parent)).as_posix(),
                 "height": small.shape[1], "width": small.shape[2],
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "source": note, "licence": "see source manifests"})
with open(manifest, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print([(r["id"], r["height"], r["width"]) for r in rows])
