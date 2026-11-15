"""Phase 8: manifest of the 8 images DreamBooth-LoRA is trained on (plan §9): the 3 natural images that
SinDDM ships pretrained models for + 5 OOD images (2 floor plans, 2 patterns, 1 circuit).
Also stores the per-category instance prompt. No new images: paths point at the existing sets."""
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {"natural": "a photo in the style of sks", "floorplan": "a floor plan drawing in the style of sks",
           "pattern": "a geometric tiling pattern in the style of sks", "circuit": "a circuit diagram in the style of sks"}
PICK = [("uncond_sinddm15", i) for i in ("forest", "night_sky", "pyramids")] + \
       [("ood_setA", i) for i in ("fp_layout", "fp_house2", "pat_p4", "pat_cmm", "ckt_24aa")]
rows = []
for s, i in PICK:
    with open(ROOT / "data" / "manifests" / f"{s}.csv", newline="", encoding="utf-8") as f:
        r = next(x for x in csv.DictReader(f) if x["id"] == i)
    r["category"] = r.get("category") or "natural"
    r["prompt"] = PROMPTS[r["category"]]
    rows.append(r)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(ROOT / "data" / "manifests" / "lora8.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print([(r["id"], r["category"]) for r in rows])
