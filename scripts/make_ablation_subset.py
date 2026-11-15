"""Write data/manifests/ablation_subset.csv: the fixed 6-image subset used by every ablation (plan §10)
= 2 floor plans, 2 patterns, 1 circuit (Set A) + balloons (natural, sinddm15)."""
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PICK = [("ood_setA", i) for i in ("fp_layout", "fp_house2", "pat_p4", "pat_cmm", "ckt_24aa")] + [("uncond_sinddm15", "balloons")]
rows = []
for s, i in PICK:
    with open(ROOT / "data" / "manifests" / f"{s}.csv", newline="", encoding="utf-8") as f:
        r = next(x for x in csv.DictReader(f) if x["id"] == i)
    r["category"] = r.get("category") or "natural"
    r["path"] = "../" + r["path"].split("../", 1)[-1] if r["path"].startswith("../") else r["path"]
    rows.append(r)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(ROOT / "data" / "manifests" / "ablation_subset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print([r["id"] for r in rows])
