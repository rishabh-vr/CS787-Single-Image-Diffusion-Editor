"""Evaluate generated samples (runner output layout) -> per-sample and per-image CSV tables.

    python scripts/evaluate.py <out_root>/<experiment> [--groups paper,patch,copy,structure]
           [--manifest data/manifests/x.csv ...] [--max-samples 50] [--eval-root <dir>]

For every job folder (has job.json) and every image folder in it, the input image is looked up by
id: first in the given manifests (so Colab paths in job.json can be remapped), then in job.json.
Results go to <eval-root>/<experiment>/<job>__<hash>/{per_sample.csv, per_image.csv}
(default eval-root: <out_root>/../eval). A job whose per_image.csv already covers the same samples
is skipped unless --force.

Metric groups (definitions: src/sidx/metrics/*):
    paper      SIFID, NIQE/NIMA/MUSIQ (pyiqa), pixel + LPIPS diversity, near-duplicate rates
    patch      patch SWD, patch precision (per sample), patch recall (per image)
    copy       block-copy ratio, mean copied-region size
    structure  line/skeleton/ghosting/edge/periodicity statistics vs the input
CPU only; run it on the laptop or a Colab CPU runtime, never on GPU quota.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidx.core.resize import resize  # noqa: E402
from sidx.core.sampler import load_image  # noqa: E402

ALL_GROUPS = ("paper", "patch", "copy", "structure")


def read_manifests(paths: List[str]) -> Dict[str, dict]:
    rows = {}
    for p in paths:
        p = Path(p)
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rp = Path(r["path"])
                r["abs_path"] = str(rp if rp.is_absolute() else (p.parent / rp).resolve())
                rows[r["id"]] = r
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _mean(xs) -> float:
    v = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def evaluate_image(ref, samples, seeds, groups, allowed_angles=None, iqa_names=("niqe", "nima", "musiq")):
    from sidx.metrics import copying, paper, patchdist, structure

    per_sample = [{"seed": s} for s in seeds]
    img_row: Dict[str, float] = {"n_samples": len(samples), "height": samples[0].shape[1],
                                 "width": samples[0].shape[2]}
    same_size = all(s.shape == ref.shape for s in samples)
    if "paper" in groups:
        for r, v in zip(per_sample, paper.sifid(ref, samples)):
            r["sifid"] = v
        for r, s in zip(per_sample, samples):
            r.update(paper.iqa_scores(s, iqa_names))
            r["near_dup_frac"] = paper.near_duplicate_fraction(ref, s)
        img_row.update(paper.pixel_diversity(ref, samples) if same_size else {})
        img_row["lpips_div"] = paper.lpips_diversity(samples)
        img_row.update(paper.near_duplicate_rate([r["near_dup_frac"] for r in per_sample]))
    if "patch" in groups:
        pr = patchdist.PatchPR(ref)
        for r, s in zip(per_sample, samples):
            r["patch_swd"] = patchdist.patch_swd(ref, s)
            r["patch_precision"] = pr.add(s)
        img_row["patch_recall"] = pr.recall()
        img_row["patch_tau"] = pr.tau
    if "copy" in groups:
        for r, s in zip(per_sample, samples):
            r.update(copying.copy_regions(ref, s))
    if "structure" in groups:
        sref = structure.StructureReference(ref)
        img_row.update(sref.info())
        for r, s in zip(per_sample, samples):
            r.update(sref.compare(s, allowed_angles))
    # per-image means of the per-sample metrics
    for k in per_sample[0]:
        if k != "seed":
            img_row[f"{k}"] = _mean(r[k] for r in per_sample)
    return per_sample, img_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment_dir")
    ap.add_argument("--groups", default="paper,patch,copy,structure")
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--only-job", default=None, help="substring of the job folder name")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise SystemExit(f"unknown groups {unknown}")
    exp_dir = Path(args.experiment_dir)
    eval_root = Path(args.eval_root) if args.eval_root else exp_dir.parent.parent / "eval"
    manifest = read_manifests(args.manifest)

    for job_dir in sorted(p.parent for p in exp_dir.glob("*/job.json")):
        if args.only_job and args.only_job not in job_dir.name:
            continue
        job = json.loads((job_dir / "job.json").read_text())
        job_images = {i["id"]: i["path"] for i in job.get("images", [])}
        out_dir = eval_root / exp_dir.name / job_dir.name
        image_dirs = sorted(d for d in job_dir.iterdir() if d.is_dir() and list(d.glob("seed_*.png")))
        counts = {d.name: len(list(d.glob("seed_*.png"))) for d in image_dirs}
        if args.max_samples:
            counts = {k: min(v, args.max_samples) for k, v in counts.items()}
        stamp = {"groups": groups, "counts": counts}
        if (out_dir / "stamp.json").exists() and not args.force:
            if json.loads((out_dir / "stamp.json").read_text()) == stamp:
                print(f"[eval] {job_dir.name}: up to date")
                continue
        all_samples, all_images = [], []
        for d in image_dirs:
            meta = manifest.get(d.name, {})
            ref_path = meta.get("abs_path") or job_images.get(d.name)
            if not ref_path or not Path(ref_path).exists():
                print(f"[eval] {job_dir.name}/{d.name}: input image not found ({ref_path}); pass --manifest")
                continue
            seeds_png = sorted(d.glob("seed_*.png"))[: args.max_samples]
            samples = [load_image(str(p)) for p in seeds_png]
            seeds = [int(p.stem.split("_")[1]) for p in seeds_png]
            ref = load_image(ref_path)
            params = job.get("params", {})
            if params.get("resize_to"):
                ref = resize(ref, params["resize_to"]).clamp(0, 1)
            angles = meta.get("allowed_angles")
            angles = [float(a) for a in angles.split(";")] if angles else None
            ps, ir = evaluate_image(ref, samples, seeds, groups, angles)
            for r in ps:
                r.update(job=job_dir.name, image_id=d.name)
            ir = {"job": job_dir.name, "method": job.get("method"), "image_id": d.name,
                  **{k: meta[k] for k in ("set", "category", "factor") if k in meta}, **ir}
            all_samples += ps
            all_images.append(ir)
            print(f"[eval] {job_dir.name}/{d.name}: {len(samples)} samples", flush=True)
        write_csv(out_dir / "per_sample.csv", all_samples)
        write_csv(out_dir / "per_image.csv", all_images)
        (out_dir / "stamp.json").write_text(json.dumps(stamp))


if __name__ == "__main__":
    main()
