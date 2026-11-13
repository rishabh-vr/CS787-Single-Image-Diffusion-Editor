#!/usr/bin/env python
"""Run a job file with a time budget; rerun the same command to resume.

Examples
--------
Colab (GPU):
    python scripts/run_queue.py configs/jobs/phase2_t4_timing.yaml \
        --out-root /content/drive/MyDrive/CS787/outputs --budget-min 100 \
        --var SID_ROOT=/content/efficient-SID-code

Laptop (CPU smoke test of the runner itself):
    python scripts/run_queue.py configs/jobs/local_dummy_smoke.yaml --out-root outputs_local

Placeholders like {SID_ROOT} in the job file are filled from --var KEY=VALUE, then from environment
variables of the same name. {PROJECT_ROOT} defaults to this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # works without `pip install -e .`

from sidx.runner import load_jobs, run_jobs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_file", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--budget-min", type=float, default=None, help="stop cleanly after ~this many minutes")
    ap.add_argument("--max-samples", type=int, default=None, help="stop after this many new samples")
    ap.add_argument("--var", action="append", default=[], metavar="KEY=VALUE", help="placeholder value")
    ap.add_argument("--only", action="append", default=[], help="run only jobs with these names")
    ap.add_argument("--dry-run", action="store_true", help="list jobs, hashes and pending counts; run nothing")
    args = ap.parse_args()

    variables = {"PROJECT_ROOT": str(PROJECT_ROOT)}
    for key in ("SID_ROOT", "DATA_ROOT", "DRIVE_ROOT", "SINDDM_ROOT"):
        if key in os.environ:
            variables[key] = os.environ[key]
    for kv in args.var:
        k, v = kv.split("=", 1)
        variables[k] = v

    experiment, jobs = load_jobs(args.job_file, variables)
    if args.only:
        jobs = [j for j in jobs if j.name in set(args.only)]

    if args.dry_run:
        from sidx.runner import seed_png
        for j in jobs:
            job_dir = args.out_root / experiment / f"{j.name}__{j.cfg_hash}"
            pending = sum(1 for im in j.images for s in j.seeds if not seed_png(job_dir / im.id, s).exists())
            print(f"{j.name:30s} {j.method:16s} hash={j.cfg_hash} images={len(j.images)} "
                  f"seeds={len(j.seeds)} pending={pending}")
        return 0

    repo_paths = {"project": str(PROJECT_ROOT)}
    if "SID_ROOT" in variables:
        repo_paths["efficient_sid"] = variables["SID_ROOT"]
    summary = run_jobs(experiment, jobs, args.out_root, budget_min=args.budget_min,
                       max_samples=args.max_samples, repo_paths=repo_paths)
    print(json.dumps(summary.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
