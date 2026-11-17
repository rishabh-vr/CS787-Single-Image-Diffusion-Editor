"""Sample N images from a SinDDM checkpoint and store them in the runner's output layout, so that
scripts/evaluate.py treats SinDDM exactly like every other method.

    python scripts/sinddm_collect.py --sinddm-root /content/SinDDM --scope sd_b0 \
        --dataset-folder /content/sinddm_data/sd_b0/ --image-name sd_b0.png \
        --results-folder /content/drive/MyDrive/CS787/checkpoints/sinddm/sd_b0/ --milestone last \
        --out-root /content/drive/MyDrive/CS787/outputs --job sinddm_steps20000 --image-id sd_b0 \
        --image-path data/sinddm/sd_b0.png --n 30

Output: <out-root>/phase8_sinddm/<job>__sinddm/<image_id>/seed_00NN.png  (+ job.json, samples.jsonl).
SinDDM's `--mode sample` writes its unbatched samples to a fixed file name that the next call
overwrites, so this script samples in batches (one seed per batch), moves each file out, and names
the samples seed0 + k. Resumable: samples that already exist are skipped.

--milestone: an integer (model-<k>.pt, written every --save_and_sample_every steps), "last"
(model-last.pt, the most recent checkpoint) or "pretrained" (the official model-12.pt).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sinddm-root", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--dataset-folder", required=True)
    ap.add_argument("--image-name", required=True)
    ap.add_argument("--results-folder", required=True)
    ap.add_argument("--milestone", default="last")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--image-path", required=True, help="the input image (recorded in job.json for evaluation)")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from PIL import Image  # noqa: F401  (fail early if missing)

    job_dir = Path(args.out_root) / "phase8_sinddm" / f"{args.job}__sinddm"
    img_dir = job_dir / args.image_id
    img_dir.mkdir(parents=True, exist_ok=True)
    jj = job_dir / "job.json"           # one job folder holds several images: add this one to the list
    meta = json.loads(jj.read_text()) if jj.exists() else {
        "experiment": "phase8_sinddm", "job": args.job, "method": "sinddm",
        "params": {"milestone": args.milestone, "results_folder": args.results_folder}, "images": []}
    if not any(im["id"] == args.image_id for im in meta["images"]):
        meta["images"].append({"id": args.image_id, "path": args.image_path})
        jj.write_text(json.dumps(meta, indent=2))

    milestone_arg = {"last": "last", "pretrained": "12"}.get(args.milestone, args.milestone)
    todo = [k for k in range(args.n) if not (img_dir / f"seed_{args.seed0 + k:04d}.png").exists()]
    t0 = time.time()
    for b0 in range(0, len(todo), args.batch):
        ks = todo[b0:b0 + args.batch]
        seed = args.seed0 + ks[0]
        cmd = [sys.executable, "main.py", "--scope", args.scope, "--mode", "sample",
               "--dataset_folder", args.dataset_folder, "--image_name", args.image_name,
               "--results_folder", args.results_folder, "--sample_batch_size", str(len(ks)),
               "--seed", str(seed), "--device", args.device]
        if milestone_arg == "last":
            cmd += ["--load_milestone", "last"]
        else:
            cmd += ["--load_milestone", milestone_arg]
        r = subprocess.run(cmd, cwd=args.sinddm_root, capture_output=True, text=True)
        unb = sorted(glob.glob(os.path.join(args.results_folder, args.scope, "final_samples_unbatched_*", "*_out_b*.png")),
                     key=os.path.getmtime)
        files = sorted(unb[-len(ks):], key=lambda p: int(p.rsplit("_out_b", 1)[1].split(".")[0]))
        if r.returncode != 0 or len(files) != len(ks):
            print(r.stdout[-1500:], r.stderr[-2500:])
            raise SystemExit(f"sampling failed (exit {r.returncode}, got {len(files)}/{len(ks)} files)")
        for k, f in zip(ks, files):
            shutil.move(f, img_dir / f"seed_{args.seed0 + k:04d}.png")
        print(f"[collect] {args.job}/{args.image_id}: {len(ks)} samples from seed {seed} "
              f"({time.time() - t0:.0f}s so far)", flush=True)
    print(f"[collect] {args.job}/{args.image_id}: {len(list(img_dir.glob('seed_*.png')))} samples in {img_dir}")


if __name__ == "__main__":
    main()
