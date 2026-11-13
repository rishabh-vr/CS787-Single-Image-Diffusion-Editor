"""Our core vs. the official sampler on one image, same seeds, fp32. Writes a JSON report.

Default = the paper's Table-1 preset at the image's native size (patch 15, 4 scales, T=10, eta=0,
rho=0.2). Runs on CPU (slow: minutes per sample at ~250 px) or CUDA.

    python scripts/check_equivalence.py --image <png> --seeds 10 11 --out tables/equivalence_balloons.json
    python scripts/check_equivalence.py --image <png> --eta 1.0 --num-steps 40 ...

The official path is assembled from the official modules (sidx.core.official_cpu); our path is
sidx.core. Both use the chunked exact kernel, which is mathematically the dense one.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch

from sidx.core import CoreConfig, CoreSampler
from sidx.core.official_cpu import OfficialSampler
from sidx.core.sampler import load_image
from sidx.utils.runtime import atomic_write_json, environment_info, save_png_atomic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10])
    ap.add_argument("--out", required=True, help="JSON report path (sample PNGs are written next to it)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch-size", type=int, default=15)
    ap.add_argument("--num-scales", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--fold-rho", type=float, default=0.2)
    ap.add_argument("--query-chunks", type=int, default=16)
    ap.add_argument("--sid-root", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

    cfg = CoreConfig(patch_size=args.patch_size, num_scales=args.num_scales, num_steps=args.num_steps,
                     eta=args.eta, fold_rho=args.fold_rho, backend="chunked", query_chunks=args.query_chunks)
    image = load_image(args.image)
    out = Path(args.out)
    png_dir = out.with_suffix("")
    report = {"image": str(args.image), "image_hw": list(image.shape[-2:]), "config": asdict(cfg),
              "device": args.device, "env": environment_info({"efficient-SID-code": args.sid_root or "external/efficient-SID-code"}),
              "seeds": {}}
    ours = CoreSampler(image, cfg, args.device)
    official = OfficialSampler(image, cfg, args.device, sid_root=args.sid_root)
    for seed in args.seeds:
        t0 = time.time()
        a = ours.sample(seed).cpu()
        t1 = time.time()
        b = official.sample(seed).cpu()
        t2 = time.time()
        d = (a - b).abs().flatten() * 255
        r = {"max_grey": round(float(d.max()), 4), "p99_9_grey": round(float(d.quantile(0.999)), 4),
             "mean_grey": round(float(d.mean()), 6),
             "frac_pixels_over_1_grey": float((d > 1).float().mean()),
             "seconds_ours": round(t1 - t0, 1), "seconds_official": round(t2 - t1, 1)}
        report["seeds"][str(seed)] = r
        save_png_atomic(a, png_dir / f"ours_seed_{seed:04d}.png")
        save_png_atomic(b, png_dir / f"official_seed_{seed:04d}.png")
        print(f"seed {seed}: {r}", flush=True)
        atomic_write_json(out, report)
    worst = max(v["max_grey"] for v in report["seeds"].values())
    report["verdict"] = ("equivalent (all pixels within 1 grey level)" if worst <= 1.0 else
                         "equivalent up to isolated rounding flips" if
                         max(v["frac_pixels_over_1_grey"] for v in report["seeds"].values()) < 1e-3
                         else "NOT equivalent")
    atomic_write_json(out, report)
    print(report["verdict"])


if __name__ == "__main__":
    main()
