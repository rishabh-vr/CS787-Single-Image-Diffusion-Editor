"""Build qualitative grids for every job of an experiment (and optional factor-effect plots).

    python scripts/make_figures.py grids --exp-dir <out_root>/<experiment> --eval-root <E> --manifest data/manifests/x.csv ... --out reports/figures/<name>
    python scripts/make_figures.py factors --table reports/tables/track2_setB_factor_effects.csv --out reports/figures/setB_factors.png
"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sidx.viz import grids_for_job  # noqa: E402


def cmd_grids(a):
    paths = {}
    for m in a.manifest:
        with open(m, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = Path(r["path"])
                paths[r["id"]] = str(p if p.is_absolute() else (Path(m).parent / p).resolve())
    exp = Path(a.exp_dir)
    for job_json in sorted(exp.glob("*/job.json")):
        job_dir = job_json.parent
        ps = Path(a.eval_root) / exp.name / job_dir.name / "per_sample.csv" if a.eval_root else None
        for p in grids_for_job(job_dir, ps, paths, Path(a.out), n_fixed=a.fixed):
            print(p)


def cmd_factors(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    t = pd.read_csv(a.table)
    mets = [m for m in a.metrics if m in t]
    factors = list(dict.fromkeys(t["factor"]))
    fig, axs = plt.subplots(len(mets), len(factors), figsize=(2.6 * len(factors), 2.2 * len(mets)), squeeze=False)
    for i, m in enumerate(mets):
        for j, f in enumerate(factors):
            g = t[t["factor"] == f]
            axs[i][j].bar(range(len(g)), g[m], color="#4c78a8")
            axs[i][j].set_xticks(range(len(g)), [str(x).replace(" (base)", "*") for x in g["level"]], fontsize=7, rotation=30)
            if j == 0: axs[i][j].set_ylabel(m, fontsize=8)
            if i == 0: axs[i][j].set_title(f, fontsize=9)
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=140)
    print(a.out)


ap = argparse.ArgumentParser(description=__doc__)
sub = ap.add_subparsers(dest="cmd", required=True)
g = sub.add_parser("grids"); g.add_argument("--exp-dir", required=True); g.add_argument("--eval-root")
g.add_argument("--manifest", action="append", default=[]); g.add_argument("--out", required=True); g.add_argument("--fixed", type=int, default=6)
f = sub.add_parser("factors"); f.add_argument("--table", required=True); f.add_argument("--out", required=True)
f.add_argument("--metrics", nargs="+", default=["struct_dev", "endpoint_excess", "mean_seg_len_ratio", "grey_ratio", "patch_precision"])
a = ap.parse_args()
{"grids": cmd_grids, "factors": cmd_factors}[a.cmd](a)
