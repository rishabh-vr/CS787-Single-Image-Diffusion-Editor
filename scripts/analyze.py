"""Aggregate evaluation tables into the report's tables (Phase 10). Reads <eval-root>/<experiment>/<job>/per_image.csv
(written by scripts/evaluate.py); writes CSV + markdown to --out (default reports/tables/).

    python scripts/analyze.py table1    --eval-root E                 # Track 1: paper Table 1 side by side
    python scripts/analyze.py ood       --eval-root E                 # Track 2: categories, factors, descriptor correlations
    python scripts/analyze.py ablations --eval-root E                 # Phase 9: paired deltas vs the baseline job
    python scripts/analyze.py baselines --eval-root E [--cost-dir D]  # Track 3: ours vs SinDDM vs LoRA, per image + cost
    python scripts/analyze.py pilot     --eval-root E                 # laptop pilot (Set B64) factor effects
Every command prints the tables it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidx import analysis as A  # noqa: E402
from sidx.metrics.paper import PAPER_TABLE1  # noqa: E402

DESCRIPTORS = ["edge_density", "ink_fraction", "line_width_px", "periodicity_score", "patch_code_entropy",
               "rare_patch_fraction"]


def manifests(names):
    frames = []
    for n in names:
        p = ROOT / "data" / "manifests" / f"{n}.csv"
        if p.exists():
            frames.append(pd.read_csv(p).rename(columns={"id": "image_id"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["image_id"])


def emit(df: pd.DataFrame, name: str, out: Path, title: str = "", floatfmt: str = ".3f") -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    md = (f"### {title}\n\n" if title else "") + A.to_markdown(df, floatfmt) + "\n"
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    print(f"\n== {name} ==\n{md}")


def with_meta(df, names):
    m = manifests(names)
    keep = ["image_id"] + [c for c in m.columns if c not in df.columns and c != "image_id"]
    return df.merge(m[keep].drop_duplicates("image_id"), on="image_id", how="left")


def cmd_table1(a):
    df = A.load_eval(a.eval_root, "phase4_reproduction")
    cols = ["sifid", "niqe", "nima", "musiq", "pixel_div", "lpips_div"]
    rows = []
    for job, g in df.groupby("job"):
        r = {"method": job, "n_images": len(g)}
        for c in cols:
            r[c] = A.fmt_mean_std(g[c]) if c in g else "n/a"
        rows.append(r)
    for name in ("proposed (T=10, eta=0)", "proposed (T=40, eta=1)", "proposed (T=10, eta=0, k=5)", "SinDDM"):
        rows.append({"method": f"PAPER Table 1: {name}", "n_images": 15, **{c: f"{PAPER_TABLE1[name][c]}" for c in cols}})
    emit(pd.DataFrame(rows), "track1_table1", a.out, "Track 1 — our runs vs the paper's Table 1 (mean±std over images)")
    near = [c for c in df.columns if c.startswith("near_dup_r")]
    if near:
        emit(A.summarize(df, ["job"], near)[["job"] + near], "track1_near_duplicates", a.out,
             "Near-duplicate rate (Table S1 protocol): fraction of samples with >= r% near-identical pixels")


def cmd_ood(a):
    df = A.load_eval(a.eval_root, "phase7_ood")
    df = with_meta(df, ["ood_setA", "synthetic_setB"])
    df["category"] = df["category"].fillna("unknown")
    mets = ["patch_swd", "patch_precision", "patch_recall", "block_copy_ratio", "struct_dev", "endpoint_excess",
            "mean_seg_len_ratio", "grey_ratio", "edge_density_ratio", "junction_hist_l1", "spectral_peak_recall",
            "lpips_div", "sifid"]
    mets = [m for m in mets if m in df]
    setA = df[df["job"].str.startswith("setA")]
    emit(A.summarize(setA, ["category"], mets), "track2_setA_by_category", a.out,
         "Track 2, Set A — structural and distribution metrics by category (mean, 95% CI over images)")
    nat = None
    try:
        nat = A.load_eval(a.eval_root, "phase4_reproduction", ["official_paper_fp32"])
        nat["category"] = "natural (sinddm15)"
        both = pd.concat([setA, nat], ignore_index=True)
        emit(A.summarize(both, ["category"], mets), "track2_vs_natural", a.out,
             "Track 2 — OOD categories against the natural-image reference")
    except FileNotFoundError:
        pass
    B = df[df["job"].str.startswith("setB")]
    if len(B):
        fcols = ["thickness", "density", "length", "periodicity", "junctions", "text"]
        emit(A.factor_effects(B, fcols, mets), "track2_setB_factor_effects", a.out,
             "Track 2, Set B — one-factor-at-a-time effects (mean over images)")
    corr = A.spearman_table(setA.dropna(subset=["category"]), DESCRIPTORS, mets)
    if len(corr):
        emit(corr, "track2_setA_descriptor_correlations", a.out,
             "Spearman correlation between image descriptors and metrics (Set A, n = images)")


def cmd_ablations(a):
    df = A.load_eval(a.eval_root, "phase9_ablations")
    mets = [m for m in ["struct_dev", "endpoint_excess", "mean_seg_len_ratio", "grey_ratio", "patch_swd",
                        "patch_precision", "patch_recall", "block_copy_ratio", "lpips_div", "sifid"] if m in df]
    d = A.paired_delta(df, "baseline_p15_rho0.2_T10_eta0", mets)
    d["ablation"] = d["job"].str.extract(r"^(ab\d)")[0]
    emit(d[["ablation", "job", "metric", "n_images", "baseline", "delta_mean", "ci_lo", "ci_hi", "frac_increase"]],
         "phase9_ablation_deltas", a.out, "Ablations — paired difference to the baseline setting (mean over matched images, 95% CI)")


def cmd_baselines(a):
    frames = []
    for exp in ("phase4_reproduction", "phase8_ours", "phase8_sinddm", "phase8_lora"):
        try:
            f = A.load_eval(a.eval_root, exp)
            f["experiment"] = exp
            frames.append(f)
        except FileNotFoundError:
            pass
    df = pd.concat(frames, ignore_index=True)
    mets = [m for m in ["sifid", "patch_swd", "patch_precision", "patch_recall", "struct_dev", "block_copy_ratio",
                        "lpips_div", "pixel_div", "niqe", "nima", "musiq"] if m in df]
    df["method"] = df["job"].map(lambda j: "SinDDM" if j.startswith("sinddm") else "LoRA-SD1.5" if j.startswith("lora")
                                 else "ours (training-free)")
    keep = df[(df["method"] != "ours (training-free)") | df["job"].isin(
        ["official_paper_fp32", "ours_sinddm_ood3_128px", "ours_lora8_ood5"])]
    emit(A.summarize(keep, ["method", "job"], mets), "track3_methods", a.out,
         "Track 3 — methods compared (mean, 95% CI over images; ours uses the paper preset)")
    emit(keep[["method", "job", "image_id"] + mets].sort_values(["image_id", "method", "job"]), "track3_per_image", a.out,
         "Track 3 — per image")
    if a.cost_dir:
        cd = Path(a.cost_dir)
        rows = []
        for f in cd.glob("phase8_*cost*.json"):
            rows.append({"file": f.name, **{"images": len(json.load(open(f)))}})
        if rows:
            emit(pd.DataFrame(rows), "track3_cost_files", a.out, "Cost files found")


def cmd_pilot(a):
    df = A.load_eval(a.eval_root, "phase7_pilot")
    df = with_meta(df, ["synthetic_setB64"])
    mets = [m for m in ["struct_dev", "endpoint_excess", "mean_seg_len_ratio", "grey_ratio", "edge_density_ratio",
                        "junction_hist_l1", "orientation_conformity", "spectral_peak_recall", "patch_swd",
                        "patch_precision", "patch_recall", "block_copy_ratio", "lpips_div"] if m in df]
    emit(A.factor_effects(df, [], mets), "pilot_setB64_factor_effects", a.out,
         "PRELIMINARY (laptop pilot, 64 px, 8 samples per image) — one-factor effects on the paper method (our core)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["table1", "ood", "ablations", "baselines", "pilot"])
    ap.add_argument("--eval-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "tables")
    ap.add_argument("--cost-dir", default=None)
    a = ap.parse_args()
    {"table1": cmd_table1, "ood": cmd_ood, "ablations": cmd_ablations, "baselines": cmd_baselines, "pilot": cmd_pilot}[a.command](a)


if __name__ == "__main__":
    main()
