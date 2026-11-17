"""Curated figures for the final report and slides (Phase 10).

    python scripts/report_figures.py --results kaggle_results/unpacked --out reports/figures/final

--results is a folder holding outputs/ (samples) and eval/ (scripts/evaluate.py tables), i.e. the Kaggle zips
K10-K14 + the Colab Phase 4 zip extracted on top of each other. Every figure is a PNG (200 dpi).
Colours: categorical slots 1-3 of the validated default palette (dataviz skill), one per method, fixed order.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OURS, SINDDM, LORA = "#2a78d6", "#eb6834", "#1baf7a"      # palette slots 1-3 (validated, light mode)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


# ------------------------------------------------------------------ helpers
def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def seeds(job_dir, image_id, k, skip=0):
    return sorted(glob.glob(f"{job_dir}/{image_id}/seed_*.png"))[skip:skip + k]


def job(res, exp, prefix):
    hits = sorted(glob.glob(f"{res}/outputs/{exp}/{prefix}__*"))
    if not hits:
        raise FileNotFoundError(f"{exp}/{prefix}")
    return hits[0]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def image_grid(rows, col_titles, out, name, cell=1.35, row_label_w=1.1, title_colors=None):
    """rows: [(row_label, [image paths or arrays or None])]; one column per entry."""
    nr, nc = len(rows), max(len(r[1]) for r in rows)
    ar = []
    for _, ims in rows:
        a = [load(x) if isinstance(x, (str, Path)) else x for x in ims if x is not None]
        ar.append(np.mean([i.shape[0] / i.shape[1] for i in a]) if a else 1)
    fig = plt.figure(figsize=(row_label_w + nc * cell, sum(cell * r for r in ar) + 0.35))
    gs = fig.add_gridspec(nr, nc + 1, width_ratios=[row_label_w / cell] + [1] * nc, height_ratios=ar,
                          wspace=0.04, hspace=0.06)
    for r, (label, ims) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off")
        ax.text(0.95, 0.5, label, ha="right", va="center", fontsize=8, color=INK, transform=ax.transAxes)
        for c, im in enumerate(ims):
            ax = fig.add_subplot(gs[r, c + 1]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if im is None:
                continue
            ax.imshow(load(im) if isinstance(im, (str, Path)) else im, interpolation="lanczos")
            if r == 0 and col_titles[c]:
                ax.set_title(col_titles[c], fontsize=8, color=(title_colors or {}).get(c, INK), pad=3)
    save(fig, out, name)


def per_image(res, exp, prefix=None):
    frames = []
    for f in sorted(glob.glob(f"{res}/eval/{exp}/*/per_image.csv")):
        d = pd.read_csv(f)
        d["job_dir"] = Path(f).parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if prefix:
        df = df[df["job_dir"].str.startswith(prefix)]
    return df


# ------------------------------------------------------------------ figures
def fig_baselines_sinddm(res, out):
    ours = job(res, "phase8_ours", "ours_sinddm_ood3_128px")
    sd = job(res, "phase8_sinddm", "sinddm_steps3000")
    names = {"sd_b0": "grid", "sd_fp_house2": "floor plan", "sd_pat_p4": "p4 pattern"}
    rows = [(names[i], [ROOT / f"data/sinddm/{i}.png"] + seeds(ours, i, 3) + seeds(sd, i, 3)) for i in names]
    image_grid(rows, ["input", "ours", "", "", "SinDDM 3k steps", "", ""], out, "fig_baselines_sinddm",
               title_colors={1: OURS, 4: SINDDM})


def fig_baselines_lora(res, out):
    ours = job(res, "phase8_ours", "ours_lora8_ood5")
    lo = job(res, "phase8_lora", "lora_r8_s400")
    names = {"fp_house2": "floor plan", "fp_layout": "floor plan 2", "pat_p4": "p4 pattern", "pat_cmm": "cmm pattern",
             "ckt_24aa": "circuit"}
    rows = [(names[i], [ROOT / f"data/ood/setA/{i}.png"] + seeds(ours, i, 3) + seeds(lo, i, 3)) for i in names]
    image_grid(rows, ["input", "ours", "", "", "DreamBooth-LoRA", "", ""], out, "fig_baselines_lora",
               title_colors={1: OURS, 4: LORA})


def fig_natural(res, out):
    ours = job(res, "phase4_reproduction", "official_paper_fp32")
    sd = job(res, "phase8_sinddm", "sinddm_pretrained_full")
    nat = ROOT / "data/natural/uncond_sinddm15"
    rows = [(i, [nat / f"{i}.png"] + seeds(ours, i, 4)) for i in ("balloons", "eucalyptus", "aurora")]
    rows.append(("forest (ours)", [nat / "forest.png"] + seeds(ours, "forest", 4)))
    rows.append(("forest (SinDDM)", [nat / "forest.png"] + seeds(sd, "forest", 4)))
    image_grid(rows, ["input", "samples", "", "", ""], out, "fig_natural", cell=1.5, row_label_w=1.2)


def fig_ood_gallery(res, out):
    a = job(res, "phase7_ood", "setA_paper_fp32")
    pick = [("fp_layout", "floor plan"), ("fp_apartment", "floor plan"), ("pat_p4", "pattern p4"),
            ("pat_circlepack", "pattern"), ("ckt_555", "circuit"), ("dia_life", "diagram")]
    rows = [(lab, [ROOT / f"data/ood/setA/{i}.png"] + seeds(a, i, 4)) for i, lab in pick]
    image_grid(rows, ["input", "samples (6 seeds run, 4 shown)", "", "", ""], out, "fig_ood_gallery", cell=1.4)


def fig_applications(res, out):
    q = Path(res) / "outputs/qualitative"
    fig, axs = plt.subplots(2, 3, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 2, 2]})
    panels = [(ROOT / "data/ood/setA/fp_house2.png", "input"), (q / "retarget_fp_house2_wide.png", "retargeted to 2x width"),
              (q / "retarget_ckt_24aa_wide.png", "circuit retargeted (chip repeated)"),
              (q / "tileable_pat_p4_input.png", "input"), (q / "tileable_pat_p4_input_tiled_2x2.png", "input tiled 2x2: seams"),
              (q / "tileable_pat_p4_sample_tiled_2x2.png", "tileable sample tiled 2x2: seamless")]
    for ax, (p, t) in zip(axs.flat, panels):
        ax.imshow(load(p)); ax.set_title(t, fontsize=8, color=INK); ax.axis("off")
    save(fig, out, "fig_applications")


def fig_failures(res, out):
    q = Path(res) / "outputs/qualitative"
    a = job(res, "phase7_ood", "setA_paper_fp32")
    sd = job(res, "phase8_sinddm", "sinddm_steps3000")
    lo = job(res, "phase8_lora", "lora_r8_s400")
    panels = [(q / "retarget_dia_life_wide.png", "(a) text stutter under retargeting\n\"RESPOSPOKSPOND\", \"LIFE IFFFE\""),
              (seeds(a, "dia_life", 1, 1)[0], "(b) diagram: hub node duplicated,\nedges lead nowhere"),
              (seeds(a, "ckt_555", 1)[0], "(c) circuit: symbols kept, wiring\nand labels scrambled"),
              (seeds(a, "pat_circlepack", 1, 2)[0], "(d) pattern: lattice seam /\nbroken cells"),
              (seeds(sd, "sd_fp_house2", 1)[0], "(e) SinDDM, 1 GPU-hour:\nfloor plan almost blank"),
              (seeds(lo, "pat_cmm", 1, 1)[0], "(f) LoRA: palette and lattice\ndrift from the input")]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, (p, t) in zip(axs.flat, panels):
        ax.imshow(load(p)); ax.set_title(t, fontsize=8, color=INK, loc="left"); ax.axis("off")
    save(fig, out, "fig_failures")


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def fig_baseline_metrics(res, out):
    """Per-image SIFID / copy / diversity for ours vs the baselines on the SAME images (paired dots)."""
    o = per_image(res, "phase8_ours"); s = per_image(res, "phase8_sinddm", "sinddm_steps3000")
    l = per_image(res, "phase8_lora")
    sets = [("vs SinDDM (3 images, 128 px)", o[o.job_dir.str.startswith("ours_sinddm")], s, "SinDDM 3k", SINDDM),
            ("vs LoRA (5 OOD images)", o[o.job_dir.str.startswith("ours_lora")], l[~l.image_id.isin(["forest", "night_sky", "pyramids"])],
             "LoRA", LORA)]
    mets = [("sifid", "SIFID (lower = closer to input)"), ("patch_precision", "patch precision"),
            ("block_copy_ratio", "copied-block ratio"), ("lpips_div", "LPIPS diversity")]
    fig, axs = plt.subplots(2, 4, figsize=(10, 4.6))
    for r, (title, a, b, bname, bcol) in enumerate(sets):
        m = a.merge(b, on="image_id", suffixes=("_o", "_b"))
        for c, (k, lab) in enumerate(mets):
            ax = axs[r, c]
            for _, row in m.iterrows():
                ax.plot([0, 1], [row[f"{k}_o"], row[f"{k}_b"]], color=GRID, lw=1.5, zorder=1)
            ax.scatter(np.zeros(len(m)), m[f"{k}_o"], s=36, color=OURS, edgecolor="white", lw=1.5, zorder=3)
            ax.scatter(np.ones(len(m)), m[f"{k}_b"], s=36, color=bcol, edgecolor="white", lw=1.5, zorder=3)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["ours", bname]); ax.set_xlim(-0.4, 1.4)
            ax.grid(axis="x", visible=False)
            if k == "sifid":
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(lab, fontsize=8, color=INK)
        axs[r, 0].set_ylabel(title, fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig_baseline_metrics")


def fig_ood_metrics(res, out):
    a = per_image(res, "phase7_ood", "setA_paper")              # per_image.csv already carries 'category'
    n = per_image(res, "phase4_reproduction", "official_paper_fp32"); n["category"] = "natural"
    d = pd.concat([n, a], ignore_index=True)
    order = ["natural", "floorplan", "circuit", "diagram", "pattern"]
    mets = [("patch_recall", "patch recall\n(input patches reproduced)"), ("block_copy_ratio", "copied-block ratio"),
            ("endpoint_excess", "line endpoints vs input\n(1 = same; >1 = broken lines)"), ("lpips_div", "LPIPS diversity")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, (k, lab) in zip(axs, mets):
        g = d.groupby("category")[k]
        mu = [g.get_group(c).mean() for c in order]; ci = [_ci(g.get_group(c)) for c in order]
        cols = [MUTED if c == "natural" else OURS for c in order]
        ax.bar(range(len(order)), mu, yerr=ci, color=cols, width=0.62, error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2})
        if k == "endpoint_excess":
            ax.axhline(1, color=INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{c}\n(n={len(g.get_group(c))})" for c in order], fontsize=7, rotation=0)
        ax.set_title(lab, fontsize=8, color=INK); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig_ood_metrics")


def fig_ablations(res, out):
    d = per_image(res, "phase9_ablations")
    d["job"] = d["job_dir"].str.split("__").str[0]
    base = "baseline_p15_rho0.2_T10_eta0"
    groups = [("patch size", [("ab1_patch7", "7"), ("ab1_patch11", "11"), (base, "15*"), ("ab1_patch23", "23")]),
              ("kernel temperature rho", [("ab2_rho0.1", "0.1"), (base, "0.2*"), ("ab2_rho1.0", "1"), ("ab2_rhoinf", "inf")]),
              ("pyramid scales", [("ab4_scales2", "2"), ("ab4_scales3", "3"), (base, "4*"), ("ab4_scales5", "5†")]),
              ("patch stride", [(base, "1*"), ("ab7_stride2", "2"), ("ab7_stride4", "4")])]
    mets = [("sifid", "SIFID (log)"), ("block_copy_ratio", "copied-block ratio"), ("lpips_div", "LPIPS diversity")]
    fig, axs = plt.subplots(3, 4, figsize=(10, 6.4), sharey="row")
    for c, (gname, levels) in enumerate(groups):
        for r, (k, lab) in enumerate(mets):
            ax = axs[r, c]
            ims = sorted(d[d.job == base].image_id)
            for im in ims:                                  # one thin line per image, mean on top
                ys = [d[(d.job == j) & (d.image_id == im)][k].mean() for j, _ in levels]
                ax.plot(range(len(levels)), ys, color=GRID, lw=1.2, marker="o", ms=3, zorder=1)
            mu = [d[d.job == j][k].mean() for j, _ in levels]
            ax.plot(range(len(levels)), mu, color=OURS, lw=2, marker="o", ms=6, mec="white", mew=1.5, zorder=3)
            ax.set_xticks(range(len(levels))); ax.set_xticklabels([l for _, l in levels])
            ax.grid(axis="x", visible=False)
            if k == "sifid":
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(gname, fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.text(0.01, -0.01, "* = paper default (patch 15, rho 0.2, 4 scales, stride 1).  Thin grey lines: the 4 images "
             "(balloons, fp_layout, pat_p4, ckt_24aa); blue: mean.  † 5 scales runs only on 2 images "
             "(the 5th scale is smaller than a patch for the other two).", fontsize=7, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, out, "fig_ablations")


def fig_cost(res, out):
    """GPU time to get 20 samples of ONE new image, from scratch (T4)."""
    def mean_s(jdir):
        v = [json.loads(l)["seconds"] for f in glob.glob(f"{jdir}/*/samples.jsonl") for l in open(f) if l.strip()]
        return float(np.median(v))
    ours128 = mean_s(job(res, "phase8_ours", "ours_sinddm_ood3_128px"))
    ours256 = mean_s(job(res, "phase8_ours", "ours_lora8_ood5"))
    prog = [json.load(open(p)) for p in glob.glob(f"{res}/checkpoints/sinddm/*/progress.json")]
    sd_train = float(min(p["total"] / p["steps_per_s"] for p in prog))   # the run alone on its GPU (two others shared one)
    lc = json.load(open(f"{res}/tables/phase8_lora_cost.json"))
    lo_train = float(np.median([v["train_s"] for v in lc.values()])); lo_s = float(np.median([v["sample_s_per_img"] for v in lc.values()]))
    sd_s = 2.3                                            # K11 log: 120 samples (128 px) in 138 s on 2 T4s
    rows = [("ours, 128 px", 0, 20 * ours128, OURS), ("ours, <=256 px", 0, 20 * ours256, OURS),
            ("LoRA (SD 1.5, 512 px)", lo_train, 20 * lo_s, LORA), ("SinDDM 3k steps, 128 px", sd_train, 20 * sd_s, SINDDM)]
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    for y, (lab, tr, sa, col) in enumerate(rows):
        if tr:
            ax.barh(y, tr, color=col, alpha=0.45, height=0.55, edgecolor="white", lw=2)
        ax.barh(y, sa, left=tr, color=col, height=0.55, edgecolor="white", lw=2)
        tot = tr + sa
        txt = f"{tot / 60:.1f} min" if tot >= 60 else f"{tot:.0f} s"
        detail = f"  (training {tr / 60:.0f} min + sampling {sa:.0f} s)" if tr else "  (no training)"
        ax.text(tot * 1.03, y, txt + detail, va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(20, 4e4); ax.set_xlabel("GPU seconds on one T4 for 20 samples of a new image (log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("light segment = training, dark = sampling", fontsize=7, color=MUTED, loc="right")
    save(fig, out, "fig_cost")
    return {"ours128_s": ours128, "ours256_s": ours256, "sinddm_train_s": sd_train, "lora_train_s": lo_train, "lora_s": lo_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "kaggle_results/unpacked"))
    ap.add_argument("--out", default=str(ROOT / "reports/figures/final"))
    a = ap.parse_args()
    res, out = a.results, Path(a.out)
    for f in (fig_baselines_sinddm, fig_baselines_lora, fig_natural, fig_ood_gallery, fig_applications, fig_failures,
              fig_baseline_metrics, fig_ood_metrics, fig_ablations, fig_cost):
        r = f(res, out)
        if r:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
