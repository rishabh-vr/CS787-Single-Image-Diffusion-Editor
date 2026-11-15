"""Aggregation and statistics over the evaluator's per_image.csv files (plan Phase 10).

Unit of analysis = the IMAGE: each image's metric is already the mean over its samples, and images
are what vary between conditions, so confidence intervals bootstrap over images (never over
samples of one image, which are not independent evidence about the method).

Directions ("better"):  see METRICS. For structural ratios the ideal is 1 (same statistics as the
input), so they are also reported as `|log2 ratio|` deviations, lower = better.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: metric -> (label, "up"/"down"/"one": which direction is better; "one" = ratio, ideal 1)
METRICS: Dict[str, tuple] = {
    "sifid": ("SIFID", "down"), "niqe": ("NIQE", "down"), "nima": ("NIMA", "up"), "musiq": ("MUSIQ", "up"),
    "pixel_div": ("Pixel div.", "up"), "lpips_div": ("LPIPS div.", "up"),
    "patch_swd": ("Patch SWD", "down"), "patch_precision": ("Patch precision", "up"),
    "patch_recall": ("Patch recall", "up"),
    "block_copy_ratio": ("Block-copy ratio", "down"), "mean_region_px": ("Copied-region size (px)", "down"),
    "near_dup_r85": ("Near-dup rate (r=85%)", "down"),
    "endpoint_excess": ("Endpoint excess", "one"), "mean_seg_len_ratio": ("Mean segment length ratio", "one"),
    "grey_ratio": ("Grey-level (ghosting) ratio", "one"), "edge_density_ratio": ("Edge density ratio", "one"),
    "grad_mag_ratio": ("Gradient magnitude ratio", "one"), "periodicity_ratio": ("Periodicity ratio", "one"),
    "junction_hist_l1": ("Junction histogram L1", "down"), "seg_len_w1_rel": ("Segment-length W1 (rel.)", "down"),
    "seg_angle_w1_deg": ("Segment-angle W1 (deg)", "down"), "spectral_peak_recall": ("Spectral peak recall", "up"),
    "orientation_conformity": ("Orientation conformity", "up"),
}
STRUCT_RATIOS = ["endpoint_excess", "mean_seg_len_ratio", "grey_ratio", "edge_density_ratio"]


def load_eval(eval_root: Path, experiment: str, jobs: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Concatenate per_image.csv of every job of an experiment (job name = folder name before '__')."""
    frames = []
    for d in sorted((Path(eval_root) / experiment).glob("*/per_image.csv")):
        job = d.parent.name.split("__")[0]
        if jobs is not None and job not in set(jobs):
            continue
        df = pd.read_csv(d)
        df["job"] = job
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no per_image.csv under {Path(eval_root) / experiment}")
    return add_derived(pd.concat(frames, ignore_index=True))


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """|log2 ratio| deviations and the composite structural deviation (mean of the four, lower = better)."""
    df = df.copy()
    for m in STRUCT_RATIOS + ["periodicity_ratio", "grad_mag_ratio"]:
        if m in df:
            v = df[m].astype(float).clip(lower=1e-3, upper=1e3)
            df[f"{m}_dev"] = np.abs(np.log2(v))
    devs = [f"{m}_dev" for m in STRUCT_RATIOS if f"{m}_dev" in df]
    if devs:
        df["struct_dev"] = df[devs].mean(axis=1, skipna=False)
    return df


def boot_ci(x: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    if len(x) == 1:
        return (float(x[0]),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(df: pd.DataFrame, by: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """mean [95% bootstrap CI over images] and n images, per group."""
    rows = []
    for key, g in df.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n_images"] = len(g)
        for m in metrics:
            if m in g:
                mean, lo, hi = boot_ci(g[m].values)
                row[m] = mean
                row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_delta(df: pd.DataFrame, baseline_job: str, metrics: Sequence[str], job_col: str = "job",
                 key: str = "image_id") -> pd.DataFrame:
    """For every non-baseline job: mean over matched images of (job - baseline), with a bootstrap CI
    over images and a sign-test-style win fraction. Images missing in either job are dropped."""
    base = df[df[job_col] == baseline_job].set_index(key)
    rows = []
    for job, g in df[df[job_col] != baseline_job].groupby(job_col):
        g = g.set_index(key)
        common = g.index.intersection(base.index)
        for m in metrics:
            if m not in df:
                continue
            d = (g.loc[common, m] - base.loc[common, m]).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            mean, lo, hi = boot_ci(d.values)
            rows.append({"job": job, "metric": m, "n_images": len(d), "delta_mean": mean, "ci_lo": lo, "ci_hi": hi,
                         "frac_increase": float((d > 0).mean()) if len(d) else float("nan"),
                         "baseline": float(base.loc[common, m].mean()) if len(common) else float("nan")})
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame, predictors: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Spearman rho and p between image descriptors and metrics (rank-based, robust to scale)."""
    from scipy.stats import spearmanr

    rows = []
    for p in predictors:
        for m in metrics:
            if p not in df or m not in df:
                continue
            d = df[[p, m]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 6 or d[p].nunique() < 3 or d[m].nunique() < 3:
                continue
            rho, pv = spearmanr(d[p], d[m])
            rows.append({"predictor": p, "metric": m, "n": len(d), "rho": float(rho), "p": float(pv)})
    out = pd.DataFrame(rows)
    return out.sort_values("p") if len(out) else out


def factor_effects(df: pd.DataFrame, factor_cols: Sequence[str], metrics: Sequence[str],
                   base_label: str = "base") -> pd.DataFrame:
    """One-factor-at-a-time table: for each factor and level, the mean of each metric over the
    images where only that factor differs from the base (rows with factor == that factor), plus
    the base itself. Expects the manifest's `factor` column and the factor values as columns."""
    rows = []
    base = df[df["factor"] == base_label]
    for f in df["factor"].dropna().unique():
        if f == base_label:
            continue
        sub = df[df["factor"] == f]
        levels = sorted(set(sub[f].astype(str))) if f in sub else []
        for lv in levels:
            g = sub[sub[f].astype(str) == lv]
            row = {"factor": f, "level": lv, "n_images": len(g)}
            for m in metrics:
                if m in g:
                    row[m] = float(g[m].mean())
            rows.append(row)
        # the base level of this factor is the shared base image
        if f in base:
            row = {"factor": f, "level": str(base[f].iloc[0]) + " (base)", "n_images": len(base)}
            for m in metrics:
                if m in base:
                    row[m] = float(base[m].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_mean_std(x: Sequence[float], digits: int = 2) -> str:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    return "n/a" if len(x) == 0 else f"{x.mean():.{digits}f}±{x.std(ddof=0):.{digits}f}"


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Small dependency-free GitHub markdown table."""
    cols = list(df.columns)
    def cell(v):
        if isinstance(v, float):
            return "n/a" if not math.isfinite(v) else format(v, floatfmt)
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(out)
