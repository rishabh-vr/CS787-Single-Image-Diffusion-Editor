"""Failure-isolation experiments F2-F5 (plan §10) with our core sampler, which can log the winning input patch
and every finished scale (the official code cannot). CPU-friendly: 64 px images, 3 pyramid scales.

  F2  patch selection: do rare input patches get dropped?  Isolation of every input patch (distance to its
      5th-nearest NON-overlapping neighbour) vs how often the last denoising step copies it.
  F3  per-scale: at which pyramid scale does structure break?  Structural statistics of every scale's result
      against the input's own pyramid level.
  F4  blend ablation: with/without the two-scale Laplacian blend: layout consistency between scales and
      distribution/structure metrics.
  F5  background fraction: the same drawing padded with more white space.

    python scripts/failure_isolation.py --which f2 f3 f4 f5 --samples 4 --out reports/failure
Results: <out>/<f>.json (+ figures in <out>/figures when matplotlib is available). Small and committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidx.core import CoreConfig, CoreSampler  # noqa: E402
from sidx.core.hooks import Hooks, ScaleRecorder, WinnerRecorder  # noqa: E402
from sidx.core.pyramid import build_pyramid, pyramid_shapes  # noqa: E402
from sidx.core.resize import resize  # noqa: E402
from sidx.core.sampler import load_image  # noqa: E402
from sidx.data.synthetic import BASE, SynthSpec, render  # noqa: E402
from sidx.metrics import patchdist, structure  # noqa: E402

CFG = CoreConfig(num_scales=3, backend="chunked", query_chunks=2)
SETB64 = ROOT / "data" / "synthetic" / "setB64"


def tensor_of(pil) -> torch.Tensor:
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)


def nonoverlap_isolation(patches: torch.Tensor, hw, p: int, k: int = 5) -> torch.Tensor:
    """Distance of every patch to its k-th nearest patch at least p pixels away (larger = rarer)."""
    nh, nw = hw[0] - p + 1, hw[1] - p + 1
    d = torch.cdist(patches, patches)
    yy, xx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
    yy, xx = yy.reshape(-1), xx.reshape(-1)
    d[((yy[:, None] - yy[None]).abs() < p) & ((xx[:, None] - xx[None]).abs() < p)] = float("inf")
    return d.kthvalue(k, dim=1).values


# --------------------------------------------------------------------------------------------------
def f2_patch_selection(images, n_samples, log):
    from scipy.stats import spearmanr

    out = {}
    for name, img in images.items():
        s = CoreSampler(img, CFG, "cpu")
        data = s.denoisers[0].dataset                            # (N, D) in [-1, 1]
        iso = nonoverlap_isolation(data, s.shapes[0], CFG.patch_size).numpy()
        iso = np.where(np.isfinite(iso), iso, np.nanmax(iso[np.isfinite(iso)]))
        counts = np.zeros(len(data))
        n_out = 0
        for seed in range(10, 10 + n_samples):
            w = WinnerRecorder([1])
            s.sample(seed, hook=w, record_argmax=True)
            win = w.winners[(0, 1)].numpy()
            counts += np.bincount(win, minlength=len(data))
            n_out += len(win)
            log(f"  f2 {name} seed {seed}")
        share = counts / n_out                                    # how often each input patch is copied
        order = np.argsort(iso)
        rare, common = order[-len(order) // 10:], order[: len(order) // 2]
        out[name] = {
            "n_patches": int(len(data)), "samples": n_samples,
            "spearman_isolation_vs_selection": float(spearmanr(iso, counts)[0]),
            "never_selected_overall": float((counts == 0).mean()),
            "never_selected_rarest10pct": float((counts[rare] == 0).mean()),
            "never_selected_commonest50pct": float((counts[common] == 0).mean()),
            "mean_selection_share_rarest10pct_vs_uniform": float(share[rare].mean() * len(data)),
            "mean_selection_share_commonest50pct_vs_uniform": float(share[common].mean() * len(data)),
        }
        out[name]["_iso"], out[name]["_share"] = iso.tolist(), share.tolist()      # for the figure only
    return out


# --------------------------------------------------------------------------------------------------
def scale_struct(res: torch.Tensor, ref_level: torch.Tensor) -> dict:
    sref = structure.StructureReference(ref_level)
    m = sref.compare(res)
    return {k: float(m[k]) for k in ("endpoint_excess", "mean_seg_len_ratio", "grey_ratio", "edge_density_ratio")}


def f3_per_scale(images, n_samples, log):
    out = {}
    for name, img in images.items():
        s = CoreSampler(img, CFG, "cpu")
        shapes = s.shapes
        levels = build_pyramid(img, shapes)
        per = {sc: [] for sc in range(len(shapes))}
        for seed in range(10, 10 + n_samples):
            rec = ScaleRecorder()
            s.sample(seed, hook=rec)
            for sc in range(len(shapes) - 1):                    # the coarsest (16 px) is too small for LSD
                per[sc].append(scale_struct(rec.scales[sc], levels[sc]))
            log(f"  f3 {name} seed {seed}")
        out[name] = {f"scale{sc}_{shapes[sc][0]}px": {k: float(np.mean([r[k] for r in v])) for k in v[0]}
                     for sc, v in per.items() if v}
    return out


# --------------------------------------------------------------------------------------------------
def lowfreq_corr(final: torch.Tensor, coarse: torch.Tensor) -> float:
    """Correlation between the final image reduced to the coarse scale's size and that coarse result."""
    d = resize(final, coarse.shape[-2:])
    a, b = d.flatten() - d.mean(), coarse.flatten() - coarse.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def f4_blend(images, n_samples, log):
    out = {}
    for name, img in images.items():
        res = {}
        for blend in (True, False):
            s = CoreSampler(img, replace(CFG, blend=blend), "cpu")
            ref = StructRef = structure.StructureReference(img)
            rows = []
            for seed in range(10, 10 + n_samples):
                rec = ScaleRecorder()
                x = s.sample(seed, hook=rec)
                m = ref.compare(x)
                rows.append({"layout_corr_scale1": lowfreq_corr(x, rec.scales[1]),
                             "layout_corr_scale2": lowfreq_corr(x, rec.scales[2]),
                             "patch_swd": patchdist.patch_swd(img, x), **{
                                 k: float(m[k]) for k in ("endpoint_excess", "mean_seg_len_ratio", "grey_ratio")}})
                log(f"  f4 {name} blend={blend} seed {seed}")
            res["blend" if blend else "no_blend"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[name] = res
    return out


# --------------------------------------------------------------------------------------------------
def f5_background(n_samples, log):
    from PIL import Image

    out = {}
    for d in (64, 48, 40):
        pil = render(replace(BASE, size=d, seed=0))
        canvas = Image.new("RGB", (64, 64), "white")
        canvas.paste(pil, ((64 - d) // 2, (64 - d) // 2))
        img = tensor_of(canvas)
        bg = 1 - (d / 64) ** 2
        s = CoreSampler(img, CFG, "cpu")
        ref = structure.StructureReference(img)
        pr = patchdist.PatchPR(img)
        rows = []
        for seed in range(10, 10 + n_samples):
            x = s.sample(seed)
            m = ref.compare(x)
            ink_in, ink_out = float((img.mean(0) < 0.5).float().mean()), float((x.mean(0) < 0.5).float().mean())
            rows.append({"patch_precision": pr.add(x), "ink_ratio": ink_out / max(ink_in, 1e-9),
                         **{k: float(m[k]) for k in ("endpoint_excess", "mean_seg_len_ratio", "grey_ratio")}})
            log(f"  f5 drawing {d}px seed {seed}")
        out[f"drawing_{d}px_bg{bg:.2f}"] = {"background_fraction": round(bg, 3), "patch_recall": pr.recall(),
                                            **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}
    return out


# --------------------------------------------------------------------------------------------------
def figures(res, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if "f2" in res:
        names = list(res["f2"])
        fig, axs = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), squeeze=False)
        for ax, n in zip(axs[0], names):
            iso, share = np.array(res["f2"][n]["_iso"]), np.array(res["f2"][n]["_share"]) * res["f2"][n]["n_patches"]
            ax.scatter(iso, share, s=3, alpha=0.35)
            ax.axhline(1, color="k", lw=0.6, ls="--")
            ax.set_xlabel("isolation (dist. to 5th NN, non-overlapping)")
            ax.set_ylabel("selection share / uniform")
            ax.set_title(f"{n}  rho={res['f2'][n]['spearman_isolation_vs_selection']:.2f}", fontsize=9)
            ax.set_yscale("symlog", linthresh=1)
        fig.tight_layout()
        fig.savefig(out_dir / "f2_patch_selection.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="+", default=["f2", "f3", "f4", "f5"], choices=["f2", "f3", "f4", "f5"])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "failure")
    ap.add_argument("--images", nargs="+", default=["b0_s0", "thickness_1_s0", "text_sparse_s0"],
                    help="ids from data/synthetic/setB64 used by F2-F4")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    log = lambda m: print(m, flush=True)
    images = {i: load_image(str(SETB64 / f"{i}.png")) for i in a.images}
    res = {}
    for w in a.which:
        t0 = time.time()
        log(f"== {w}")
        res[w] = {"f2": lambda: f2_patch_selection(images, a.samples, log),
                  "f3": lambda: f3_per_scale(images, a.samples, log),
                  "f4": lambda: f4_blend(images, a.samples, log),
                  "f5": lambda: f5_background(a.samples, log)}[w]()
        slim = json.loads(json.dumps(res[w]))
        for v in slim.values():                                   # keep the committed JSON small
            if isinstance(v, dict):
                v.pop("_iso", None)
                v.pop("_share", None)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{w}.json").write_text(json.dumps({"samples": a.samples, "images": a.images,
                                                      "config": vars(CFG) if hasattr(CFG, "__dict__") else str(CFG),
                                                      "seconds": round(time.time() - t0), "result": slim}, indent=1))
        log(json.dumps(slim, indent=1)[:2500])
    figures(res, a.out / "figures")


if __name__ == "__main__":
    main()
