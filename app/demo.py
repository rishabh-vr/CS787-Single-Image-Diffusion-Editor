"""Interactive prototype: training-free single-image diffusion (Qiu, Kutulakos & Lindell, 2026).

Run from the repository root:

    streamlit run app/demo.py

Everything runs on the project's own re-implementation (src/sidx/core), on CPU or a CUDA GPU.
No training, no pretrained weights: the "model" is the input image's own patches.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                     # noqa: E402
import streamlit as st                 # noqa: E402
import torch                           # noqa: E402
from PIL import Image                  # noqa: E402

from engine import (DemoSampler, auto_num_scales, copying_metrics, fit_long_side,  # noqa: E402
                    generate, pil_to_tensor, position_colours, tensor_to_pil)
from sidx.core import CoreConfig       # noqa: E402
from sidx.core.pyramid import pyramid_shapes  # noqa: E402

st.set_page_config(page_title="Single-Image Diffusion Demo", layout="wide")

EXAMPLE_DIRS = {
    "Floor plans, circuits, diagrams, patterns (OOD set A)": ROOT / "data/ood/setA",
    "Synthetic line drawings (OOD set B)": ROOT / "data/synthetic/setB",
    "Natural images (paper benchmark)": ROOT / "data/natural/uncond_sinddm15",
}
SPEED = {
    "Fast · patch stride 3": 3,
    "Balanced · patch stride 2": 2,
    "Exact paper preset · stride 1 (slow on CPU)": 1,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def examples():
    out = {}
    for group, d in EXAMPLE_DIRS.items():
        files = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")) if d.exists() else []
        if files:
            out[group] = files
    return out


def show(img, caption=None, scale_to=None):
    """Display a tensor / array / PIL image, enlarged with nearest-neighbour so pixels stay crisp."""
    if isinstance(img, torch.Tensor):
        img = tensor_to_pil(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if scale_to and img.width < scale_to:
        k = max(1, scale_to // img.width)
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    st.image(img, caption=caption, width="stretch")


# =============================================================================== sidebar
st.sidebar.title("Settings")

st.sidebar.subheader("1 · Input image")
ex = examples()
source = st.sidebar.radio("Source", (["Example"] if ex else []) + ["Upload"], horizontal=True)
if source == "Example":
    group = st.sidebar.selectbox("Set", list(ex))
    path = st.sidebar.selectbox("Image", ex[group], format_func=lambda p: p.stem)
    pil = Image.open(path)
    name = path.stem
else:
    up = st.sidebar.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg", "webp"])
    if up is None:
        st.info("Upload an image in the sidebar to begin.")
        st.stop()
    pil = Image.open(up)
    name = Path(up.name).stem

long_side = st.sidebar.slider("Working size (long side, px)", 96, 256, 160, 8,
                              help="The paper works at about 250 px. Smaller is faster on CPU.")

st.sidebar.subheader("2 · Speed")
speed = st.sidebar.radio("Denoiser evaluation", list(SPEED), index=1 if DEVICE == "cpu" else 2,
                         help="Stride s evaluates the closed-form denoiser on every s-th patch "
                              "(query and dataset), about s⁴ times less work. Stride 1 is the "
                              "paper's exact setting.")
stride = SPEED[speed]
st.sidebar.caption(f"Running on **{DEVICE.upper()}**"
                   + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else
                      f" · {torch.get_num_threads()} threads"))

st.sidebar.subheader("3 · Output")
n_samples = st.sidebar.slider("Number of samples", 1, 8, 3)
seed0 = st.sidebar.number_input("First seed", 0, 10_000, 0)
c1, c2 = st.sidebar.columns(2)
w_mult = c1.select_slider("Width ×", [0.75, 1.0, 1.25, 1.5, 2.0], 1.0)
h_mult = c2.select_slider("Height ×", [0.75, 1.0, 1.25, 1.5, 2.0], 1.0)

with st.sidebar.expander("Advanced (method hyper-parameters)"):
    patch = st.select_slider("Patch size", [7, 9, 11, 13, 15, 17, 21], 15,
                             help="Paper: 15. Larger patches keep more structure but copy more.")
    scales_opt = st.selectbox("Pyramid scales", ["auto", 1, 2, 3, 4, 5], 0,
                              help="Paper: 4 at ~250 px. 'auto' keeps the coarsest level ≥ 1.5 patches.")
    steps = st.slider("Diffusion steps per scale (T)", 2, 20, 10)
    eta = st.slider("η (0 = deterministic DDIM, 1 = DDPM noise)", 0.0, 1.0, 0.0, 0.1)
    rho = st.slider("Fold window ρ (Gaussian patch blending)", 0.1, 1.0, 0.2, 0.05)
    blend = st.checkbox("Two-scale Laplacian blend", True,
                        help="Off = ablation: each scale ignores the coarser result's low frequencies.")

# =============================================================================== prepare
x = fit_long_side(pil_to_tensor(pil), long_side)
H, W = x.shape[-2:]
n_scales = auto_num_scales((H, W), patch) if scales_opt == "auto" else int(scales_opt)
out_hw = (max(patch * 2, round(H * h_mult)), max(patch * 2, round(W * w_mult)))
shapes = pyramid_shapes((H, W), n_scales)
problem = None
if min(shapes[-1]) < patch:
    problem = f"With {n_scales} scales the coarsest level is {shapes[-1][0]}×{shapes[-1][1]} px, smaller " \
              f"than the {patch}×{patch} patch. Use fewer scales, a smaller patch or a larger working size."
elif min(pyramid_shapes(out_hw, n_scales)[-1]) < patch:
    problem = "The output size is too small for this many scales."

cfg_key = (name, long_side, stride, patch, n_scales, steps, eta, rho, blend, pil.size)

# =============================================================================== main
st.title("Training-free single-image diffusion")
st.markdown("Generate new variations of **one image**, with **no training and no neural network**. "
            "Each denoising step is a closed-form weighted average of the image's own patches. "
            "Reproduces Qiu, Kutulakos & Lindell (2026) · CS787 project, Rishabh Verma & Divyansh Gupta.")

tab_gen, tab_how = st.tabs(["Generate", "How it works"])

with tab_gen:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Input")
        show(x, f"{name} · working size {W}×{H}", scale_to=320)
        st.caption(f"Pyramid (fine → coarse): " + " → ".join(f"{w}×{h}" for h, w in shapes)
                   + f" · {steps} steps per scale · output {out_hw[1]}×{out_hw[0]}")
        go = st.button("Generate", type="primary", width="stretch", disabled=problem is not None)
        if problem:
            st.error(problem)

    if go:
        cfg = CoreConfig(patch_size=patch, num_scales=n_scales, num_steps=steps, eta=eta,
                         fold_rho=rho, blend=blend, backend="dense" if DEVICE == "cpu" else "sdpa",
                         query_chunks=1 if DEVICE == "cpu" else 4)
        sampler = DemoSampler(x, cfg, DEVICE, stride)
        bar = right.progress(0.0, text="Starting…")
        total = n_samples * n_scales * steps
        done = [0]
        results = []
        for i in range(n_samples):
            def on_step(s, t, i=i):
                done[0] += 1
                bar.progress(done[0] / total,
                             text=f"Sample {i + 1}/{n_samples} · scale {n_scales - s}/{n_scales} "
                                  f"(coarse → fine) · step {steps - t + 1}/{steps}")
            results.append(generate(sampler, int(seed0) + i, out_hw, on_step))
        bar.progress(1.0, text="Measuring copying…")
        metrics = copying_metrics(x, [r.image for r in results])
        bar.empty()
        st.session_state["run"] = dict(key=cfg_key, results=results, metrics=metrics, x=x, name=name,
                                       shapes=sampler.shapes, out_hw=out_hw)

    run = st.session_state.get("run")
    with right:
        if run is None:
            st.subheader("Samples")
            st.info("Choose settings in the sidebar and press **Generate**. "
                    f"On this {DEVICE.upper()}, a 160 px sample takes about 5 s (fast), 15 s (balanced) "
                    "or 2 min (exact)." if DEVICE == "cpu" else "Press **Generate**.")
        else:
            res, met = run["results"], run["metrics"]
            st.subheader("Samples")
            if run["key"] != cfg_key:
                st.caption("⚠ Settings changed since this run: press Generate to update.")
            cols = st.columns(min(len(res), 4))
            for i, r in enumerate(res):
                with cols[i % len(cols)]:
                    show(r.image, f"seed {r.seed} · {r.seconds:.1f} s", scale_to=240)

            st.markdown("**How much is new, how much is copied** (the project's own metrics)")
            mcols = st.columns(3)
            mcols[0].metric("Patch recall (pooled)", f"{met['recall']:.2f}",
                            help="Fraction of the input's 7×7 patches that are the nearest neighbour of "
                                 "some sample patch. Natural images ≈ 0.97; OOD inputs 0.60–0.73.")
            mcols[1].metric("Patch precision (mean)", f"{np.mean(met['precision']):.2f}",
                            help="Fraction of sample patches within the input's own 95th-percentile "
                                 "nearest-neighbour distance: how plausible the local content is.")
            cp = [c for c in met["copied"] if c == c]
            mcols[2].metric("Copied in blocks ≥ 48×48", f"{np.mean(cp):.0%}" if cp else "n/a",
                            help="Fraction of sample pixels inside regions copied verbatim at a constant "
                                 "offset, at least 48×48 input pixels. Only for same-size outputs.")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for r in res:
                    b = io.BytesIO()
                    tensor_to_pil(r.image).save(b, "PNG")
                    z.writestr(f"{run['name']}_seed{r.seed}.png", b.getvalue())
            st.download_button("Download samples (.zip)", buf.getvalue(), f"{run['name']}_samples.zip",
                               "application/zip")

    if run is not None:
        st.divider()
        st.subheader("Inside one sample")
        res = run["results"]
        pick = st.selectbox("Sample", range(len(res)), format_func=lambda i: f"seed {res[i].seed}")
        r = res[pick]
        st.markdown("**Coarse to fine.** Each scale starts from noise; the blend takes its low "
                    "frequencies from the finished coarser scale and its detail from the patches.")
        tcols = st.columns(len(r.trace))
        for j, s in enumerate(sorted(r.trace, reverse=True)):
            with tcols[j]:
                show(r.trace[s], f"scale {len(r.trace) - s} · {r.trace[s].shape[2]}×{r.trace[s].shape[1]}",
                     scale_to=240)
        if r.provenance is not None:
            st.markdown("**Where every patch came from.** At the last step each output patch is "
                        "essentially one input patch. Left: every input position gets its own colour. "
                        "Right: each output location painted with the colour of the patch it took. "
                        "Smooth colour ramps are verbatim copied blocks; speckle means patches "
                        "were reshuffled.")
            h0, w0 = run["x"].shape[-2:]
            pcols = st.columns(4)
            with pcols[0]:
                show(run["x"], "input", scale_to=240)
            with pcols[1]:
                show(position_colours(h0, w0), "colour key (input positions)", scale_to=240)
            with pcols[2]:
                show(r.provenance, "provenance of the sample", scale_to=240)
            with pcols[3]:
                show(r.image, "sample", scale_to=240)

with tab_how:
    st.markdown(r"""
### The idea in one equation
A diffusion model needs a **denoiser**: given a noisy image $x_t = \alpha_t x_0 + \sigma_t \varepsilon$,
estimate the clean one. If the "training set" is just the patches $\{y_j\}$ of a single image, the
best possible (MMSE) patch denoiser has a closed form:

$$
D(x) \;=\; \sum_j \operatorname{softmax}_j\!\Big(-\frac{\lVert x - \alpha_t\, y_j \rVert^2}{2\sigma_t^2}\Big)\; y_j
$$

That is **scaled dot-product attention**: queries are the noisy patches, keys are the scaled clean
patches, values are the clean patches. No weights are learned, so nothing is trained.

### The sampler
1. Build a pyramid of the input (e.g. 4 scales, ½ each).
2. At the coarsest scale, start from pure noise and run **T = 10** deterministic DDIM steps; every
   step unfolds the image into overlapping 15×15 patches, denoises each with $D$, and folds them back
   with a Gaussian window (ρ = 0.2).
3. Move one scale finer: start from noise again, but at each step replace the low frequencies of the
   estimate with the finished coarser image (**two-scale Laplacian blend**), so the global layout is
   inherited and only detail is generated.
4. The finest scale is the sample. Changing the output size (retargeting) only changes the noise.

### What this demo adds
* **Patch stride** — evaluate the denoiser on every *s*-th patch (always including the borders) for
  about $s^4$ less work on CPU. Stride 1 is bit-identical to our tested re-implementation.
* **Provenance map** and **copying metrics** — the project's finding is that this method makes
  convincing natural textures, but on structured images (floor plans, circuits, text) it either
  copies large blocks verbatim or breaks lines and garbles text. Try a floor plan and a natural
  image and compare the maps.

Code: `src/sidx/core/` (method), `app/engine.py` (demo additions), `src/sidx/metrics/` (metrics).
""")
