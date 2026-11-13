# Phase 1 — Technical Notes: Paper Analysis, Code Audit, Literature

**Status:** Phase 1 complete (see the completion checklist in §8).
**Date:** 2026-11-13.
**Sources audited:**

| Source | Version |
|---|---|
| Paper | arXiv:2606.04299v1 (`Project/2606.04299v1.pdf`) |
| Official code | `external/efficient-SID-code` @ `85381e3c6f29fc07144da854d86be07150404716` (2026-09-04), MIT |
| SinDDM baseline | `external/SinDDM` @ `e7107d9fd198928f26e278f99ec428e261726a45` (2023-08-20), MIT |

**Tags:** **[P]** paper · **[C]** code (file:line in `external/efficient-SID-code/efficient_sid/` unless noted) · **[D]** our derivation · **[V]** verified by running a check on this machine · **[!]** still open, to verify on a GPU in Phase 4.

---

## 1. The algorithm in our own words (one page, for the professor)

**Goal.** Given one image *y*, produce new images that are "made of the same pieces" — the same distribution of small patches at every scale — but arranged differently.

**Key idea.** A diffusion model only needs a *denoiser*: a function that takes a noisy image and the noise level, and guesses the clean image. Normally this is a neural network trained for hours. But if the "training set" is just the set of all 15×15 patches of one image (~40 000 vectors of 675 numbers), the *best possible* denoiser — the one any perfectly trained network would converge to — can be written down exactly:

> denoised patch = weighted average of all clean patches of *y*,
> weight of patch *j* ∝ exp( −‖noisy patch − α·yⱼ‖² / 2σ² ).

**Why that formula.**
1. The data distribution is a sum of spikes, one per patch.
2. Adding Gaussian noise turns it into a mixture of Gaussians, one centred on each α·yⱼ.
3. The minimum-mean-squared-error estimate of the clean patch is the posterior mean E[y | xₜ].
4. For a Gaussian mixture, that posterior mean is exactly this softmax-weighted average (paper Supp. S2, Eqs. S1–S18).

**How sampling works.**
1. Start from pure noise.
2. At every step:
   - cut the current noisy image into all overlapping patches;
   - replace each patch by its weighted average of clean input patches;
   - glue the patches back together (each pixel is a Gaussian-weighted average of the patches that cover it, favouring patch centres);
   - compute the implied noise;
   - step to a slightly less noisy image (DDIM).
3. Early on (large σ) the weights are almost uniform, so the output is a blurry average.
4. At the end (σ → 0) the softmax becomes a hard "pick the nearest patch", so every output patch is a real patch of the input. The sampler is a **soft-to-hard patch-selection schedule**.

**Why it needs multiple scales.** A 15-px patch can't see global layout, so single-scale samples have the right texture but scrambled structure. The fix is to run the same procedure on an image pyramid:
- Start at a scale where the image is only ~24 px tall, so a 15-px patch sees most of it.
- At each finer scale, after every denoising step, replace the low-frequency band of the estimate with the upsampled result of the finished coarser scale. Keep only this scale's high-frequency detail (a two-level Laplacian blend).

**Why it is fast.** The weighted average is mathematically identical to one attention operation, softmax(QKᵀ/σ²)V, so it can be computed with PyTorch's fused attention kernel. For large images it can also run on VAE latents (64× fewer patches) and use approximate nearest neighbours.

**Its main limitation.** Every output patch is an input patch; nothing new can be invented. "Meaning" (connectivity, text, symbols) is never modelled — this is the gap our OOD study probes.

---

## 2. Complete pipeline as implemented (unconditional generation)

Entry point `scripts/sample_uncond.py` → `applications/uncond.py:sample_uncond`. Shapes are for a 186×248 input with the Table-1 preset `configs/uncond/paper_sinddm15.yaml`.

| # | Step | Code | Detail |
|---|---|---|---|
| 1 | Load | `utils.py:64` | RGB → float in [0,1] → **cast to `precision.dtype` (bfloat16 in every preset, including the Table-1 one)** → [-1,1] (`utils.py:26`) |
| 2 | Resolve config | `build.py:330` | `"auto"` fields filled from the image size (the Table-1 preset has none) |
| 3 | Schedulers | `scheduler.py:189` | One per scale; see §3.2 for the exact grid |
| 4 | Pyramid | `pyramid.py:117`, `resize_right.py` | 4 scales, ×0.5 each, `ceil`. Each scale is resized from the previous one with **antialiased cubic (support 6)**: 186×248 → 93×124 → 47×62 → 24×31 |
| 5 | Patch datasets | `build.py:433`, `image_denoiser.py:21` | `Tensor.unfold`, patch 15, stride 1 ⇒ N = 172·234 = 40 248 patches (D = 675) at the finest scale; 9·17 = 153 at the coarsest |
| 6 | Noise init | `scheduler.py:165`, `uncond.py:76` | x_{s,T} = 1.0 · N(0, I) for **every** scale, drawn up front |
| 7 | Scale loop | `uncond.py:80` | s = 3 (coarsest) → 0 |
| 8 | Step loop | `uncond.py:82` | t = T … 1 (index 0 is the clean terminal) |
| 8a | Unfold queries | `image_denoiser.py:132` | All overlapping noisy patches, stride 1 |
| 8b | Closed-form denoise | `closed_form_denoiser.py:142` | logits = −(‖αy‖² − 2 xᵀ(αy)) / 2σ² — the ‖x‖² term is dropped (softmax is shift-invariant) — then softmax over the N clean patches, then weights @ y |
| 8c | Fold | `image_denoiser.py:145-156` | Multiply by the Gaussian window, `F.fold` (sum over overlaps), divide by the folded window sum ⇒ per-pixel weighted average |
| 8d | ε̂ | `scheduler.py:120` | ε̂ = (xₜ − α x̂)/σ, computed **before** the blend |
| 8e | Blend | `pyramid.py:140`, `uncond.py:88` | Only for s < coarsest and t ≠ 1: x̂ ← x̂ − up(down(x̂)) + up(x_{s+1, final}) |
| 8f | Step | `scheduler.py:127` | x_{t−1} = α_{t−1} x̂ + √(σ²_{t−1} − (ησ_{t−1})²) ε̂ + ησ_{t−1} z; landing on index 0 returns x̂ itself |
| 9 | Output | `uncond.py:95` | x_{0,final} → [0,1] → PNG. Sample *i* uses seed `seed+i` (`sample_uncond.py`) |

---

## 3. Resolution of the open items ([?]) from PROJECT_PLAN.md

| # | Open item | Resolution | Evidence |
|---|---|---|---|
| Q1 | **Units of ρ** (`fold_rho`) | Std of a separable Gaussian over the patch's **normalized coordinates in [−1, 1]**. In pixels, σ_px = ρ·(p−1)/2. For p = 15, ρ = 0.2 ⇒ **1.4 px**; for p = 7 ⇒ 0.6 px; ρ = ∞ ⇒ uniform averaging | [C] `image_denoiser.py:94-106` |
| Q2 | **Blur in TwoScaleBlend** | Blur(x̂) = up(down(x̂)), resampled to the next pyramid shape (×0.5, ceil) with the vendored ResizeRight: **cubic kernel, support 6.0, antialiasing on when downscaling**. Same resampler for upsampling the coarser result | [C] `pyramid.py:140-176`, `resize_right.py:5-39` |
| Q3 | **Value range** | [−1, 1] in pixel space. Latent space uses `(z − shift_factor)·scaling_factor` | [C] `utils.py:26-31`, `latent.py:136` |
| Q4 | **η update** | Matches paper Eq. 5 with c = ησ_{t−1}. η = 0 is deterministic DDIM (an Euler step of the flow); η = 1 discards ε̂ entirely. The code warns this is **not** the DDIM paper's η (whose η = 1 means DDPM) | [C] `scheduler.py:127-163` |
| Q5 | **Scale count S vs S+1** | `num_scales: 4` means **4 levels total** (s = 0…3), consistent with Fig. S18. Paper Alg. 2's "s = S…0" with "S = 4" over-counts by one | [C] `pyramid.py:62`, `paper_sinddm15.yaml` |
| Q6 | **t = 0 blend skip** | Implemented as "no blend on the step from index 1 onto the terminal" | [C] `uncond.py:88` |
| Q7 | **Which SIFID / NIQE / NIMA / MUSIQ / diversity implementation** | **Not in the official repo** (no metric code at all), **not in the SinDDM repo**, and not named in either paper. The two papers' numbers for SinDDM disagree, so the efficient-SID authors used their own implementation. We define our metrics and calibrate them in §5 | [C] grep over both repos; SinDDM paper text |
| Q8 | Is FLUX VAE access needed for reproduction? | **No.** Table 1 and every pixel preset need no Hugging Face login; only `latent_*` presets do | [C] README, `latent.py:68-90` |
| Q9 | Does the method need faiss? | **No**, except for `ann*` backends. A failed import is caught and only the ANN backends are disabled | [C] `closed_form_denoiser.py:35-43` |
| Q9b | *(Phase 2)* Is CLIP needed for unconditional runs? | **Yes.** `config.py` imports `applications/text_style.py`, which imports `clip_extractor.py` → `import clip` at module level. `clip-anytorch` in turn needs `setuptools<70` (`from pkg_resources import packaging` fails on 70.0.0; verified locally) | [C] `config.py:21`, `clip_extractor.py:25`; [V] |
| Q10 | Does the code run on CPU? | **No.** `.cuda()` is hard-coded in `build_image_denoiser` and the autocast is CUDA-only | [C] `build.py:457`, `closed_form_denoiser.py:143` |

---

## 4. Paper vs. code — discrepancy table

Impact column: **High** = can change reproduced numbers or a conclusion; **Med** = matters for some experiments; **Low** = documentation only.

| # | Item | Paper | Code | Impact | Our handling |
|---|---|---|---|---|---|
| D1 | **Noise grid** | α(t) = 1 − t/T, σ = t/T; evaluations at σ ∈ {1.0, 0.9, …, 0.1}; the last update lands on t = 0 (§3.6) | Evaluation αs = `linspace(1−σ_min, 0, T)`: **T points from σ = 1 down to σ = σ_min = 1e-4, spacing 1/(T−1)**. The step from index 1 returns x̂ itself | **High** | The last denoise is at σ = 1e-4 — effectively a hard nearest-patch copy — instead of a soft average at σ = 0.1. This sharpens outputs and supports the provenance claim. Our re-implementation follows the **code**, and the report states this |
| D2 | **Precision of the Table-1 run** | "float32" timing (§3.6) | `paper_sinddm15.yaml`: `precision.dtype: bfloat16` (image, pyramid, xₜ, x̂ all bf16) and `bottleneck_dtype: float32` (autocast inside the kernel) | **Med** | bf16 holds an 8-bit image almost losslessly (error < ½ grey level) [D]. But see D3 |
| D3 | `bottleneck_dtype: float32` effect | — | `torch.autocast(device_type='cuda', dtype=float32)`. In torch 2.11 the CUDA branch does not reject float32 [V: `autocast_mode.py`], so matmul/softmax should run in fp32 on GPU, while elementwise ops (α·y, ‖αy‖²) stay **bf16** — 0-dim fp32 × bf16 → bf16 [V] | **Med** | **[!]** Log the `weights` dtype on a GPU in Phase 4; run the equivalence test in fp32 *and* in the shipped mixed mode |
| D4 | Tiling | 3 shifted passes, unspecified mask | `num_shifts: 4` (adds a diagonal shift for corners). **Hard binary quarter masks**, not soft blending. `num_shifts=3` gives the paper variant | Low | Use 3 for reproduction, 4 for OOD |
| D5 | Symmetrization | "flip-copy half after every denoising timestep" | Applied to x̂ **before** ε̂ is computed (so ε̂ is consistent with the symmetric x̂); preset T = 25, η = 1 | Low | Document |
| D6 | Retargeting | "use it to initialize x̂_{0,S}" | The coarsest scale is the resized image itself and is **never denoised**; preset T = 25, η = 1, min_short_side = 15 | Low | Document |
| D7 | Structural analogy | Invert at s = 2 to t′ = T/10 or T/2 | 3 scales (coarsest = 2), **patch 11**, `noise_level 0.5`, per-scale steps [10, 10, 50] (50 at the coarsest), **σ_min = 1e-2** (stronger inversion warm-up) | Low | Use the preset |
| D8 | Inversion warm start | "small noise at the first step" | x₁ = α₁y + σ₁ε with σ₁ = σ_min (1e-4 default, 1e-2 for analogy) | Low | Document |
| D9 | CLIP gradient mask (S39–S40) | Hard top-f mask, then energy renormalization | Additionally **soft-thresholds**: the gradient magnitude is reduced by the (1−f)-quantile before masking (`thresholded_grad`). Preset: patch 7, T = 200, noise 0.3, 16 CLIP augmentations, fp32 | Low | Only qualitative use for us |
| D10 | **Latent presets use ρ = ∞** | Table 3 (pixel, single-scale): ρ = ∞ is worst | All `latent_*` presets: `fold_rho: .inf`, patch 7, T = 20, η = 1 | **Med** for the pixel-vs-latent ablation | Compare at matched ρ as well as at the presets |
| D11 | Default (non-paper) pixel presets | S = 4 fixed | `num_scales: auto` with `min_short_side: 24` ⇒ n = the smallest with 24·2ⁿ⁻¹ ≥ short side. That gives **4 scales** for the sinddm15 images (short side 167–200 px ≤ 192), but **5 scales for a 256-px image**. The coarsest is floored to 24 px. `pixel_exact` also uses a bf16 bottleneck | **Med** (our OOD images are 256 px) | Always use `paper_sinddm15.yaml` for Track 1; for OOD, fix `num_scales` explicitly and log the scale table that `report_scale_table` prints |
| D12 | ANN default | n_probe = 40 (gigapixel), k = 5 | `pixel_ann`: IVF, n_probe = **2**, k = 5, bf16. Gigapixel: IVF-PQ, n_probe = 40, pq_m = 28 | Med (AB6) | Sweep n_probe in AB6 |
| D13 | Auto backend switch | — | `type: auto` picks `ann` above 250 000 patches and strides queries above 8M; tuned for a 96 GB GPU | Low | Don't use `auto` in controlled experiments |

---

## 5. Evaluation-metric definitions (resolving Q7)

The official repo ships no metric code, so these are the definitions we will implement. They follow SinGAN/SinDDM conventions; details confirmed from the SinDDM papers are marked.

### 5.1 What the SinDDM paper (Kulikov et al., ICML 2023) states
These quotes come from the arXiv v3 text, extracted on 2026-11-13.
- *"Each measure in the table is computed over 50 samples per training image (we report mean and standard deviation)."*
- Diversity = *"pixel standard-deviation and average LPIPS distance between pairs of samples."*
- IQA = NIQE (Mittal 2012), NIMA (Talebi & Milanfar 2018), MUSIQ (Ke 2021). SIFID = Shaham et al. 2019. **No implementations or LPIPS backbone are named.**
- Training: *"120×10³ steps … lr 0.001, halved at [20,40,70,80,90,110]×10³ … batch 32 … EMA 0.995. For a 200×250 image, training takes around 7 hours"* on an A6000; 1.1 M parameters.

### 5.2 Key finding: the two papers' numbers for SinDDM disagree

| SinDDM metric | SinDDM paper (12 images) | Efficient-SID Table 1 (15 images) |
|---|---|---|
| SIFID ↓ | 0.34 ± 0.30 | 0.48 ± 0.62 |
| Pixel div. ↑ | 0.32 ± 0.13 | 0.10 ± 0.03 |
| LPIPS div. ↑ | 0.21 ± 0.08 | 0.36 ± 0.07 |
| NIQE ↓ / NIMA ↑ / MUSIQ ↑ | 7.1 / 5.8 / 48 | 7.69 / 4.30 / 50.74 |
| Training time | ~7 h (A6000) | 8.0 h (A6000) |

What this tells us [D]:
1. The efficient-SID authors **recomputed every metric with their own (unpublished) implementation and image set**.
2. Pixel diversity in particular must be normalized differently: a 3× lower value for the same method.
3. NIMA's absolute level differs by ~1.5, which suggests a different NIMA model/weights.

**Consequence:** absolute agreement with Table 1 cannot be guaranteed by any implementation. Track 1's acceptance criterion changes from "within 10–15 % of the paper's numbers" to:
- (a) **the same ranking/trend** across methods and settings;
- (b) **matching the paper's numbers only for metrics whose definition we can pin down** (see 5.3).

### 5.3 Our metric definitions (to implement in Phase 5)

| Metric | Definition we will use | Variants to compute for calibration |
|---|---|---|
| **SIFID** | SinGAN's official code (`tamarott/SinGAN`, `SIFID/sifid_score.py`): InceptionV3 features from the **first block (64-ch, before the 2nd max-pool)**; Fréchet distance between the spatial feature sets of the real and generated image; averaged over 50 samples | — |
| **Pixel diversity** | SinGAN definition: per-pixel std across the 50 samples, averaged over pixels and channels, **divided by the std of the input image's pixels** | Also unnormalized std in [0,1] and in [−1,1] |
| **LPIPS diversity** | Mean LPIPS over all 1 225 sample pairs, **AlexNet** backbone (`lpips` default), inputs in [−1,1] | VGG backbone; 50 random pairs |
| **NIQE / MUSIQ / NIMA** | `pyiqa` defaults (`niqe`, `musiq`, `nima`) | `musiq-koniq` vs `musiq-ava`; `nima` vs `nima-vgg16-ava` |
| **Near-duplicate rate** | Paper Table S1 exactly: a sample is a near-duplicate at level r if ≥ r % of its pixels have mean-over-RGB absolute difference < 15/255 from the input | — |
| **Runtime** | Wall clock around the sampler with `torch.cuda.synchronize()`; excludes model/index build and I/O. The official `timing.py` already separates these | — |

**Calibration procedure (Phase 5)** [R]:
1. Compute every variant on **(a)** our efficient-SID samples and **(b)** SinDDM samples from the 3 pretrained models.
2. Pick the variant per metric that reproduces the *ratio* between the two methods in Table 1 (e.g. pixel div. ours / SinDDM = 0.15 / 0.10 = 1.5).
3. Freeze it for all later experiments.

---

## 6. SinDDM baseline — facts from its repository

| Item | Value | Source |
|---|---|---|
| Environment | Python 3.8, **torch 1.13.1 + CUDA 11.7** (pinned `nvidia-*-cu11` wheels), numpy 1.24.1, opencv 4.6, scikit-image 0.19.3 | `SinDDM/requirements.txt` |
| Training cost | ~7 h for a 250×200 image on an A6000 (SinDDM paper); 8 h in efficient-SID Table 1. On an RTX 2080: batch 16 and 4× the steps | SinDDM paper App. |
| Training defaults | **120 001 steps**, batch 32, lr 1e-3 with milestone schedule, L1 loss, 100 diffusion timesteps per scale, scale factor 1.411 between scales, conv width 160, checkpoint every 10 000 steps (`model-12.pt` = the final one) | `SinDDM/main.py:34-58, 97` |
| Pretrained models | **Only 3**: `forest`, `night_sky`, `pyramids` (`results/<name>/model-12.pt`) | repo tree |
| Benchmark images | 15 PNGs in `datasets/` (aqueduct, aurora, balloons, elephants, eucalyptus, field_poppies, forest, marinabaysands, mountains, mountains_2, night_sky, pyramids, pyramids_singan, seascape, starry_night), **248–298 px wide × 167–200 px tall** | [V] read with PIL |
| Metrics code | None | grep |
| Multi-scale difference | SinDDM blurs coarser-scale training images ("blurry_img") and mixes blurry/clean with learned γ — vs Laplacian blending here | `SinDDM/trainer.py:36-60, 122` |

**Consequences for the plan.**
1. The planned "calibrate our metrics with SinDDM's pretrained models" step (Phase 5) only works for **3 of the 15** images.
2. Since SinDDM's Table-1 numbers were computed on all 15, the calibration becomes: compute *our* metrics on SinDDM samples for those 3 images and compare to SinDDM's per-image results if published; otherwise use them only as a sanity range.
3. Full calibration needs SinDDM training on the other 12 (≈ 96 GPU-h) — **not planned**.
4. Torch 1.13/CUDA 11.7 cannot run on Ada/Hopper/Blackwell-class GPUs reliably (sm_89+ needs CUDA ≥ 11.8) **[!]** — budget time for a torch-2.x port in Phase 2.

---

## 7. Literature reading notes

| Work | One-line takeaway | Why it matters to us |
|---|---|---|
| **Buades et al., Non-Local Means (CVPR 2005)** | Denoise each pixel/patch as a similarity-weighted average of other patches of the same image | Our denoiser is NLM with a *clean* reference set and a schedule over σ — the paper's §3.2 |
| **Zoran & Weiss, EPLL (ICCV 2011)** | Learn a GMM prior over patches; restore whole images by patch-wise MAP with overlap averaging | The closed-form prior is a GMM with one isotropic component per patch; EPLL's overlap-averaging issue ≈ our ρ |
| **Efros & Leung (1999); Efros & Freeman, image quilting (2001)** | Non-parametric texture synthesis by copying best-matching neighbourhoods | Historical root; known to fail on structured, non-stationary images — a prior for our OOD hypotheses |
| **SinGAN (Shaham et al., ICCV 2019)** | Pyramid of patch-GANs trained on one image; introduced **SIFID** | Task definition and the main metric |
| **GPNN (Granot et al., CVPR 2022)** | Coarse-to-fine nearest-neighbour patch replacement, noise injected once; no training | Closest non-parametric relative; the paper shows it near-duplicates the input (Table S1) |
| **GPDM (Elnekave & Weiss, ECCV 2022)** | Optimizes a sliced-Wasserstein patch-distribution distance, coarse to fine | Motivates our **patch-SWD** metric |
| **SinDDM (Kulikov et al., ICML 2023)** | Multi-scale DDPM trained on one image with a small receptive field; ~8 h/image | Our primary training-based baseline |
| **Kamb & Ganguli (ICML 2025); Niedoba et al. (ICML 2025)** | Trained CNN diffusion models behave like *locally* optimal (patch-wise) closed-form denoisers — this explains their creativity/generalization | Theoretical justification that a patch closed-form denoiser is a meaningful model, and that its failures (non-local structure) may also be failures of trained models — relevant to the SinDDM comparison |
| **DDIM (Song et al., ICLR 2021); Flow matching (Lipman et al., ICLR 2023)** | Deterministic sampler; linear-interpolation schedule | Sampler and schedule used here |

---

## 8. Phase 1 completion checklist

| Criterion (from PROJECT_PLAN §12, Phase 1) | Status |
|---|---|
| Official source read (config, denoiser, fold, pyramid, scheduler, apps) | ✅ all 20 modules listed in §2–§4 |
| Every [?] resolved or logged | ✅ Q1–Q10 resolved; 2 items moved to GPU verification **[!]** (D3 autocast dtype; SinDDM on new GPUs) |
| Paper-vs-code difference table | ✅ §4, 13 items, 3 high/medium impact on reproduction (D1, D2/D3, D11) |
| SinDDM and GPNN read | ✅ SinDDM repo audited (§6); GPNN covered by the paper's S1.1 + notes (§7) |
| "Algorithm in our own words" note | ✅ §1 |

### Changes this phase forces on PROJECT_PLAN.md
1. **Track 1 must use the code's noise grid (D1)**, and our re-implementation must match it — otherwise the equivalence test in Phase 4 will fail by design.
2. **Phase 5 metric calibration**: only 3 SinDDM pretrained models exist (§6) — scope reduced accordingly.
3. **Phase 2**: the local machine has **no NVIDIA GPU** (CPU-only torch 2.11; WSL2 Ubuntu is installed but has no CUDA). The official code is CUDA-only (Q10) ⇒ a remote/cloud/lab GPU is a hard requirement before Phase 4.
4. **Metrics (Q7)**: SIFID, IQA and diversity must be implemented by us — Phase 5 gets one extra day.
5. **The pixel-vs-latent ablation (AB5)** must control ρ (D10).
6. **Track 1 acceptance criterion (R1) changes** from "within 10–15 % of Table 1" to "same trends/rankings, plus a calibrated match where the metric definition can be pinned down" (§5.2). The two papers report different numbers for the same SinDDM method.
7. **For OOD runs, fix `num_scales` explicitly** — `auto` gives 5 scales for 256-px inputs versus 4 in the paper (D11).
