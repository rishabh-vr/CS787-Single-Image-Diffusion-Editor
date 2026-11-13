# Project Blueprint: Training-Free Single-Image Diffusion — Reproduction, OOD Generalization, and Comparison to Training-Based Baselines

**Reference paper:** H. Qiu, K. N. Kutulakos, D. B. Lindell, *Efficient and Training-Free Single-Image Diffusion Models*, arXiv:2606.04299 (June 2026; CVPR 2026 per the code README).
**Local copy:** [2606.04299v1.pdf](2606.04299v1.pdf) (11 pages main + 27 pages supplement).
**Official code:** https://github.com/haojunqiu/efficient-SID-code (MIT).

### Provenance tags used throughout

| Tag | Meaning |
|---|---|
| **[P]** | Stated explicitly in the paper (main text or supplement). Section/equation cited. |
| **[C]** | Stated in the official repository README / config files (checked 2026-11-13). Source *code* itself has **not yet been audited** — that is Phase 1. |
| **[D]** | My own derivation from the paper's equations (mathematically implied, not written in the paper). |
| **[R]** | My recommendation. |
| **[H]** | Hypothesis that must be validated experimentally. |
| **[E]** | Estimate (not measured). |
| **[?]** | Unknown — must be resolved. |

---

## 1. Executive Summary

**What the paper does.** Given one image, it generates new images whose multi-scale *patch distribution* matches the input (unconditional generation, retargeting, symmetrization, tiling, structural analogy, CLIP-guided stylization). It does this **with no training and no neural denoiser**: because the "dataset" (all overlapping patches of one image) is finite and small, the MMSE-optimal diffusion denoiser has an exact closed form — a softmax-weighted average of the input's clean patches. This closed-form patch denoiser is run inside a DDIM / flow-matching sampler, and a coarse-to-fine image pyramid with Laplacian-style blending supplies global layout. **[P]**

**Two corrections to the project brief that change the plan:**

1. **There is no pretrained diffusion model in this method.** The brief describes "manipulating the diffusion process of a pretrained model during inference". That is not what this paper does. The *denoiser is the image itself*. Pretrained networks appear only as *optional* accessories: the FLUX.1 VAE (latent-space acceleration), CLIP ViT-B/32 (text-guided stylization), and Inception/LPIPS/IQA networks (evaluation only). **[P §3.5, §4.2; C]** Consequence: "same pretrained backbone" cannot be held constant in the baseline comparison; we must control other factors instead (§9).
2. **The paper's natural comparison point is a *training-from-scratch* single-image diffusion model (SinDDM), not a fine-tuned large model.** DreamBooth/LoRA fine-tune a large text-to-image model on one image — this solves a *different* task (semantic subject personalization) and does not aim to reproduce patch statistics. I recommend **SinDDM as the primary training-based baseline** and **DreamBooth-LoRA as a secondary, smaller "pretrained-prior fine-tuning" contrast** (§9). This choice should be confirmed with your professor.

**Recommended research focus.** Line drawings (architectural floor plans and circuit schematics), regular geometric patterns/textures, and a small number of text-heavy scientific diagrams, plus a **procedurally generated synthetic set** with controllable line thickness, density and periodicity. The synthetic set is what lets us answer *why* the method fails, not just *whether*.

**Central hypotheses [H]:** the method preserves *local* structure (inside a patch) but breaks *long-range* structure (line continuity, junction topology, lattice phase) at patch-stitching boundaries; it under-samples rare patches (symbols, junctions), causing detail loss; and thin structures alias away at coarse pyramid scales so the global layout is lost.

**Budget: zero cost — free Google Colab (T4, 16 GB) + a CPU laptop.**
- A ~10-week plan for 2–3 people, ≈ 40 T4-hours in total (§5.3).
- The training-free method fits easily.
- The training-based baseline (SinDDM ≈ 7–8 h/image on an A6000) runs only at a reduced budget: 3 pretrained natural-image models plus 3 OOD images (Phase 8).
- A 6-week compressed variant is given in §13.

---

## 2. Paper Understanding and Algorithm Breakdown

### 2.1 Answers to the 12 analysis questions

| # | Question | Answer |
|---|---|---|
| 1 | Problem | Generate images whose internal structure (distribution of patches at multiple scales) matches a single reference image, *without* the hours of per-image training that SinGAN/SinDDM/SinFusion/SinDiffusion need. **[P §1]** |
| 2 | Contributions | (a) Closed-form, training-free patch denoiser for single-image diffusion; (b) connection to non-local means and GMM patch priors; (c) coarse-to-fine sampler with two-scale Laplacian blending; (d) three accelerations: fused attention, latent (VAE) space, approximate NN — megapixel in ~1 s, gigapixel in minutes; (e) applications: unconditional, retargeting, symmetrization, tiling, structural analogy, CLIP text stylization, ROI conditioning. **[P §1, §3–4, S3–S5]** |
| 3 | Theory | Empirical data distribution = sum of Diracs over patches ⇒ noisy marginal is a Gaussian mixture ⇒ MMSE denoiser = posterior mean = softmax-weighted average of clean patches (Eq. 2, S1–S25). Interpretable as Nadaraya–Watson kernel regression and as scaled dot-product attention. **[P S2, S3.2]** |
| 4 | How the "manipulation" works | Not a manipulation of a network: the *denoiser itself* is replaced by an analytical function of the input's patches; each sampling step = extract noisy patches → closed-form denoise each → fold back into an image → DDIM/ancestral noise update. **[P §3.3, Alg. 1]** |
| 5 | Modifications to standard sampling | (i) Denoising done per-patch and reassembled with Gaussian-weighted folding (ρ); (ii) multi-scale outer loop with a full T-step diffusion per scale; (iii) TwoScaleBlend after each denoise at s<S; (iv) blend skipped at t=0; (v) app-specific constraints injected after each denoise (flip-copy, circular shifts, ROI masks, CLIP gradient). **[P §3.3–3.4, §4.2, S4]** |
| 6 | Equations / params | Eqs. 2–6, 8; S18, S27, S32–S36, S37–S40. Params: T=10, η=0, patch 15×15, stride 1, ρ=0.2, scale factor 0.5, S=4 (see 2.4 note), flow schedule α=1−t/T, σ=t/T. **[P §3.6; C config]** |
| 7 | Pretrained model | **None for the core method.** Optional: FLUX VAE (`black-forest-labs/FLUX.1-schnell`, gated on HF), CLIP ViT-B/32. **[P §3.5, §4.2; C]** |
| 8 | Inference compute | ~3 s for 186×248 at T=10 on A6000 in a naive PyTorch implementation, float32 (§3.6); Table 1: 3.09 s (A6000), 0.88 s with ANN k=5. Table 2 (RTX 6000 Ada, 48 GB): 512² 23.4 s with fused attention, 0.39 s latent, 0.96 s +ANN; 8192² 69 s with all accelerations. **[P]** |
| 9 | Data | 15 images from the SinDDM repo (origin: Places50, SIGD16) for Table 1; additional figure inputs; three gigapixel images (308 MP, 500 MP, 1216 MP). No training data. **[P S5; C]** |
| 10 | Metrics/protocol | SIFID; NIQE, NIMA, MUSIQ; pixel diversity, LPIPS diversity; training time; inference time (TITAN RTX & A6000). 50 samples/image, mean±std over 15 images. Near-duplicate rate (Table S1, 15/255 RGB threshold, r% pixels). Hyperparameter studies: Table 3 (patch size × ρ, single-scale, 5 samples/image), Fig. 8 (SIFID vs T and η). **[P §4, S5]** |
| 11 | Acknowledged limitations | The paper has **no dedicated limitations / failure-case section.** Scattered admissions: O(N²) cost of exact denoising (S3.1); memorization concerns of closed-form denoisers at image level (S2.2); explicit multi-scale loop may be unnecessary (S6.1); GPNN/GPDM beat it on SIFID (attributed to copying); ANN slightly worsens SIFID; higher T/η trade diversity for fidelity; Fig. S19 hypothesis that reconstruction at coarse scale biases patch selection. **[P]** ⇒ *This gap is precisely where our project contributes.* |
| 12 | Implicit assumptions | (a) The input is internally self-similar — new images can be made by recombining its patches; (b) meaningful structure at each scale fits into a ~15 px patch; (c) the coarsest scale (patch ≈ half image) captures global layout; (d) L2 distance in RGB (or VAE latent) space is a good patch similarity; (e) **outputs cannot contain any patch not in the input** at σ→0 (provenance property, S5); (f) no semantics — any "meaning" (text, symbols, connectivity) is not modeled. **[D from P]** |

### 2.2 Notation

- Input image **y** ∈ ℝ^{H×W×3}. Patch extractor **P**⁽ⁱ⁾ (stride 1) ⇒ clean patch set 𝒴 = {**y**⁽ʲ⁾}, each d = 15·15·3 = 675 dims.
- Noisy image **x**_t = α(t)**y** + σ(t)**ε**. Schedule (flow matching): α(t) = 1 − t/T, σ(t) = t/T. **[P §3.6]** Config also sets `sigma_min = 1e-4`. **[C]**
- Pyramid scales s = 0 (full res) … S (coarsest), downscale factor 0.5 per scale. **[P]**

### 2.3 The complete computational pipeline (coarse-to-fine unconditional generation)

```
INPUT  y (e.g. 186×248×3, values in [-1,1] or [0,1] — [?] verify range in code)
CONFIG T=10, eta=0, patch=15, stride=1, rho=0.2, scales=4, factor=0.5

1. Build input pyramid           y_s = Resize(y, 0.5^s)            s = 0..S
2. Build patch datasets          Y_s = unfold(y_s, 15, stride 1)   shape [N_s, 675]
3. Init noise pyramid            x_{s,T} ~ N(0, I), same shape as y_s, for all s
4. for s = S down to 0:                          # coarse → fine
     for t = T down to 1:
        a. X  = unfold(x_{s,t})                  # noisy patches [N_s, 675]
        b. L  = -||X_i - α_t Y_j||² / (2σ_t²)    # [N_s, N_s] logits
        c. W  = softmax_rows(L)                  # soft patch assignment
        d. X̂  = W @ Y_s                          # denoised patches
        e. x̂  = fold_rho(X̂)                      # Gaussian-weighted reassembly
        f. if s < S and t > 1(*):
              x̂ = x̂ − Blur(x̂) + Upsample(x_{s+1,0})   # TwoScaleBlend
        g. ε̂  = (x_{s,t} − α_t x̂) / σ_t
        h. x_{s,t−1} = α_{t−1} x̂ + sqrt(σ_{t−1}² − c²) ε̂ + c·z,  c = η σ_{t−1}, z~N(0,I)
5. RETURN x_{0,0}
(*) "we omit the two-stage blending on the t=0 diffusion step" [P §3.6]
```

Memory note **[D]**: at 250×250 with 15×15 patches, N ≈ 236² ≈ 55.7 k; the dense N×N logit matrix is 55.7k² × 4 B ≈ **12.4 GB in fp32**. This is why the code has `query_chunks` and the FlashAttention path, which never materializes W.

### 2.4 Component-by-component analysis

For each component: **what / why / how / internals / if removed or changed**.

#### (a) Patch extraction (stride 1) — **[P]**
- **What:** turns one image into a dataset of ~N=H·W overlapping patches per scale.
- **Why:** this dataset *is* the model; stride 1 maximizes the number of samples and gives translation-dense coverage.
- **How:** `torch.nn.functional.unfold` (or equivalent); code config `strides: 1`. **[C]**
- **Internals:** border patches only exist where the full patch fits ⇒ an image with H×W pixels yields (H−p+1)(W−p+1) patches.
- **If changed:** a larger stride reduces cost quadratically but removes sub-pixel-aligned patch variants ⇒ **[H]** misaligned edges / jagged lines, especially on thin-line drawings (a good ablation for OOD).

#### (b) Noise schedule & sampler (Eqs. 3–6) — **[P]**
- **What:** flow-matching linear schedule α=1−t/T, σ=t/T; DDIM-style update with stochasticity η=c/σ.
- **Why:** standard diffusion reverse process; T=10 is enough (SIFID plateaus by ~10 steps, Fig. 8).
- **Internals [D]:** at t=T, α=0, so every noisy patch is equidistant (in the α·y term) from all clean patches ⇒ **W is uniform ⇒ x̂_T = the mean patch everywhere**; all randomness is carried by ε̂ = x_T. As σ→0 the softmax sharpens to an argmax ⇒ at the last step each location *copies one real input patch* (provenance). The sampler is therefore a **soft-to-hard patch selection schedule**, unlike GPNN's hard NN from the start (S1.1).
- **If changed:** η>0 adds fresh noise each step — improves SIFID at larger T but reduces diversity (Fig. 8, Table 1). Fewer than ~5 steps ⇒ SIFID rises sharply (Fig. 8).

#### (c) Closed-form patch denoiser (Eq. 2 / S18) — **[P]**
- **What:** x̂ = Σ_j softmax_j(−‖x−α y_j‖²/2σ²) · y_j.
- **Why:** it is the exact minimizer of the diffusion training loss (Eq. 1) for a finite dataset — i.e. what a perfectly trained network would converge to (S2.1). So no training is needed.
- **How:** naive `torch.cdist`/matmul + softmax + matmul (S3.1); or SDPA with q=[x;1], k=[αy; −‖αy‖²/2], v=y, scale 1/σ² (S32–S36); or top-k ANN via Faiss IVF(+PQ) (S3.4). Config `type ∈ {exact, exact_flash_attn, ann, ann_flash_attn}`. **[C]**
- **Internals:** a Gaussian-kernel regression with bandwidth σ_t that shrinks over time. Equivalent to NLM with a clean reference set (§3.2) and to MAP/MMSE under a GMM with one isotropic component per patch.
- **If changed:** replacing L2 with another metric (e.g. adding gradient channels) breaks the optimality proof but could help structured images — **[H] possible extension**. ANN with small k truncates the kernel ⇒ lower diversity at high σ? Table 1 actually shows diversity unchanged, SIFID slightly worse (0.29→0.38).

#### (d) Image reconstruction `fold_rho` (R_ρ) — **[P §3.3]**
- **What:** each pixel receives contributions from all overlapping denoised patches, weighted by a Gaussian (std ρ, in units relative to patch size — **[?] verify units in code**) centred on the patch; then normalized.
- **Why:** overlapping patches disagree; ρ decides whether we average many (smooth but blurry) or trust the patch centre (sharp but possibly noisy/seamy). ρ→0 = NLM centre-pixel copying.
- **If changed:** Table 3: ρ=∞ (uniform averaging) → worst SIFID (e.g. 4.0 at patch 15), ρ=0.2 → 1.6, ρ=0.1 → 1.3. **[H]** For binary line art, averaging misaligned candidate patches creates grey "ghost" lines; this should be measurable as an increase in intermediate grey values.

#### (e) Single-scale sampling (Alg. 1) — **[P]**
- Captures statistics at patch scale but not global layout (Fig. 3: scrambled pyramids). It is used alone for text stylization (7×7 / 3×3 patch, S4.4) and inversion.

#### (f) Coarse-to-fine sampling & TwoScaleBlend (Alg. 2) — **[P]**
- **What:** run a full T-step diffusion at the coarsest scale, then at each finer scale run a full T-step diffusion whose denoised estimate is corrected by `x̂ − Blur(x̂) + Upsample(x_{s+1,0})` — high frequencies from the current scale, low frequencies from the finished coarser output.
- **Why:** at the coarsest scale (~16× smaller, patch ≈ half image) a 15×15 patch sees global layout; finer scales add texture without re-arranging layout.
- **Internals:** every scale starts from *pure noise* x_{s,T} (not from the upsampled coarse result) — guidance enters only through the blend. The blur kernel/upsampler: ResizeRight is vendored for pyramids **[C]**; blur type **[?]**.
- **If removed:** reverts to single-scale behaviour (no global structure). If the blend uses a different cutoff, the split of responsibility between scales shifts. SinDDM uses scheduled linear interpolation instead; the authors claim better cross-scale consistency (Fig. S1).
- **Note on S ⇒ number of scales:** the paper says "S=4 scales" and Fig. S18 shows s∈{3,2,1,0}, i.e. **4 scales total**, while Alg. 2's loop `s=S..0` would imply S+1. The config sets 4 scales with relative scale 0.5. **[C]** ⇒ Treat it as 4 pyramid levels; verify in code.

#### (g) Acceleration — **[P §3.5, S3]**
| Technique | Mechanism | Exactness | Effect |
|---|---|---|---|
| Fused attention | homogeneous-coordinate trick makes the denoiser exactly `SDPA(Q,K,V, scale=1/σ²)`; memory-efficient backend; pad d to multiples of 16/32/64; keep d ≲ 1000 | exact | ~1.8× and no N×N memory |
| Latent space | FLUX VAE: 8× down, 16 ch; latent patch 7×7×16 = 784 dims | changes the data space | (1/64)² = 1/4096 cost of the core step |
| ANN | Faiss IVF, n_list=√N, n_probe (40 for gigapixel), top-k=5, optional PQ (28 B) | approximate | O(N²)→O(N^1.5) |

Gigapixel recipe: exact on scales < 500×500, ANN+PQ on finer scales, T=20, η=1. **[P S3.5]**

#### (h) Applications — **[P §4.2, S4]**
- **Retargeting:** resize the coarsest-scale input to the target aspect ratio and use it to initialize x̂_{0,S}.
- **Vertical symmetry:** after every denoise, flip-copy one half.
- **Tiling:** denoise the original, the half-height circular-shift and the half-width circular-shift versions; unshift; mask-average (Fig. S3). Paper: 3 passes; code default `num_shifts: 4` (adds a diagonal shift for corners). **[C]**
- **DDIM inversion (Alg. S1):** warm start x₁ = α(1)y + σ(1)ε (otherwise the closed-form denoiser returns y exactly — a trivial fixed point), then deterministic DDIM steps up to t′.
- **Structural analogy:** invert the structure image at coarse scale s=2 using *its own* patches to t′ = T/10 or T/2; denoise with the *style* image's patches; then coarse-to-fine.
- **Text stylization:** single scale, 7×7 patches, CLIP ViT-B/32; Eq. S37 update with momentum λ=0.1; γ(t)=σ(t)/σ(t′)·γ_max; gradient masked to top-f fraction of pixels and renormalized (S38–S40); γ_max=1.0, f=0.5 for comparisons.
- **ROI:** keep the masked region fixed after each denoise, except at the finest scale.

### 2.5 Paper vs. implementation differences (from README/config only)

> Superseded by the full source audit: see the 13-item table in [docs/technical_notes.md §4](docs/technical_notes.md).

| Item | Paper | Code (README/config) | Impact |
|---|---|---|---|
| Precision | float32 timing (§3.6) | presets default **bfloat16**; Table-1 preset `paper_sinddm15.yaml` uses bf16 dtype with `matmul: highest` ("float32 bottleneck precision") | Same seed can differ visibly in bf16 — fix the preset for reproduction. **[C]** |
| Tiling shifts | 3 passes | `num_shifts: 4` | Use `tiling.num_shifts=3` to match the paper. **[C]** |
| Scale count | "S=4" | `num_scales` 4, `min_short_side: null` (fixed depth) for Table 1; other presets derive depth from size | Use the paper preset for Table 1. **[C]** |
| σ_min | not stated | `sigma_min: 1e-4` | Clarifies the last step. **[C]** |
| Gigapixel | n_probe=40 | 40 (moon/Tokyo), 20 (painting) | Minor. **[C]** |

A full source-code audit (blur kernel, ρ units, value range, the exact form of the η update, SIFID implementation) is a Phase-1 task.

---

## 3. Technical Requirements — Models & Architectures

| Component | Role | Required? | Source | Access |
|---|---|---|---|---|
| Closed-form denoiser | The whole generative model | Yes | Code `efficient_sid/` | MIT |
| FLUX.1 VAE (AutoencoderKL, 16-ch) | Latent-space acceleration only (≥ ~512 px) | Optional | `black-forest-labs/FLUX.1-schnell` (`vae` subfolder) via `diffusers` | **Gated on HF** — accept terms + `HF_TOKEN`; pixel presets need no login **[C]** |
| CLIP ViT-B/32 | Text-guided stylization loss | Optional | `clip-anytorch` 2.6.0 (OpenAI weights) | Public |
| Faiss IVF/PQ | ANN acceleration | Optional | `faiss-gpu` 1.9.0 (conda) | **Linux-only GPU builds** — matters on Windows |
| InceptionV3 | SIFID metric | Eval | SinGAN SIFID code / pytorch-fid weights | Public |
| LPIPS (AlexNet/VGG) | Diversity, perceptual distance | Eval | `lpips` package | Public |
| NIQE/NIMA/MUSIQ | No-reference IQA | Eval | `pyiqa` (IQA-PyTorch) **[R]** — the paper doesn't say which implementation **[?]** | Public |
| SinDDM | Primary training-based baseline | Baseline | github.com/fallenshock/SinDDM (Python 3.8, torch 1.13; pretrained models for some example images) | Public |
| Stable Diffusion 1.5 (or SDXL) + LoRA | Secondary fine-tuning baseline | Baseline | `stable-diffusion-v1-5/stable-diffusion-v1-5` (HF), diffusers DreamBooth-LoRA script | Public (licence: CreativeML OpenRAIL-M) |

**Is it architecture-specific?** No. The method is backbone-free. The only "architecture" choice is the *space the patches live in*: RGB pixels or a VAE latent. Any VAE could be swapped in (e.g. the SD VAE with 4 channels). **[D]** Reproducibility depends on the data space and the hyperparameters, not on any checkpoint — a strong reproducibility property worth stating in the report.

---

## 4. Dataset Strategy

### 4.1 Datasets used by the paper

| Dataset | Source | # | Resolution | Content | Licence | Our use |
|---|---|---|---|---|---|---|
| **sinddm15** (Table 1 benchmark) | `python datasets/download_data.py uncond/sinddm15` (~1.5 MB) **[C]**; originally from the SinDDM repo; images from Places50 + SIGD16 **[P S5]** | 15 | ≈250 px max side (Table 1 timing at 186×248) | Natural scenes: mountains, pyramids, balloons, lakes, etc. | Third-party, **non-commercial**, credits in `datasets/README.md` **[C]** | Track 1 reproduction (must) |
| Figure inputs (retarget, symmetry, tiling, analogy, text style) | `download_data.py --all` (~450 MB, 11 sets) **[C]** | small | varied | natural + textures | non-commercial | Qualitative reproduction (subset) |
| Gigapixel inputs | `uncond/gigapixel` (394 MB) | 3 | 308 MP–1.2 GP | painting, moon, Tokyo | NASA PD / CC BY-NC(-ND) | **Skip** (needs a ~96 GB GPU per README) |

No large training dataset is involved anywhere in the method.

### 4.2 Candidate OOD domains — suitability analysis

The key question for single-image generation is not "is it OOD?" but **"does the image have enough internal self-similarity that 'a new image with the same patches' is a meaningful target?"** If not, failure is trivially guaranteed and uninformative.

| Domain | Candidate source | Visual characteristics | Failure modes it probes | Self-similarity | Access / licence | Verdict |
|---|---|---|---|---|---|---|
| **Technical line drawings: floor plans** | **FloorPlanCAD** (floorplancad.github.io; ICCV 2021; >10k drawings in **SVG**; also mirrored on HF via Voxel51) | Thin 1–3 px lines on a white background, Manhattan geometry, repeated symbols (doors, furniture), junctions, some text | Line straightness and continuity, junction topology, rare-symbol dropping, ghosting from averaging, background-dominated patch set | High (walls, repeated rooms and symbols) | CC BY-NC 4.0 annotations; the authors don't own the drawings' copyright; project site shut down in 2022 — **download early** | **Strongly recommended.** SVG ⇒ we control rasterization resolution and line width. |
| **Regular geometric patterns / textures** | **DTD** (robots.ox.ac.uk/~vgg/data/dtd/; 5,640 images, 47 classes × 120); use classes *chequered, grid, honeycombed, lined, zigzagged, polka-dotted, interlaced, braided*; plus Islamic geometric tilings / tessellations from Wikimedia Commons (PD/CC) | Strict or near periodicity, lattices, symmetric motifs | Lattice dislocations / phase breaks at stitching boundaries, distorted periodicity, broken symmetry | Very high | DTD: "research purposes" (mirrors differ — verify); Wikimedia per-file licences | **Strongly recommended.** Expected to be the method's best OOD case — a needed contrast. |
| **Circuit schematics** | Wikimedia Commons category "Circuit diagrams" — **curated by hand** | Lines + symbols + text labels; connectivity carries meaning | Connectivity breaks, symbol corruption, glyph soup | Medium | Per-file PD/CC | **Recommended, small (4–6 images).** |
| **Scientific diagrams with text** | **AI2D** (Allen AI, ~5k grade-school science diagrams) **[?] verify licence and image quality** | Mixed illustration + arrows + text labels | Semantic failure: text and arrows recombined meaninglessly | Low–medium | Verify | **Optional "hard negative" (3–4 images)** — documents the limit; don't over-invest. |
| **Scientific micrographs / planetary surfaces** | NASA imagery (public domain); open microscopy collections **[?]** | Quasi-stochastic texture, natural-like | Mostly a control: near-natural OOD | High | NASA PD | **Optional control (3–4 images).** |
| Charts / plots, UI screenshots, maps | — | Mostly unique, non-repetitive, semantic content | Trivially fail (nothing to recombine) | Low | — | **Not recommended** — expected failure without diagnostic value. |

### 4.3 Recommended strategy: curated real set + procedural synthetic set **[R]**

**Set A — OOD-Real (curated, ~24 images).** 8 floor plans (FloorPlanCAD SVG → raster), 8 geometric patterns (5 DTD + 3 Wikimedia tilings), 5 circuit schematics, 3 text-heavy diagrams (AI2D). Each image gets a manifest row: `id, category, source_url, licence, original_size, processed_size, line_width_px, edge_density, ink_fraction, periodicity_score, patch_rarity_entropy`. These complexity descriptors are computed automatically and used later as regression covariates in the failure analysis.

**Set B — OOD-Synthetic (procedural, ~40 images, fully controlled).** Generated by our own script (`src/sidx/data/synthetic.py`), with one factor varied at a time:
| Factor | Levels |
|---|---|
| Line thickness | 1, 2, 4, 8 px |
| Structural density (lines per 100 px) | 2, 5, 10 |
| Line length vs patch size | short segments (< 15 px) vs long lines spanning the image |
| Periodicity | perfect lattice / jittered lattice (σ=2 px) / random |
| Junction types | none / T / X junctions |
| Text glyphs | none / sparse labels |

Why synthetic: it lets us attribute failures to specific causes (e.g. "fragmentation appears only when lines are longer than the patch *and* thinner than 2 px") rather than to uncontrolled image differences. Ground-truth structural properties are known exactly (e.g. every line is straight, every junction has degree 3/4).

**Set C — Natural (sinddm15).** Reproduction and the in-distribution reference point.

**Preprocessing (all sets).**
- RGB, max side **256 px** for the main experiments (matches the paper's ~250 px regime and the pixel-exact preset).
- 512 px for the resolution ablation (pixel exact with flash attention, and latent mode).
- SVG rasterization with `cairosvg` or `resvg` at a fixed DPI so the line width is controlled. Anti-aliasing on (document it).
- Floor-plan crops: choose a 256×256 or 512×512 crop region containing ≥2 rooms (store crop coordinates).

---

## 5. Hardware and Compute Requirements — **free Google Colab only**

**Constraint (decided 2026-11-13): zero budget.**
- All GPU work runs on **free Colab**.
- The local laptop (Intel Iris Xe, no NVIDIA GPU, CPU-only PyTorch) is used for writing code, data preparation, CPU unit tests and evaluation.

### 5.1 What free Colab gives us (and its limits)

| Resource | Free Colab | Implication |
|---|---|---|
| GPU | Usually **NVIDIA T4, 16 GB**, Turing (sm_75). No native bfloat16 | Run everything in **float32** |
| Session length | ≤ 12 h; disconnects after ~90 min idle **[E]** | Every job must be **resumable** and sized to ≤ ~2 h |
| GPU quota | Not published; varies from day to day. Heavy use leads to temporary "no GPU available" **[E]** | Spread the work across days; ~5–8 GPU-h per week is a safe target **[E]** |
| RAM / disk | ~12 GB RAM, ~100 GB ephemeral disk (wiped every session) **[E]** | Re-download models each session; keep only results on Drive |
| Persistent storage | Google Drive, 15 GB free | Store outputs, manifests, LoRA weights and SinDDM checkpoints there — **not** model downloads |
| CPU-only runtime | Doesn't consume GPU quota | Run metric evaluation on a CPU runtime or the laptop |

**Team multiplier.** Each team member using **their own** free account adds their own quota. Split the work by lane (§13). Don't create extra accounts per person — that breaks Colab's terms.

### 5.2 Per-workload feasibility on a T4

| Workload | Fits 16 GB? | Speed on T4 | Verdict |
|---|---|---|---|
| Training-free, ~250 px, pixel exact, fp32 | Yes. Dense N×N ≈ 6.5 GB at 186×248; ≈ 13.7 GB at 256×256 ⇒ use `query_chunks: 4` **[D]** | **~10–20 s/sample [E]** (A6000: 3.1 s **[P]**) | ✅ main workhorse |
| Training-free, 128 px (synthetic sweeps) | Yes | ~16× cheaper than 256 px (O(N²)) **[D]** | ✅ use for failure isolation |
| Training-free, 512 px pixel | Dense needs ~220 GB ⇒ only chunked or flash attention; slow **[D]** | minutes/sample **[E]** | ⚠️ only for AB5, a few samples |
| Latent mode (FLUX VAE) | Yes | fast | ✅ optional (AB5); needs a free HF account + gated-model acceptance |
| ANN (faiss) | `pip install faiss-gpu-cu12` **[?]** — verify in Phase 2 | — | ✅ optional (AB6) |
| **SinDDM training** | Yes (1.1 M params) | ~7 h/image on an A6000 **[P SinDDM]** ⇒ likely **15–25 h on a T4 [E]** | ❌ full budget infeasible ⇒ **reduced budget** (§9, Phase 8) |
| SinDDM sampling (3 pretrained models) | Yes | seconds | ✅ |
| DreamBooth-LoRA, SD 1.5, fp16, gradient checkpointing, 8-bit Adam | Yes **[E]** | ~20–40 min/image for ~600 steps **[E]** | ✅ on ~8 images |
| SDXL / FLUX LoRA | Tight / no | — | ❌ dropped |
| Gigapixel | No (needs ~96 GB **[C]**) | — | ❌ out of scope |

### 5.2b Free Colab TPU (v5e-1) — evaluated 2026-11-13

The free Colab TPU is now a **v5e-1** (1 chip, 16 GB HBM). Colab replaced the free v2-8 with it in 2025, and availability is dynamic, as for the T4.

On paper it is much faster than a T4 for bf16 matmuls (v5e ≈ 197 TFLOP/s bf16 vs T4 ≈ 65 fp16 / 8 fp32 TFLOP/s) **[E — vendor peak numbers, not measured]**. But it only helps code that runs through PyTorch/XLA (or JAX) without host syncs.

| Workload | On TPU? | Why |
|---|---|---|
| Paper's method (training-free) | **No** | Nothing to train. The code is CUDA-only (`.cuda()`, CUDA autocast, SDPA backend selection, faiss-gpu), and it already fits the T4 budget |
| **SinDDM training** (the bottleneck) | **Try it** | A plain PyTorch conv net. Our port now has `--device xla` (sync-free loop, one graph per step, CPU-tensor checkpoints). **Adopt only if ≥ 2× faster than the T4** in notebook 07 |
| LoRA (SD 1.5) | **No** | ~20–40 min/image on a T4 already; diffusers' DreamBooth-LoRA script is CUDA-oriented |
| Evaluation | No | Runs on CPU |

Caveats:
- TPU matmuls default to bf16 passes, while the T4 trains in fp32. Notebook 07 also times `--xla_matmul_precision highest`.
- torch_xla must match torch exactly (pinned in notebook 07).
- **Measured (notebook 07, 2026-11-13): the TPU is 0.55–0.79× the T4's speed for SinDDM, so it is not used** (Phase 2 status).

### 5.3 Free-tier GPU budget (T4-hours) **[E — recalibrated in Phase 2]**

| Item | Size | T4-h |
|---|---|---|
| Phase 2 smoke tests + timing calibration | — | ~1 |
| Track 1 reproduction | 15 images × 50 samples (~250 px), Fig. 8 subset (5 images), Table 3 subset (5 images, single-scale, 5 samples) | ~5 |
| Track 2 OOD | Set A 24 images × **30** samples at 256 px; Set B 40 synthetic × 20 samples at **128 px**; retarget/tiling on 6 images | ~6 |
| Ablations AB1–AB7 | **6-image** subset × ~25 configs × **10** samples | ~8 |
| Failure isolation F1–F5 | Synthetic, 128 px | ~2 |
| LoRA baseline | **8 images** (train + 30 samples each) | ~5 |
| SinDDM baseline | 3 pretrained (sampling only) + **3 OOD images at reduced budget** | ~12 |
| **Total** | | **≈ 40 T4-h over ~8 weeks ≈ 5 h/week** (less per person with 2–3 accounts) |

If Phase 2 measures T4 speed at more than 1.5× these estimates, cut sample counts first (50 → 30 for Track 1), then ablation configurations — never the image count of Track 1.

### 5.4 Session discipline (makes free Colab workable) **[R]**
1. **Small jobs.** Each notebook run processes a queue of (image, config, seed) jobs and stops cleanly after ~100 min.
2. **Resumable.** Outputs go straight to `Drive/CS787/outputs/<run_hash>/…`; a finished `seed_XXX.png` + `meta.json` is skipped on rerun.
3. **Checkpointed training.** SinDDM and LoRA save a checkpoint to Drive every ~15–20 min and resume from the latest one.
4. **Evaluation off the GPU.** Metrics run on a CPU runtime or the laptop, reading from Drive (synced with Google Drive for desktop).
5. **Precision.** Override the presets to `precision.dtype=float32 image_denoiser.patch_denoiser.bottleneck_dtype=float32` on the T4. This is also closer to the paper's float32 timing setup (notes D2/D3).

---

## 6. Software and Dependency Stack — Colab

Colab ships its own Python (3.12 at the time of writing **[E]**) and a recent CUDA build of PyTorch. **We use Colab's preinstalled torch** rather than reproducing the official conda environment, and pin only what we add.
- This differs from the official env (Python 3.10, torch 2.8). The official code uses only standard torch APIs (SDPA, autocast, unfold/fold), so it is expected to work **[E]**.
- Phase 2 verifies this with the Quickstart smoke test, and every run records the exact versions.

### 6.1 Notebook setup cell (one per notebook)
```bash
# --- mount Drive (outputs + checkpoints persist here) ---
# from google.colab import drive; drive.mount('/content/drive')

# --- code: clone our repo (GitHub, or copy from Drive) + the official code at the audited commit ---
git clone https://github.com/haojunqiu/efficient-SID-code /content/efficient-SID-code
git -C /content/efficient-SID-code checkout 85381e3c6f29fc07144da854d86be07150404716
pip install -q -e /content/efficient-SID-code --no-deps
pip install -q omegaconf==2.3.0 "setuptools<70" clip-anytorch==2.6.0   # clip is REQUIRED: config.py imports it (Phase-2 finding)
# optional, per notebook:
pip install -q diffusers==0.35.1 transformers==4.53.2 accelerate==1.9.0   # latent mode / LoRA
pip install -q faiss-gpu-cu12                                              # AB6 only — verify it imports
# evaluation (also on CPU runtime / laptop):
pip install -q lpips pyiqa scikit-image opencv-contrib-python-headless scipy pandas matplotlib cairosvg
# record the environment
pip freeze > /content/drive/MyDrive/CS787/env_logs/$(date +%F_%H%M)_freeze.txt
```

### 6.2 SinDDM on Colab
- Its pinned stack (Python 3.8, torch 1.13.1 + CUDA 11.7 **[C]**) **cannot be installed on Colab's Python 3.12**.
- **We port it to Colab's torch.** Expected changes are small (deprecated APIs, `torch.load` defaults) **[E]**. Every change goes in `external/SinDDM_port.patch` and is listed in the report.
- **Validation of the port:** sample from the 3 provided pretrained models (`forest`, `night_sky`, `pyramids`) and compare visually against the repo's example outputs.
- **Resume support:** the trainer must save and restore `step`, the optimizer, the scheduler and EMA in each checkpoint (`SinDDM/trainer.py` `save`/`load`). Add this in Phase 2 if it is missing.

### 6.3 LoRA on Colab
- Use `diffusers/examples/dreambooth/train_dreambooth_lora.py` **from the same git tag as the installed `diffusers`**, plus `peft` and `bitsandbytes`.
- Flags: `--mixed_precision fp16 --gradient_checkpointing --use_8bit_adam --resolution 512 --train_batch_size 1 --checkpointing_steps 100`.
- Base model `stable-diffusion-v1-5/stable-diffusion-v1-5`, downloaded each session (~4 GB, not stored on Drive).

### 6.4 Local laptop (Windows, CPU)
- Python 3.12 + CPU torch (already installed: torch 2.11.0+cpu), plus the evaluation libraries above.
- Used for: our `sidx.core` re-implementation and its unit tests on tiny images (the official code is CUDA-only — notes Q10), data preparation, the synthetic generator, metrics, plots and writing.

### 6.5 Reproducibility
- Seeds per sample (`seed + i`, like the official code).
- Every `meta.json` records the GPU name, torch/CUDA versions, dtype, matmul precision, the commit hashes of our repo and the official code, and the `pip freeze` file name.
- Because Colab's torch version changes over time, the report states the versions actually used.

---

## 7. Mathematical and Technical Prerequisites

| Concept | What it is | Why relevant here | Where used | Depth needed |
|---|---|---|---|---|
| Forward diffusion & schedules | x_t = α_t y + σ_t ε | Defines noise levels at each step | Sampler, inversion | **Deep** |
| DDPM / DDIM / η stochasticity | Reverse update using x̂ and ε̂; η interpolates deterministic↔ancestral | Eqs. 3–6 are exactly this | Sampler, inversion | **Deep** |
| Flow matching / rectified flow schedule | α=1−t/T, σ=t/T linear interpolation | The schedule the paper uses | Sampler | Medium |
| Denoising ↔ score (Tweedie) | E[y|x_t] relates to ∇log p_t | Why the denoiser defines the generative model | Theory section of report | Medium |
| MMSE estimation / posterior mean | Minimizer of squared error = conditional expectation | Derivation of Eq. 2 (S2) | Report, understanding | **Deep** |
| Gaussian mixtures / kernel regression | p_t is a GMM; Nadaraya–Watson | Interpretation; GMM-prior link | Report | Medium |
| Softmax temperature | σ² acts as temperature | Soft→hard patch selection; failure analysis | Analysis | **Deep** |
| Scaled dot-product attention & FlashAttention | softmax(QKᵀ/τ)V computed in tiles | Fused acceleration; memory | Implementation | Medium (use PyTorch SDPA) |
| Patch-based synthesis (NLM, Efros–Leung, PatchMatch, GPNN) | Non-parametric patch recombination | Positioning and baselines | Related work | Medium |
| Image pyramids / Laplacian blending | Multi-scale decomposition | TwoScaleBlend | Implementation | Medium |
| Approximate NN (IVF, PQ) | Coarse quantizer + probing | ANN acceleration | Ablation only | Light |
| Latent diffusion / VAEs | Encode to compressed latent | Latent mode | Resolution ablation | Light–medium |
| CLIP guidance | Gradient of image–text cosine | Text stylization | Qualitative reproduction only | Light |
| DreamBooth / LoRA / classifier-free guidance | Fine-tune a T2I model; low-rank adapters; CFG at sampling | **Baseline only** — CFG is *not* part of the paper's method | Track 3 | Medium |
| Metrics: SIFID, LPIPS, SSIM, NIQE/MUSIQ, sliced Wasserstein | Distribution / perceptual / structural measures | Evaluation | All tracks | **Deep** for SIFID & LPIPS (know their biases) |
| Line / skeleton / Fourier analysis | LSD, skeletonization, power spectra | OOD structural metrics | Track 2 | Medium |

---

## 8. Proposed Experimental Methodology

### Track 1 — Reproduction

**Strategy decision: how to implement.**

| Option | Pros | Cons |
|---|---|---|
| (1) Use the official code only | Fast, faithful | Shows no understanding; harder to modify for experiments |
| (2) Re-implement from scratch only | Full understanding | Risk of silent discrepancies; slower |
| **(3) Hybrid [R]** — re-implement the *core* (pixel-space exact denoiser, fold_rho, pyramid, blend, sampler; ~300 LoC) and **validate it numerically against the official code** (fp32, same seed, same noise); use official code for ANN, latent, CLIP apps | Understanding + trustworthy numbers; our core is easy to instrument (patch-selection logging, per-scale dumps) | Slightly more work |

**Protocol.**
1. **Inputs:** sinddm15 at their native sizes.
2. **Config:** `configs/uncond/paper_sinddm15.yaml` (frozen Table-1 preset: T=10, η=0, patch 15, stride 1, ρ=0.2, 4 scales × 0.5, flow_linear, σ_min 1e-4, exact).
3. **Generate** 50 samples/image, seeds `10+i`.
4. **Metrics** exactly as the paper: SIFID, NIQE, NIMA, MUSIQ, pixel diversity, LPIPS diversity, inference time; mean±std over images.
5. **Equivalence test:** our core vs official in fp32 → max abs pixel diff per step < 1e-4 (or < 1 grey level in the final image, the README's own tolerance).

**Experiments to reproduce (priority order).**
| ID | Target | Pass criterion |
|---|---|---|
| R1 | Table 1 "proposed (T=10, η=0)" row (SIFID 0.29, LPIPS-div 0.49, pixel-div 0.15, NIQE 8.08, NIMA 4.53, MUSIQ 55.41) | *Revised after Phase 1:* the metric implementations are unpublished, and the two papers disagree on SinDDM's numbers (notes §5.2). Criterion: (a) the same ranking/trends vs. SinDDM and across our settings; (b) after calibrating metric variants on the 3 SinDDM pretrained models, the ratios to SinDDM match Table 1 within ~15 %. SIFID's std (±0.39) exceeds its mean, so compare **per image**. |
| R2 | Table 1 ANN (k=5) and T=40, η=1 rows | Same trend: ANN faster, SIFID slightly worse; T=40/η=1 better SIFID, lower diversity |
| R3 | Fig. 3 (single-scale scrambles layout) and Fig. 4 qualitative | Visual match |
| R4 | Fig. 8 (SIFID vs T for η ∈ {0, .5, .75, 1}), subset T ∈ {2, 5, 10, 20, 40} | Plateau by ~T=10 |
| R5 | Table 3 subset (patch ∈ {7, 11, 15, 23} × ρ ∈ {∞, 1.0, 0.2, 0.1}), single-scale, 5 samples | Same ranking: ρ small + patch 11–15 best |
| R6 | Table S1 near-duplicate rate (ours only; GPNN optional) | ≈0 % at r ≥ 85 % |
| R7 | Inversion → reconstruct sanity (Fig. S4 top) | PSNR > 35 dB **[E]** |
| R8 | Runtime vs resolution (Table 2 columns 256²–1024², vanilla vs flash vs latent) on our GPU | Same ordering; absolute values hardware-dependent |
| R9 | Qualitative: retarget, symmetric, tileable, analogy, 1–2 text prompts | Visual plausibility |

**Metric implementation risk:** the paper doesn't name its SIFID/IQA implementations **[?]**. Use SinGAN's SIFID code and pyiqa. If absolute values deviate but *relative* trends hold, document it — re-running SinDDM's pretrained models through the *same* metric pipeline gives an internal calibration point (SinDDM Table 1: SIFID 0.48, LPIPS-div 0.36).

### Track 2 — Out-of-Distribution Generalization

**Tasks:** (i) unconditional generation (main, quantitative); (ii) retargeting to 1:2 and 2:1 (structure under deformation); (iii) tileable generation on patterns (periodicity is the point); (iv) structural analogy *floor plan → texture style* (qualitative); (v) optional: inversion-reconstruct as an upper bound of fidelity.

**Protocol:** Set A at 256 px (30 samples/image), Set B at 128 px (20 samples/image), paper preset with `num_scales: 4` fixed, fp32 on a Colab T4. Ablations use 10 samples/image.

**Metric design — separating "similar to the input" from "a good variation".**

The core principle: **a good variation should match the input's *distribution* of local structures, not its pixels.** So we evaluate in five groups.

| Group | Metric | What it measures | Penalizes meaningful edits? | Caveats |
|---|---|---|---|---|
| **G1 Distribution fidelity** | SIFID | Inception early-layer feature distribution distance | No | Inception was trained on natural images; questionable on line art → report but don't rely on it alone |
| | **Multi-scale patch SWD [R]** — sliced Wasserstein distance between raw-pixel 7×7/15×15 patch sets at 3 scales | Domain-agnostic patch-distribution match | No | Sensitive to the patch size choice |
| | **Patch precision / recall [R]** — precision = % of output patches whose NN distance to input patches < τ (τ = 95th pct of input self-NN distances excluding ±p overlap); recall = % of input patches that are the NN of ≥1 output patch | Precision: no invented garbage. Recall: no dropped rare details. | No | τ calibration |
| **G2 Structural validity (reference-free / statistic-matched)** | Line statistics via LSD (OpenCV): Wasserstein distance of segment-length and orientation distributions vs the input; **mean segment length ratio** | Straightness, fragmentation | No (compares distributions) | LSD parameters must be fixed |
| | **Skeleton endpoint excess [R]**: (endpoints per 1000 skeleton px)_out ÷ (same)_in after binarize+skeletonize | Broken lines / dangling stubs | No | Binarization threshold (Otsu) |
| | Junction degree histogram (T/X/L) distance | Topology of connections | No | — |
| | **Grey-level bimodality / ghosting**: fraction of pixels with intensity in [0.2, 0.8] for binary-ish inputs, ratio out/in | Blur and ghost lines from patch averaging | No | Only for line art |
| | Edge density ratio (Canny) and mean edge gradient magnitude | Missing or extra edges; sharpness | No | — |
| | **Periodicity**: peak positions and peak-height ratio in the 2-D power spectrum; autocorrelation secondary-peak height | Lattice preservation vs dislocations | No | Only for periodic set |
| | Synthetic-only exact checks: % of detected lines within ±1° of the generator's allowed orientations; deviation from straight-line fit (px RMS) | Ground truth is known | No | — |
| **G3 Diversity / novelty** | LPIPS diversity, pixel diversity (paper protocol) | Variation between samples | — | High diversity can mean noise → read together with G1/G2 |
| | Near-duplicate rate (Table S1 protocol) | Copying the whole input | — | — |
| | **Large-block copy ratio [R]**: % of the output covered by ≥48×48 blocks matching an input region within 5/255 | Verbatim copying of large regions | — | — |
| **G4 Perceptual quality** | NIQE, MUSIQ, NIMA | Paper continuity | — | Natural-image-trained; **unreliable for diagrams** — report, de-emphasize |
| **G5 Reference-based (only where a pixel-aligned target exists)** | PSNR/SSIM/LPIPS | Inversion-reconstruction; structural analogy: SSIM between the low-pass/edge map of the output and the structure image | — | **Never** used to score unconditional variations |

**Qualitative protocol.**
1. For each category, show the input + 6 random samples (seeds fixed; no cherry-picking: show seeds 0–5) + 1 best + 1 worst by the composite structural score.
2. Zoom-in crops at junctions and at line ends.
3. **Blind human rating [R]:** 3 raters, 60 image pairs, 4 criteria on a 1–5 scale (structural correctness, detail preservation, plausibility as "another drawing of the same kind", novelty). Report Krippendorff's α.
4. Build a **failure taxonomy** with examples: fragmentation, ghosting, rare-symbol dropout, lattice dislocation, glyph soup, layout collapse, copy-paste memorization.

### Track 3 — Training-based baselines (see §9 for the selection)

**Common protocol.**
- Same input images and the same processed resolution.
- 30 samples/image; same seed lists.
- Same GPU model; same evaluation code.
- Metrics G1–G4 plus cost: training wall-clock, inference wall-clock per sample (`torch.cuda.synchronize`), peak VRAM (`torch.cuda.max_memory_allocated`), GPU-hours, disk footprint.

**Cost reporting.**
- Time-to-first-sample.
- Time for 50 samples.
- **Amortization crossover [D]:** n* = T_train / (t_ours − t_baseline). With paper numbers (A6000): 28,800 s / (3.09 − 1.25 s) ≈ **15.7 k samples** before SinDDM's per-sample advantage pays back its training; versus our ANN variant (0.88 s) it never does. Recompute with our measurements.

---

## 9. Fine-Tuning Baseline Selection

| Candidate | Task match | Same data regime | Compute | Code | Fairness | Verdict |
|---|---|---|---|---|---|---|
| **SinDDM** (train-from-scratch multi-scale DDPM on one image) | **Exact** (same task, metrics, images) | Yes — one image, nothing else | ~7–8 h/img on A6000 (~15–25 h on a T4 **[E]**) ⇒ **free tier: 3 pretrained + 3 OOD images at a reduced budget** | Official, pretrained models for 3 images | Highest; the paper already compares to it | **Primary [R]** |
| SinFusion | Same task | Yes | ~3.2 h/img | Official | High | Alternative if SinDDM's environment fails |
| **DreamBooth-LoRA on SD 1.5** | Different task (subject personalization), text-conditioned | Uses a prior pretrained on billions of images; optionally ~100–200 regularization images | 10–20 min/img | diffusers example | Partial — answers "does a large pretrained prior + fine-tuning beat a patch prior on technical images?" | **Secondary [R]**, on a subset |
| Full DreamBooth (UNet fine-tune) | as above | as above | heavier, 24 GB+ | diffusers | No advantage over LoRA for this question | No |
| FLUX LoRA | as above; shares the FLUX VAE with our latent mode | — | 24–48 GB, slow | community | Interesting "same-VAE" control but expensive | Only if compute is abundant |
| SinGAN | Same task, GAN | Yes | ~2 h | Official | Paper already covers it | Not needed |

**Why SinDDM is the scientifically fair primary baseline.**
- It isolates the variable we care about — *closed-form vs learned denoiser* — while keeping the task, data (one image), multi-scale design, and evaluation identical.
- Its "training" is the conventional way to get this capability, so it directly answers "training-free vs training-based".

**Why add LoRA anyway.** Your brief's motivation ("conventional fine-tuning") points to it, and it tests a different, interesting hypothesis: **[H] a large pretrained prior produces globally coherent, "clean" line drawings but low patch precision (hallucinated symbols/styles not in the input) and drifts from the input's patch distribution**; the patch method shows the opposite profile.

**LoRA configuration [R].**
- `train_dreambooth_lora.py`, SD 1.5, rank 8, lr 1e-4, 600 steps, batch 1, 512 px (input resized/padded from 256 → 512, outputs downsampled back to 256 for evaluation — document this resolution mismatch).
- Prompt: "a drawing in the style of sks" or "a sks floor plan".
- **No** prior preservation in the main runs (so it uses one image only, like ours). A prior-preservation variant (100 class images generated by SD 1.5 itself) is **dropped on the free tier**; mention it as a limitation.
- Sampling: 30 DDIM steps, CFG 7.5.
- Tune steps/lr on a **held-out dev set of 2 images** and freeze before running the test images — the same rule applies to our method's hyperparameters.

**Fairness checklist.**
- Same input images and processed size.
- Equal sample counts and seed lists.
- Hyperparameters tuned on the same dev split, with equal tuning effort (count the configurations tried).
- Same GPU, with precision reported.
- Identical evaluation code.
- Report both "quality at default settings" and "quality per GPU-hour".
- Disclose every extra resource used by each method: pretrained weights (≈1 B parameters for SD 1.5 incl. text encoder; ours: 0 for pixel mode), regularization images, text prompts.

---

## 10. Ablation Study Design

Keep it small: **7 ablations**, run on a fixed **ablation subset** = 6 images (2 floor plans, 2 patterns, 1 circuit schematic, 1 natural) + the 128-px synthetic set where marked, 10 samples each (free-tier budget). AB5 and AB6 are optional.

| ID | Changed | Why | Values | Expectation **[H]** | Metrics |
|---|---|---|---|---|---|
| AB1 | Patch size p | Structure vs patch scale | 7, 11, 15, 23 | Larger p preserves longer line segments but reduces diversity (more copying) | Patch P/R, endpoint excess, LPIPS-div, copy ratio |
| AB2 | Fold ρ | Ghosting from averaging | 0.1, 0.2, 1.0, ∞ | Large ρ ⇒ ghost lines and grey pixels on line art; small ρ ⇒ sharper but more seams | Bimodality, endpoint excess, SIFID/SWD |
| AB3 | T × η | Soft→hard selection schedule | T ∈ {10, 40} × η ∈ {0, 1} | η=1 improves continuity (more correction steps) but lowers diversity | Full G1–G3 |
| AB4 | Number of scales / coarsest size | Global layout & aliasing | 2, 3, 4, 5 scales | Thin lines vanish at the coarsest scale when it's too small ⇒ layout errors | Edge retention at s=S; junction histogram |
| AB5 | Data space: pixel vs FLUX-latent (512 px input) | VAE effect on thin lines | pixel-exact vs latent-exact | Latent smears 1-px lines / text at small sizes; fine at high res | Bimodality, LSD stats, runtime |
| AB6 | Exact vs ANN top-k | Accuracy–speed trade-off on structured data | exact, k=5, k=20 | ANN hurts rare patches more (recall ↓) | Patch recall, runtime |
| AB7 | Stride of patch dataset | Translation coverage | 1, 2, 4 | Stride > 1 ⇒ jagged / misaligned lines | LSD straightness, endpoint excess |

### Failure-isolation experiments (the "why")

| ID | Experiment | Isolates |
|---|---|---|
| F1 | Synthetic sweeps (thickness × density × length × periodicity) at default settings; fit a regression of each structural metric on the factors | Which image properties cause failure |
| F2 | **Patch-selection analysis** (extend Fig. S19): log which input patch wins the argmax at t=0 per location; correlate selection frequency with patch *rarity* (kNN density in patch space) | Rare-detail dropout mechanism |
| F3 | Per-scale dumps (Fig. S18 style): measure line continuity at each scale's output x_{s,0} | Where in the pyramid breakage appears (coarse layout vs fine stitching) |
| F4 | Blend/no-blend at fine scales; blur cutoff variation | Whether TwoScaleBlend propagates or repairs coarse errors |
| F5 | Background-fraction sweep (pad the same drawing with more white space) | Dominance of background patches in the softmax |
| F6 (optional, extension) | One targeted intervention motivated by F1–F5, e.g. (a) rarity-reweighted patch prior, (b) gradient channels added to the patch vector, (c) ink-preserving (min/max-pool) downsampling for the pyramid | Causal test of the strongest hypothesis |

Recorded for every run: config hash, all G1–G4 metrics per sample, runtime, peak VRAM.

---

## 11. Software Architecture and Repository Structure

```
Project/
├── PROJECT_PLAN.md                  # this document
├── 2606.04299v1.pdf
├── external/                        # git submodules, pinned commits
│   ├── efficient-SID-code/          # official implementation (oracle + apps)
│   └── SinDDM/                      # baseline
├── src/sidx/                        # our package ("sidx" = SID eXtended)
│   ├── core/                        # our clean re-implementation
│   │   ├── patches.py               # unfold, fold_rho (Gaussian-weighted)
│   │   ├── denoiser.py              # exact / chunked / SDPA closed-form denoiser
│   │   ├── schedule.py              # flow_linear α,σ; eta handling
│   │   ├── pyramid.py               # build pyramid, blur, upsample, two_scale_blend
│   │   ├── sampler.py               # single-scale, coarse-to-fine, inversion
│   │   └── hooks.py                 # instrumentation: per-step / per-scale dumps, argmax-patch logging
│   ├── methods/                     # uniform interface over all generators
│   │   ├── base.py                  # class Method: setup(img) -> state; sample(n, seeds) -> imgs; cost stats
│   │   ├── sid_ours.py              # wraps sidx.core
│   │   ├── sid_official.py          # wraps external/efficient-SID-code (subprocess or import)
│   │   ├── sinddm.py                # subprocess into env "sinddm"; train/sample; parses outputs
│   │   └── lora_sd.py               # subprocess into env "lora"; train + sample
│   ├── data/
│   │   ├── registry.py              # dataset name -> manifest CSV
│   │   ├── preprocess.py            # resize, crop, SVG rasterize, normalize
│   │   ├── synthetic.py             # procedural line/lattice/junction/text generator (seeded)
│   │   └── descriptors.py           # edge density, ink fraction, periodicity, patch rarity entropy
│   ├── metrics/
│   │   ├── distribution.py          # SIFID, patch SWD, patch precision/recall
│   │   ├── diversity.py             # LPIPS-div, pixel-div, near-duplicate, block-copy ratio
│   │   ├── structure.py             # LSD stats, skeleton endpoints/junctions, bimodality, FFT periodicity
│   │   ├── iqa.py                   # NIQE/NIMA/MUSIQ via pyiqa
│   │   └── reference.py             # PSNR/SSIM/LPIPS (inversion, analogy only)
│   ├── eval/
│   │   ├── run_metrics.py           # folder of samples -> per-sample parquet
│   │   └── aggregate.py             # tables (mean±std per category/method), stats tests
│   ├── viz/                         # grids, zoom crops, per-scale strips, histograms
│   └── utils/                       # seeding, timing (cuda sync), memory, logging, config hashing
├── configs/                         # OmegaConf YAML, composable
│   ├── method/   {sid_paper.yaml, sid_ann.yaml, sid_latent.yaml, sinddm.yaml, lora_sd15.yaml}
│   ├── data/     {sinddm15.yaml, ood_real.yaml, ood_synth.yaml, ablation_subset.yaml}
│   ├── sweep/    {ab1_patch.yaml, ab2_rho.yaml, ...}
│   └── experiment/{track1_table1.yaml, track2_uncond.yaml, track3_compare.yaml}
├── notebooks/                       # free-Colab drivers: 00_setup_smoke, 05_sinddm_port, 10_track1, 20_ood, 30_ablations, 40_lora, 50_sinddm_train (thin: clone repo → run_queue.py)
├── scripts/                         # thin CLIs (incl. run_queue.py — resumable, time-budgeted job queue): generate.py, evaluate.py, sweep.py, make_figures.py
├── data/   {raw/, processed/, manifests/*.csv}      # gitignored except manifests
├── outputs/runs/<experiment>/<method>/<config_hash>/<image_id>/{seed_XXX.png, meta.json}
├── reports/ {figures/, tables/, report.tex|md, slides/}
├── tests/   {test_denoiser_equivalence.py, test_fold.py, test_metrics_sanity.py}
└── env/     {environment.sid.yml, environment.sinddm.yml, environment.lora.yml}
```

**Module contracts.**
| Module | Responsibility | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| `data.*` | Acquire, preprocess, describe images | raw files, manifest | processed PNG + descriptor columns | PIL, cairosvg |
| `core.*` | Closed-form diffusion | image tensor, config | samples + optional traces | torch |
| `methods.*` | **Single interface for every generator** — the key to reuse and fairness | processed image path, method config, seeds | PNGs + `meta.json` (times, VRAM, config, versions) | core / external envs |
| `metrics.*` | Pure functions: (input, samples) → numbers | arrays | dict of floats | lpips, pyiqa, cv2, skimage |
| `eval.*` | Batch metric computation & aggregation | `outputs/runs/...` | parquet/CSV, LaTeX tables | metrics |
| `viz.*` | Figures | runs + metric tables | PNG/PDF | matplotlib |
| `utils.*` | Seeds, timers, logging | — | — | — |

**Reuse between pipelines:** `data`, `metrics`, `eval`, `viz`, `utils`, and the output layout are 100 % shared. Only `methods/*` differs. Baselines that need other environments run as **subprocesses** that write to the same output layout, so the evaluator never knows which method produced a folder.

**Configuration.**
- An experiment = `data` × `method` × `sweep` (Cartesian grid) with CLI overrides (`key=value`, the same convention as the official code).
- Every run directory is keyed by a hash of the fully resolved config; re-running skips finished runs (resumable).
- Nothing in `core/` reads files or global state, so a new dataset or setting never requires code changes.

**Logging.** `meta.json` per image (config, seed list, git hashes, GPU, dtype, wall times, peak VRAM) plus a run-level CSV. W&B is optional.

---

## 12. Phase-Wise Implementation Roadmap

The project is split into **12 phases in 4 stages**, with a review **milestone (gate)** at the end of each stage. Phases in the same stage can run in parallel where marked ∥.

| Stage | Phases | Weeks | Gate at end |
|---|---|---|---|
| **I. Foundations** | 1 Paper & code analysis · 2 Environment · 3 Natural data | 1–2 | — |
| **II. Reproduction** | 4 Reproduce method · 5 Validate against paper | 2–3 | **M1: reproduction go/no-go** |
| **III. Extension experiments** | 6 OOD data & metric suite · 7 OOD experiments · 8 Training-based baselines · 9 Ablations & failure isolation | 2–8 | **M2 (wk 5): OOD review** · **M3 (wk 8): experiment freeze** |
| **IV. Analysis & reporting** | 10 Quantitative & qualitative evaluation · 11 Failure analysis · 12 Documentation & presentation | 7–10 | **M4: final review** |

---

### Stage I — Foundations

#### Phase 1 — Paper analysis, code audit & literature review — ✅ DONE (2026-11-13)

> **Output:** [docs/technical_notes.md](docs/technical_notes.md). The code was audited at commit `85381e3`; SinDDM at `e7107d9`; both are cloned into `external/`.
> **Key findings that change later phases:**
> - The code's noise grid differs from the paper's: the last denoise is at σ = 1e-4, not 0.1 (D1).
> - Neither repo nor paper contains or names the metric implementations, and the two papers' SinDDM numbers disagree. ⇒ The R1 criterion becomes trend-based plus calibration (§5.2 of the notes).
> - Only 3 SinDDM pretrained models exist.
> - `num_scales: auto` gives 5 scales at 256 px (D11).
> - The official code is CUDA-only, and this machine has no NVIDIA GPU.

- **Objective:** understand the method well enough to explain every design decision, and resolve every **[?]** item in §2 before writing code.
- **Tasks:**
  1. Read the official source: `efficient_sid/config.py` (`UncondAppConfig`), the patch denoiser (exact / flash / ANN), the fold-with-ρ implementation, the pyramid/blur/upsample code, the scheduler (`flow_linear`, η update), and the app scripts.
  2. Resolve: the units of ρ, the blur kernel used in TwoScaleBlend, the image value range, the scale-count convention (S vs S+1), how the t=0 blend skip is coded, and which SIFID/IQA implementations are used (if any are in the repo).
  3. Read SinDDM (the training baseline) and GPNN (the closest non-parametric method); skim Kamb & Ganguli 2025, Niedoba et al. 2025, NLM (Buades 2005), and EPLL (Zoran & Weiss 2011).
  4. Write the "algorithm in our own words" note with the derivation of Eq. 2.
- **Technical requirements:** none (reading only).
- **Data/resources:** the paper PDF, the GitHub repos.
- **Deliverables:** `docs/technical_notes.md` (≤ 3 pages); a resolved-[?] table; a paper-vs-code difference table (extending §2.5).
- **Dependencies:** none.
- **Time:** week 1 (∥ Phase 2).
- **Difficulties:** the code may contain undocumented heuristics; notation mismatches (S vs S+1).
- **Done when:** every [?] is resolved or explicitly logged as "unknown + how we handle it", and every team member can explain the pipeline in §2.3.

#### Phase 2 — Environment & infrastructure setup (free Colab + laptop) — 🟡 MOSTLY DONE (notebook 00 outputs to be recorded)

> **Status (2026-11-13).** How-to: [docs/colab_workflow.md](docs/colab_workflow.md).
>
> **Done and tested locally (CPU):**
> - `sidx` package: the method interface, the `sid_official` wrapper and a dummy method.
> - Resumable, time-budgeted runner (`scripts/run_queue.py`); 6/6 pytest pass.
> - All 6 Phase-2 timing configs parse and resolve correctly against the official code.
> - SinDDM port in `patches/sinddm_port/`:
>   - device option and safe weights-only checkpoint loading;
>   - resumable, time-budgeted training — verified on CPU with a stop at step 24 and a resume to completion, plus sampling;
>   - skimage API fix.
> - Notebooks `00_setup_smoke`, `05_sinddm_port` and `06_lora_smoke`, generated by `scripts/make_notebooks.py`.
> - Optional `07_tpu_sinddm` (free v5e-1 TPU, see §5.2b), backed by the SinDDM port's `--device xla` / `--sync_free` modes. The CPU tests of those modes:
>   - the original path reproduces earlier losses exactly;
>   - the sync-free path stops on its budget and resumes correctly.
> - Local `.venv`, `.gitignore`, `env/requirements-*.txt`.
>
> **Pending — needs you on Colab:**
> - push `Project/` to GitHub;
> - run the 3 notebooks;
> - paste the timing tables into §5.3.
>
> **Not yet verified:** the `sid_official` wrapper needs a GPU and is first exercised by notebook 00.
>
> **Measured on Colab (2026-11-13, T4, torch 2.14.1 + CUDA 13.0):**
> - Notebook 05 (SinDDM port): the 3 pretrained models sample; training 1 395 s / 1 000 steps at 248×186 (≈ 46.5 h for the full 120k schedule) and 712 s / 1 000 steps at 171×128 (≈ 23.7 h); stop/resume works.
> - Notebook 06 (LoRA, SD 1.5): 50 steps in 191 s incl. model download; sampling 5.9 s/image; 2.6 GB peak VRAM.
> - Notebook 07 (free **v5e-1 TPU**, torch_xla 2.8, default bf16 matmul passes): SinDDM trains **slower than on the T4** — 1 765 s / 1 000 steps at 248×186 (T4: 1 395, ×0.79) and 1 297 s at 171×128 (T4: 712, ×0.55). A TPU-written checkpoint loads on CPU (step 400, CPU tensors). **Decision: SinDDM stays on the T4; the TPU path is dropped** (it remains in the port for completeness). Likely reason [H]: a 1.1 M-parameter conv net at batch 32 is far too small to use a TPU, and per-step overhead dominates.
>
> **New findings:**
> - `clip` is required even for unconditional runs.
> - `setuptools` must be `<70`, not `<80`.
> - The exact fp32 kernel needs `query_chunks ≥ 2` on a T4.
- **Objective:** a working free-Colab workflow (resumable, Drive-backed), a local CPU dev environment, and **measured T4 timings** that recalibrate the budget in §5.3.
- **Tasks:**
  1. **Code hosting:** push our project to a GitHub repo (or keep `src/` in a Drive folder) so every Colab notebook can `git clone`/copy it. Pin the official code at `85381e3` (notes §0).
  2. **Drive layout:** `MyDrive/CS787/{data, outputs, checkpoints, env_logs, tables}`.
  3. **`notebooks/00_setup_smoke.ipynb`** (GPU runtime):
     - setup cell (§6.1);
     - run the official Quickstart (`scripts/sample_uncond.py --config configs/uncond/paper_sinddm15.yaml` with `precision.dtype=float32 image_denoiser.patch_denoiser.bottleneck_dtype=float32`) on `examples/uncond/balloons.png`;
     - record seconds/sample and peak VRAM;
     - repeat at 128 px and 256×256 (with `query_chunks=4`);
     - test `exact_flash_attn` on the T4;
     - try `import faiss`.
  4. **Resolve open item D3** on the GPU: run the shipped bf16 mode once and log the dtype of the attention weights.
  5. **SinDDM port** (`notebooks/05_sinddm_port.ipynb`):
     - install it on Colab's torch and fix incompatibilities (record them in `external/SinDDM_port.patch`);
     - sample the 3 pretrained models;
     - add checkpoint resume (step, optimizer, LR scheduler, EMA);
     - **time 1 000 training steps** on a 128-px and a 256-px image.
  6. **LoRA smoke test:** a 50-step SD 1.5 DreamBooth-LoRA run with fp16, gradient checkpointing and 8-bit Adam; record time and VRAM.
  7. **Job runner** (`scripts/run_queue.py`): reads a job list (image × config × seeds), skips finished outputs on Drive, stops cleanly after a time budget (default 100 min), and writes `meta.json` (versions, GPU, dtype, time, VRAM).
  8. **Local laptop:** CPU environment for `sidx`, the evaluation libraries, and pytest.
  9. **Optional:** create a free HF account and accept the FLUX.1-schnell terms (only for AB5).
- **Technical requirements:** a Google account (free), a GitHub account (free).
- **Deliverables:** `notebooks/00_setup_smoke.ipynb`, `05_sinddm_port.ipynb`; `scripts/run_queue.py`; a **timing table** (T4 s/sample at 128/186×248/256 px, SinDDM s per 1k steps, LoRA min per 100 steps); an updated §5.3 budget.
- **Dependencies:** none (∥ Phase 1).
- **Time:** week 1–2.
- **Difficulties:**
  - the free GPU is sometimes unavailable (retry later or on a teammate's account);
  - the SinDDM port may need more fixes than expected;
  - `faiss-gpu-cu12` may not import (then AB6 becomes optional).
- **Done when:**
  - the official code produces an image on the T4;
  - a deliberately interrupted queue resumes without redoing finished jobs;
  - SinDDM pretrained sampling works;
  - the timing table exists and §5.3 is updated.

#### Phase 3 — Natural dataset acquisition & preparation — ✅ DONE (2026-11-13)

> **Output:** `scripts/prepare_natural.py` → `data/manifests/*.csv` (committed) + `data/natural/` (gitignored).
> - 59 images in 6 sets: sinddm15 (15, 167–200 × 249–298 px, i.e. the paper's ~250 px regime), retarget (14), symmetric (10), tileable (8), structural analogy (4), text style (8).
> - Each manifest row: id, path, size, **sha256**, source, licence, and 6 descriptors. `--verify` re-checks the hashes on any machine (notebook 10 does this on Colab).
> - **Descriptor set changed from §4.3:** "patch_rarity_entropy" became two measures — `patch_code_entropy` (3×3 census-code entropy: local structural variety) and `rare_patch_fraction` (share of 7×7 patches whose nearest non-overlapping neighbour is > 2× the median distance: the rare-detail risk). `periodicity_score` is computed on the high-passed image and requires repetition along **both** axes, so smooth skies and single straight lines do not count as periodic (natural images 0.03–0.20, bricks 0.45, grid texture 0.62, perfect grid 1.0).

- **Objective:** have the paper's benchmark data ready for reproduction.
- **Tasks:**
  1. `python datasets/download_data.py uncond/sinddm15`, plus the figure sets needed for qualitative checks (retarget, symmetric, tileable, analogy, text style). Skip gigapixel.
  2. Build `data/manifests/sinddm15.csv` (id, size, source, licence/credit).
  3. Compute the image descriptors (`sidx.data.descriptors`) so natural images can later serve as an in-distribution reference.
- **Technical requirements:** `sid` env.
- **Deliverables:** 15 benchmark images + figure inputs; manifest; descriptor table.
- **Dependencies:** Phase 2.
- **Time:** ~1 day in week 1–2.
- **Difficulties:** none expected.
- **Done when:** the manifest lists all images with sizes matching the paper's ~250 px regime.

---

### Stage II — Reproduction

#### Phase 4 — Reproduction of the original method — 🟡 4b DONE (our core), 4a QUEUED ON COLAB

> **4b — our core (`src/sidx/core`, ~400 lines) is implemented and validated on the laptop CPU.**
> - Modules as planned (resize, pyramid, patches, denoiser with dense/chunked/SDPA backends and winning-patch logging, schedule, sampler, hooks); inversion is deferred to Phase 5 (R7). Our resampler re-expresses ResizeRight as per-axis matrices (≤ 3·10⁻⁷ from the official one).
> - Exposed to the runner as method `sid_core`, which runs on CPU **and** CUDA (the official code is CUDA-only), so small synthetic sweeps can run on the laptop.
> - **Equivalence with the official code** (`tests/test_core_equivalence.py`, `src/sidx/core/official_cpu.py`): the official modules are assembled on CPU and both samplers run with the same seed in fp32. At 48×64, patch 7, 3 scales, η ∈ {0, 0.5, 1}: mean difference ≤ 0.003 grey levels, < 0.1 % of pixels differ by > 1 grey level (isolated near-tie flips of the σ = 10⁻⁴ hard patch pick). Full paper preset: see `reports/equivalence/`.
> - Found along the way: notes **D14** (official η > 0 noise depends on tensor memory layout) and **D15** (fp32 underflow of border weights at ρ = 0.1).
> - Tests: 38 CPU tests pass (runner 6, core 23, equivalence 5, descriptors 4).
>
> **4a — official runs:** `configs/jobs/phase4_reproduction.yaml`, run with `notebooks/10_run_queue.ipynb` over several sessions: Table-1 row (15 × 50), our core on 5 seeds (GPU equivalence), T = 40/η = 1 and ANN k = 5 variants (15 × 20 each). ≈ 1 425 samples.

- **Objective:** generate outputs with (a) the official code and (b) our own validated re-implementation of the core.
- **Tasks:**
  1. **4a — Official runs:** run `configs/uncond/paper_sinddm15.yaml` on all 15 images, 50 samples each (seeds `10+i`). Also run the ANN (k=5) and T=40/η=1 variants.
  2. **4b — Our core** (`sidx.core`, pixel space, exact): `patches.py` (unfold, fold_rho), `denoiser.py` (dense + chunked + SDPA variants), `schedule.py`, `pyramid.py` (incl. TwoScaleBlend), `sampler.py` (single-scale, coarse-to-fine, inversion), `hooks.py` (per-step/per-scale dumps, argmax-patch logging).
  3. **Unit tests:**
     - at t=T the denoiser output equals the mean patch;
     - as σ→0 it returns the nearest clean patch;
     - fold(unfold(x)) = x;
     - the dense, chunked and SDPA denoisers agree to 1e-5 in fp32.
  4. **Equivalence test** against the official code: identical input, seed and noise tensors, fp32 → final image difference < 1 grey level (the README's own tolerance).
  5. Wrap both implementations in the uniform `methods/` interface.
- **Technical requirements:**
  - Our core is developed and unit-tested **on the laptop CPU** with tiny images (32–64 px).
  - Official runs and the equivalence test run on the **Colab T4, in fp32**.
- **Data/resources:** sinddm15.
- **Deliverables:** `sidx.core` + passing tests; 750 official samples (15 × 50) with `meta.json` timing/VRAM.
- **Dependencies:** Phases 1, 2, 3.
- **Time:** week 2.
- **Difficulties:** hidden implementation details (blur, padding at borders, bf16 vs fp32 differences).
- **Done when:** the equivalence test passes, or every remaining difference is explained and documented.

#### Phase 5 — Validation against the original paper
- **Objective:** show quantitatively and qualitatively that our setup reproduces the paper (targets R1–R9 in §8).
- **Tasks:**
  1. Implement the paper's metrics: SIFID (SinGAN code), NIQE/NIMA/MUSIQ (pyiqa), pixel diversity and LPIPS diversity (SinDDM definitions), near-duplicate rate (Table S1 protocol), and wall-clock timing with `cuda.synchronize`.
  2. Compute the Table-1 row for "proposed (T=10, η=0)" and compare per image.
  3. **Calibration:** SinDDM ships pretrained models for only 3 images (forest, night_sky, pyramids). Run them through the *same* metric code, compute every metric variant (notes §5.3), and freeze the variant that reproduces Table 1's ratios between the methods.
  4. Reproduce subsets of Fig. 8 (T ∈ {2, 5, 10, 20, 40} × η ∈ {0, 1}) and Table 3 (patch ∈ {7, 11, 15, 23} × ρ ∈ {∞, 1, 0.2, 0.1}, single-scale, 5 samples).
  5. Run the inversion → reconstruction sanity check (PSNR).
  6. Produce qualitative grids: Fig. 3/4 equivalents plus one each of retarget, symmetric, tileable, analogy and text style.
  7. Measure runtime vs resolution on the T4 (128², 256², and 512² with flash attention/latent only), for vanilla / flash attention / latent. Report it as a trend, not as absolute numbers comparable to Table 2.
- **Deliverables:** `reports/tables/track1_table1.csv`, Fig. 8 and Table 3 replicas, a qualitative grid, and a 1-page reproduction report.
- **Dependencies:** Phase 4.
- **Time:** weeks 2–3.
- **Difficulties:** metric implementation mismatch (the paper doesn't name its SIFID/IQA code); SIFID's huge variance (±0.39 on a mean of 0.29).
- **Done when:** R1–R7 pass (the main metrics are within ~10–15 % of the paper, the trends match), or every deviation is explained.

> **Milestone M1 — reproduction go/no-go (end of week 3).**
> Proceed if the trends reproduce. If our core doesn't match the official code, continue with the official code as the generator and keep our core only for instrumentation. If the metrics don't match, rely on relative comparisons calibrated against SinDDM.

---

### Stage III — Extension Experiments

#### Phase 6 — OOD dataset preparation & structural metric suite
- **Objective:** a curated, licence-documented OOD dataset, plus metrics that measure structural correctness without penalizing legitimate variation.
- **Tasks:**
  1. **6a — Real OOD set (Set A, ~24 images):**
     - Download FloorPlanCAD **first** (the site is unmaintained). Rasterize the SVGs at a fixed DPI and crop 8 multi-room regions to 256×256.
     - Pick 5 DTD images (classes: chequered, grid, honeycombed, lined, zigzagged) and 3 geometric tilings from Wikimedia Commons.
     - Pick 5 circuit schematics from Wikimedia Commons ("Circuit diagrams" category).
     - Pick 3 AI2D diagrams.
     - Record source URL, licence and crop coordinates in `ood_real.csv`.
  2. **6b — Synthetic set (Set B, ~40 images):** write `sidx/data/synthetic.py` (seeded). Vary line thickness (1/2/4/8 px), density, line length vs patch size, periodicity (perfect / jittered / random), junctions (none / T / X), and text (none / sparse) — one factor at a time.
  3. **6c — Descriptors** for every image: edge density, ink fraction, periodicity score, patch-rarity entropy.
  4. **6d — Metric suite** (`sidx.metrics`): multi-scale patch SWD, patch precision/recall, LSD line statistics, skeleton endpoint excess and junction histogram, grey-level bimodality, FFT periodicity, large-block copy ratio.
  5. **6e — Metric sanity tests:** apply controlled corruptions to synthetic images (identity copy, shuffled patches, Gaussian blur, cut lines, shifted lattice phase) and check that each metric moves in the expected direction while ignoring irrelevant changes.
- **Technical requirements:** `sid` env + cairosvg/resvg and opencv-contrib.
- **Deliverables:** `data/manifests/ood_real.csv` and `ood_synth.csv`; 64 processed images; a data card; `sidx.metrics` with tests; a metric-sanity figure.
- **Dependencies:** Phase 2 (can run ∥ with Phases 4–5).
- **Time:** weeks 2–4.
- **Difficulties:** licences for real images; SVG rendering quirks; metrics that don't discriminate.
- **Done when:** the manifests are complete with licences, and every metric passes its sanity test (or is dropped).

#### Phase 7 — OOD experimentation (training-free method)
- **Objective:** Track-2 main results — how the method behaves on structured imagery, with default (paper) settings.
- **Tasks:**
  1. Unconditional generation, paper preset with `num_scales: 4` fixed, fp32:
     - Set A (256 px): 30 samples/image;
     - Set B (synthetic, 128 px): 20 samples/image.
  2. Retargeting to 1:2 and 2:1 on floor plans and patterns.
  3. Tileable generation on the pattern images.
  4. Structural analogy (floor-plan structure + texture style), qualitative only.
  5. Inversion → reconstruction as a fidelity upper bound.
  6. Compute all metric groups G1–G4 (G5 where applicable); per-category tables.
  7. First qualitative grids (seeds 0–5, no cherry-picking) and a first failure-taxonomy draft.
- **Technical requirements:** Colab T4, ~6 T4-h, run as ~4 queued sessions.
- **Deliverables:** `reports/tables/track2_*.csv`; per-category qualitative grids; failure-taxonomy draft.
- **Dependencies:** Phases 5, 6.
- **Time:** weeks 4–5.
- **Difficulties:** results may vary widely across categories; avoid over-interpreting single images.
- **Done when:** every image has its planned samples (30 or 20) and a full metric row, and the taxonomy draft has examples for each observed failure type.

> **Milestone M2 — OOD review (end of week 5).**
> Choose the 2–3 failure modes to study in depth in Phase 9. Decide whether the optional intervention (F6) is worth doing. If OOD performance is uniformly good, shift the emphasis to harder categories and to the baseline comparison. Record the decision in a short memo.

#### Phase 8 — Training-based baseline implementation
- **Objective:** produce SinDDM (primary) and DreamBooth-LoRA (secondary) outputs under a fair protocol (§9), **within the free-Colab budget**.
- **Free-tier scope (decided 2026-11-13):**
  - **SinDDM:** natural images = the **3 provided pretrained models** (sampling only); OOD = **3 images** (1 floor plan, 1 pattern, 1 circuit schematic), trained at a **reduced budget**.
  - **LoRA:** **8 OOD images**.
- **Tasks:**
  1. **8a — Selection & dev tuning:**
     - Pick the 3 SinDDM images and 8 LoRA images (stratified by category; the 3 SinDDM images are among the 8).
     - Hold out 1 dev image and tune LoRA (steps, lr, rank) on it only.
     - Freeze both configs and record how many configurations were tried for each method.
  2. **8b — SinDDM reduced-budget training** (on the free TPU instead, if notebook 07 shows ≥ 2× speed-up; the budget below is then re-scaled and may allow more images or the full schedule):
     - Train at **128 px short side** (SinDDM's own default scaling then gives fewer scales).
     - Budget: **~4 T4-hours per image** (step count set from the Phase-2 timing), split into ≤ 2-hour sessions with resume.
     - Save checkpoints at 25 / 50 / 100 % of the budget ⇒ a **quality-vs-training-time curve**. Report it as a result in its own right: "how much training does SinDDM need to match a training-free method?"
     - Run our method on the same 128-px inputs for a matched comparison.
     - Log wall-clock time and peak VRAM.
  3. **8c — LoRA training:**
     - SD 1.5, rank 8, ~600 steps, fp16 + 8-bit Adam + gradient checkpointing, no prior preservation.
     - Skip the prior-preservation variant (budget); state this in the report.
  4. **8d — Sampling:**
     - 30 samples/image per method, with the same seed lists.
     - LoRA outputs (512 px) are resized back to the input size for evaluation; document this.
  5. Wrap both as `methods/sinddm.py` and `methods/lora_sd.py`, writing to the shared Drive output layout.
- **Technical requirements:** Colab T4; ~12 T4-h (SinDDM) + ~5 T4-h (LoRA), spread over weeks 3–7.
- **Deliverables:** checkpoints (on Drive); samples; training/inference logs (time, VRAM, disk); the SinDDM training curve.
- **Dependencies:** Phase 2 (SinDDM port + resume) and 6a. Starts in **week 3**, ∥ with Phases 5–7.
- **Difficulties:**
  - interrupted sessions (mitigated by resume);
  - the reduced SinDDM budget weakens that comparison — be explicit about it;
  - the LoRA resolution mismatch.
- **Done when:** the selected images have trained models and 30 samples per method, and the fairness checklist in §9 is satisfied, including the reduced-budget disclosure.

#### Phase 9 — Controlled ablation studies & failure isolation
- **Objective:** find out *why* the method succeeds or fails, using the ablations AB1–AB7 and the isolation experiments F1–F5 (§10).
- **Tasks:**
  1. **9a — Ablations** on a **6-image** ablation subset (2 floor plans, 2 patterns, 1 circuit schematic, 1 natural), 10 samples each: patch size, ρ, T×η, number of scales, stride.
     - Pixel vs latent (AB5, 512 px) and exact vs ANN (AB6) are **optional**: run them only on 2 images, and only if the free-tier budget allows.
  2. **9b — Failure isolation:**
     - F1: synthetic factor sweeps plus a regression of each structural metric on the factors.
     - F2: patch-selection frequency vs patch rarity (using `hooks.py` argmax logging).
     - F3: per-scale continuity dumps.
     - F4: blend / no-blend at fine scales.
     - F5: background-fraction sweep.
  3. **9c — Optional intervention (only if approved at M2):** implement one targeted change in `sidx.core`, for example:
     - rarity-reweighted patch prior;
     - gradient channels added to the patch vector;
     - ink-preserving pyramid downsampling.
     Re-run it on the affected subset.
- **Technical requirements:** Colab T4, ~10 T4-h; the failure-isolation sweeps (F1–F5) run on 128-px synthetic images to stay cheap; sweep runner = `scripts/run_queue.py` with a sweep config.
- **Deliverables:** ablation tables/plots; mechanism figures (selection-frequency maps, per-scale strips); a hypothesis-status table (H1–H5: supported / refuted / inconclusive).
- **Dependencies:** Phase 7 (and the M2 decision).
- **Time:** weeks 5–7 (9c: weeks 7–8).
- **Difficulties:** scope creep; confounded factors in real images (that's why the synthetic set exists).
- **Done when:** every hypothesis has a status backed by a specific table or figure.

> **Milestone M3 — experiment freeze (end of week 8).** After this point, run no new experiments. Only re-runs to fix bugs are allowed.

---

### Stage IV — Analysis & Reporting

#### Phase 10 — Quantitative & qualitative evaluation
- **Objective:** consolidate all the results into defensible tables, figures and comparisons.
- **Tasks:**
  1. **Aggregate** all runs (`eval/aggregate.py`):
     - mean ± std per category and method;
     - paired Wilcoxon tests across images;
     - bootstrap 95 % confidence intervals.
  2. **Track-3 comparison table** (ours vs SinDDM vs LoRA):
     - metric groups G1–G4;
     - training time, inference time per sample, time-to-first-sample, peak VRAM, disk;
     - the amortization crossover n* = T_train / (t_ours − t_baseline).
  3. **Blind human rating study:**
     - 3 raters, ~60 image sets;
     - 4 criteria: structural correctness, detail preservation, plausibility, novelty;
     - report Krippendorff's α.
  4. **Metric-agreement analysis:** where do IQA/SIFID disagree with the structural metrics and with the human ratings? This becomes a reported finding.
  5. **Final qualitative figures:** fixed seeds, best and worst by composite score, zoomed crops at junctions and line ends.
- **Deliverables:** final tables (`reports/tables/`), figures (`reports/figures/`), human-study results.
- **Dependencies:** Phases 7, 8, 9.
- **Time:** weeks 7–8 (overlaps the end of Phase 9).
- **Difficulties:** rater availability; multiple-comparison issues (report effect sizes, not just p-values).
- **Done when:** every claim planned for the report is backed by a table or figure with uncertainty.

#### Phase 11 — Failure analysis
- **Objective:** turn the observations into an explained failure taxonomy — what fails, when, and why.
- **Tasks:**
  1. Finalize the taxonomy, with ≥ 2 annotated examples and a frequency per category for each failure type: fragmentation, ghosting, rare-symbol dropout, lattice dislocation, glyph soup, layout collapse, copy-paste memorization.
  2. Link each failure type to its mechanism evidence from Phase 9 (e.g. "rare-symbol dropout ↔ F2 rarity–selection correlation").
  3. Compare the failure profiles of ours, SinDDM and LoRA (shared failures ⇒ limits of patch-based modeling; method-specific ones ⇒ closed-form vs learned).
  4. Write the "when to use / when not to use" guidance.
- **Deliverables:** a failure-analysis chapter (text + figure panel + taxonomy table).
- **Dependencies:** Phases 9, 10.
- **Time:** weeks 8–9.
- **Difficulties:** separating causation from correlation in real images (lean on the synthetic results).
- **Done when:** each failure type has examples, frequency and a supported (or explicitly unexplained) cause.

#### Phase 12 — Documentation & research presentation
- **Objective:** a reproducible repository, a final report and a presentation.
- **Tasks:**
  1. README with the exact commands to reproduce each table (one command for the Table-1 replica).
  2. The report: introduction; method recap (the paper's contribution, clearly separated); reproduction; OOD study; ablations; baseline comparison; failure analysis; limitations of *our* study; future work.
  3. Slides (≈15–20) with the key figures.
  4. Clean-clone reproduction test; tag a release.
- **Deliverables:** `reports/report.(md|tex|pdf)`, `reports/slides/`, a tagged repo release.
- **Dependencies:** all previous phases.
- **Time:** weeks 9–10.
- **Difficulties:** last-minute re-runs — avoided by the M3 freeze.
- **Done when:** a fresh clone reproduces the Table-1 replica with one command, and the professor-facing report and slides are complete.

> **Milestone M4 — final review (week 10)** with the professor before submission.

---

## 13. Timeline and Milestones

```
Week:                         1    2    3    4    5    6    7    8    9    10
Stage I
 1 Paper & code analysis     ███
 2 Environment               ███
 3 Natural data              ██
Stage II
 4 Reproduce method               ███
 5 Validate vs paper              ████████
                                         ▲ M1
Stage III
 6 OOD data + metrics             ████████████
 7 OOD experiments                          ████████
                                                   ▲ M2
 8 Baselines (GPU queue)                ████████████████████
 9 Ablations + isolation                          ████████████ (+9c opt. wk 7–8)
                                                                  ▲ M3 freeze
Stage IV
10 Evaluation                                          ████████
11 Failure analysis                                         ████████
12 Documentation                                                 ████████
                                                                       ▲ M4
```

| Milestone | When | Decision |
|---|---|---|
| **M1** Reproduction go/no-go | end wk 3 | Proceed / switch to the official code as the generator / rely on relative metrics |
| **M2** OOD review | end wk 5 | Which 2–3 failure modes to study; run the intervention (9c) or not; rebalance categories |
| **M3** Experiment freeze | end wk 8 | No new experiments after this point |
| **M4** Final review | wk 10 | Professor sign-off before submission |

**Parallel lanes for a 3-person team [R]:**

| Lane | Owner focus | Phases |
|---|---|---|
| A | Method & reproduction | 1 (lead), 4, 5, 9 |
| B | Data & evaluation | 3, 6, 7, 10 |
| C | Baselines & compute | 2 (lead), 8, then supports 10–11 |
| All | Analysis & writing | 11, 12 |

**Compressed 6-week variant (if the deadline is tight):**

| Week | Phases |
|---|---|
| 1 | 1 + 2 + 3 (official code only; skip the re-implementation 4b) |
| 2 | 5 (R1, R3, R4 only) ∥ 6 (12 real + 16 synthetic, 4 core structural metrics) ∥ 8 starts (SinDDM on 5 images, no LoRA — or LoRA only if your professor requires fine-tuning) |
| 3 | 7 |
| 4 | 9 (AB1, AB2, AB4 + F1, F2 only; no intervention) |
| 5 | 10 + 11 (no human study; qualitative panel only) |
| 6 | 12 |

---

## 14. Risk Assessment

| Risk | Prob. | Impact | Mitigation | Alternative |
|---|---|---|---|---|
| Official code unavailable | **Low** (public, MIT, verified) | High | Pin the commit / fork immediately | Our own re-implementation is self-sufficient for pixel mode |
| FLUX VAE gated / access denied | Medium | Low (latent is an ablation only) | Request access in week 1 | Use the SD VAE (`stabilityai/sd-vae-ft-mse`) — document as a deviation |
| No NVIDIA GPU locally; free Colab only | **Certain** | High | Resumable ≤ 100-min jobs; Drive persistence; evaluation on CPU; spread over days; each teammate uses their own account | Cut sample counts (50→30), then ablation configs |
| Free Colab GPU temporarily unavailable / quota exhausted | High | Medium | Plan ~5 T4-h/week with slack; queue work early in the week | Use a teammate's account; Kaggle's free GPU notebooks (still zero cost) as a fallback |
| T4 slower than estimated | Medium | Medium | Phase-2 timing table recalibrates §5.3 | Lower resolutions (OOD 192 px; synthetic 96 px) |
| SinDDM can't be ported to Colab's torch | Low-Med | Medium | Port in week 1–2 (Phase 2) | Use only the 3 pretrained models + LoRA as the training baseline |
| Official code breaks on Colab's newer torch/Python | Low | High | Smoke test in Phase 2 | Pin torch in Colab (`pip install torch==2.8.0`), or use our CPU-validated core on the GPU |
| SinDDM env (torch 1.13) fails on a modern GPU | Medium | Medium | Test in week 1 | Port to torch 2.x, or switch to SinFusion |
| SinDDM compute (7–8 h/img on A6000) | **Certain** on the free tier | High | 3 pretrained + 3 OOD images at 128 px, ~4 T4-h each with resume; report the quality-vs-training-time curve as a finding | 2 OOD images; or pretrained models only |
| Metric implementation mismatch vs paper | Medium | Medium | Calibrate with SinDDM pretrained outputs; report relative trends | State the absolute deviation transparently |
| Paper details incomplete (blur, ρ units, value range) | Medium | Low | Code audit in Phase 1 | Use the official code as the oracle |
| OOD metrics don't capture structural failure | Medium | **High** | Sanity-test the metrics on controlled perturbations (Phase 6e) + human study | Lean on synthetic ground-truth checks |
| Poor OOD performance overall | Medium-High | **Low for research value** | That is a finding; the failure analysis is the contribution | — |
| "Realistic but structurally wrong" outputs fool IQA/SIFID | High | Medium | That is exactly why G2 exists; report the disagreement between metrics as a result | — |
| LoRA comparison criticized as unfair | Medium | Medium | Frame it as a different-prior contrast; SinDDM is the controlled baseline | Drop LoRA |
| Licence issues for published figures | Low-Med | Low | Track licences in manifests; prefer PD/CC images in the report | Use the synthetic set for figures |
| Scope creep | High | High | Milestones M1–M3 with explicit cut decisions | Compressed plan |

**Research opportunities beyond reproduction.**
1. First systematic OOD study of closed-form patch diffusion on structured imagery.
2. Structural metrics + patch precision/recall for single-image generation.
3. Mechanistic failure explanation via patch-selection analysis.
4. Quantified training-free vs training-based cost/quality trade-off including the amortization crossover.
5. An optional targeted fix.

**A well-supported negative result (e.g. "fails on long-range connectivity because of X, shown by Y") is a legitimate outcome.**

---

## 15. Expected Deliverables

| Deliverable | Origin |
|---|---|
| Working pipeline (official + our validated core re-implementation) | Reproduction |
| Table-1 replica, Fig. 8 / Table 3 subsets, qualitative app grid | Reproduction |
| OOD dataset (manifests, licences, preprocessing scripts, synthetic generator, data card) | **Ours** |
| Structural evaluation suite with sanity tests | **Ours** |
| OOD results per category (quantitative + qualitative) | **Ours** |
| SinDDM (+ LoRA) comparison with cost table and crossover analysis | **Ours** |
| Ablation results AB1–AB7, failure-isolation F1–F5 | **Ours** (AB1–AB3 partially mirror the paper's Table 3 / Fig. 8 on new data) |
| Failure taxonomy with annotated examples | **Ours** |
| Optional intervention result | **Ours** |
| Report + slides + reproducible repo | Both |

**Clear separation for the report:** the closed-form denoiser, coarse-to-fine sampler, accelerations, applications and natural-image results are **the paper's**. OOD evaluation, structural metrics, failure mechanisms, the training-based comparison on OOD data and any modification are **ours**.

---

## 16. Potential Research Contributions — Questions and Hypotheses

| RQ | Question | Hypothesis **[H]** | Evidence |
|---|---|---|---|
| RQ1 | Does local structural validity survive? | Yes inside patches; continuity breaks at stitching for structures longer than the patch at fine scales | AB1, F1 (length factor), endpoint excess |
| RQ2 | Are rare details dropped? | Rare patches (symbols, junctions) are under-selected ⇒ low patch recall | F2 rarity–frequency correlation, F5 |
| RQ3 | Are periodic patterns preserved? | Mostly yes (self-similarity), with dislocations at phase boundaries; η/T change the dislocation rate | FFT metrics, AB3 |
| RQ4 | Does coarse-scale aliasing destroy layout of thin structures? | 1–2 px lines vanish at the coarsest level; layout becomes random | AB4, F3, thickness sweep |
| RQ5 | Does the latent VAE hurt technical imagery? | Yes at ≤512 px for thin lines/text | AB5 |
| RQ6 | Training-free vs SinDDM on OOD | Comparable or better diversity and cost; similar structural failures (both are patch-limited models) | Track 3 |
| RQ7 | Pretrained-prior LoRA vs patch prior | LoRA: globally clean but low patch precision / hallucination; patch method: faithful patches, broken long-range structure | Track 3 |
| RQ8 | When does training pay off? | Only after ~10⁴ samples per image, if ever | Crossover analysis |

---

## 17. Immediate Next Steps (prioritized)

1. **Answer the open questions below** (deadline, team, GPU, baseline preference) — they change the scope.
2. **Fork/pin** `haojunqiu/efficient-SID-code` and `fallenshock/SinDDM` as submodules under `external/`.
3. **Build `notebooks/00_setup_smoke.ipynb` on free Colab** and run the official unconditional Quickstart in fp32 on one example image; record the T4 time and VRAM at 128/186×248/256 px.
4. **Request FLUX.1-schnell access on Hugging Face** (needed only for latent ablations, but approval can lag).
5. **Download FloorPlanCAD now** (unmaintained site) and the DTD; record licences.
6. **Code audit (Phase 1):** read `efficient_sid/config.py` and the denoiser/fold/pyramid/scheduler modules; resolve the [?] items (ρ units, blur kernel, value range, η update, scale count).
7. **Run `paper_sinddm15.yaml` on all 15 images** (50 samples) and compute SIFID + LPIPS-div to get the first reproduction numbers.
8. **Set up the `sinddm` env and sample from its provided pretrained models**; run one training job on a 256 px floor plan to measure *our* real training time (calibrates the whole compute plan).
9. **Write `sidx.core` (pixel exact) + the equivalence test** against the official code.
10. **Write the synthetic generator + 3 structural metrics** (LSD stats, skeleton endpoint excess, bimodality) and sanity-test them on perturbed synthetic images.

### Open questions to resolve before implementation

1. **Deadline and team size?** (10-week vs 6-week plan.)
2. ~~GPU access~~ — **resolved 2026-11-13:** free Google Colab only (T4), zero budget; the laptop is CPU-only.
3. **Baseline expectation from your professor:** is SinDDM (training-based, same task) acceptable as the primary "fine-tuning" baseline, or is DreamBooth/LoRA specifically required?
4. **Is re-implementation required for grading**, or is using the official code + extensions acceptable?
5. **Report format:** paper-style report, presentation, or both; any page limits?
6. **Domain preference:** are floor plans + geometric patterns + circuit schematics acceptable, or does the course want a specific domain?

---

## Sources

- Paper: Qiu, Kutulakos, Lindell. *Efficient and Training-Free Single-Image Diffusion Models*, arXiv:2606.04299 — local PDF.
- Project page: https://haojunqiu.github.io/efficient-SID/
- Official code: https://github.com/haojunqiu/efficient-SID-code (README, `environment.yml`, `configs/uncond/paper_sinddm15.yaml`)
- SinDDM: Kulikov et al., ICML 2023, arXiv:2211.16582; code https://github.com/fallenshock/SinDDM; project https://matankleiner.github.io/sinddm
- FloorPlanCAD: https://floorplancad.github.io ; Fan et al., ICCV 2021 (arXiv:2105.07147); HF mirror https://docs.voxel51.com/dataset_zoo/datasets_hf/floorplancad.html
- DTD: Cimpoi et al., CVPR 2014, https://www.robots.ox.ac.uk/~vgg/data/dtd/ ; HF mirror https://huggingface.co/datasets/Voxel51/Describable-Textures-Dataset
- FLUX.1-schnell (VAE): https://huggingface.co/black-forest-labs/FLUX.1-schnell
- SinGAN (SIFID): Shaham et al., ICCV 2019, https://github.com/tamarott/SinGAN
- DreamBooth: Ruiz et al., CVPR 2023 (arXiv:2208.12242); LoRA: Hu et al., ICLR 2022 (arXiv:2106.09685); diffusers training docs https://huggingface.co/docs/diffusers/training/dreambooth
- IQA-PyTorch (pyiqa): https://github.com/chaofengc/IQA-PyTorch ; LPIPS: https://github.com/richzhang/PerceptualSimilarity
- AI2D (verify licence before use): Kembhavi et al., ECCV 2016, Allen Institute for AI
