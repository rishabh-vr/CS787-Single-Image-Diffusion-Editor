# Final results (Kaggle runs K10–K14 + Colab Phase 4)

Written for the CS787 report and slides. Tables: `reports/tables/` (from `scripts/analyze.py`); figures:
`reports/figures/final/` (from `scripts/report_figures.py`). Raw outputs and evaluations live in the Kaggle zips
(`kaggle_results/`, not in git). All numbers are means over images. Each image's value is itself the mean over
its samples; the n per row is given.

## What was run (and what was cut)

| Part | Run | Samples | Where |
|---|---|---|---|
| Reproduction (Track 1) | official code, paper preset (T = 10, η = 0, p = 15, ρ = 0.2), fp32 | 7 of 15 natural images × 50 seeds (348) | Colab, Phase 4 |
| OOD (Track 2) | Set A: 24 real images (8 floor plans, 5 circuits, 3 diagrams, 8 patterns) × 6 seeds; Set B: 16 distinct synthetic line drawings × 6 seeds | 288 | Kaggle K10 |
| Ablations | patch size, ρ, η, number of scales, stride on 4 images × 4 seeds | 200 (+ 2 setups that cannot run) | K10 |
| SinDDM baseline | trained 3k steps (≈ 1 T4-hour) on 3 OOD images (128 px), sampled at 1.5k / 3k; official pretrained models on forest / night_sky / pyramids | 180 | K11 |
| LoRA baseline | DreamBooth-LoRA on SD 1.5, rank 8, 400 steps, 8 images | 120 | K12 |
| Ours on the baseline images | same preset, 10 seeds | 80 | K10 |
| Applications | retargeting (6 images × wide / tall), tileable textures (4 patterns) | 16 | K14 |
| Evaluation | 26 job folders, ≤ 20 samples per image, CPU | — | K13 |

**Deviations from the plan, which must be stated in the report:**
- Everything ran on free GPUs. To fit, each notebook was capped at under 2 hours, so seeds and images were truncated (the full plan was ≈ 30 GPU-h; the reduced runs used ≈ 9 GPU-h of computation: Colab 1.7, K10 3.1, K11 1.8, K12 0.7, K14 ≈ 1.7).
- Phase 4 reached 7 of the 15 natural images. The T = 40 / η = 1 and ANN reproduction jobs did not run.
- SinDDM was trained for 3k steps instead of the official 120k. This gives a *matched-compute* comparison, not a converged one. The pretrained models serve as the converged reference, but only on SinDDM's own images.
- The LoRA settings are the diffusers defaults. There was no tuning set.
- In Set B, 8 of the 24 ids are duplicates: the generators for grid, thickness, density and junctions are deterministic, so seeds s0 and s1 give identical images. The analysis counts each distinct image once.

## 1. Reproduction (Track 1)

| | SIFID | NIQE | NIMA | MUSIQ | pixel div. | LPIPS div. |
|---|---|---|---|---|---|---|
| **Our run** (7 images) | 3.45 ± 2.27 | 8.50 ± 3.58 | 4.57 ± 0.50 | 51.7 ± 12.4 | **0.15 ± 0.04** | **0.50 ± 0.07** |
| Paper, Table 1 (15 images) | 0.29 | 8.08 | 4.53 | 55.41 | 0.15 | 0.49 |
| Paper's SinDDM | 0.48 | 7.69 | 4.30 | 50.74 | 0.10 | 0.36 |

- **The diversity metrics reproduce exactly, and the IQA metrics are within ~7%.**
- **SIFID is ~12× the paper's value for every method.** This includes SinDDM's own pretrained models, which score 1.24–2.74 in our pipeline against the paper's 0.48.
  - We tested four SIFID definitions on the same samples: pytorch-fid resizing or none, and [0, 1] input or SinGAN's double-scaled [0, 1/255] input.
  - None of them gives the paper's scale.
  - Neither paper publishes its metric code (technical notes, Q7).
  - We therefore use SIFID **only to compare methods inside our pipeline**.
- **Near-duplicates (Table S1 protocol):** 20% of samples have ≥ 50% near-identical pixels. 0% reach ≥ 85%.
- **Forest, the one image with both methods:** pretrained SinDDM scores SIFID 1.27 against our 3.39, and copies less (copied-block ratio 0.44 against 0.79). The paper's ranking (ours < SinDDM on SIFID) is not confirmed on this image. One image is weak evidence.

Figure: `fig_natural.png`.

## 2. Out-of-distribution images (Track 2)

Set A, by category (from `track2_vs_natural`; natural = the Phase 4 images):

| category | n | patch recall | copied-block ratio | endpoint excess (1 = same as input) | LPIPS div. | SIFID |
|---|---|---|---|---|---|---|
| natural | 7 | 0.97 | 0.81 | 1.27 | 0.50 | 3.46 |
| floor plan | 8 | 0.73 | 0.58 | 1.22 | 0.53 | 3.80 |
| circuit | 5 | 0.60 | 0.38 | 1.34 | 0.51 | 2.26 |
| diagram | 3 | 0.64 | 0.40 | 1.35 | 0.63 | 3.07 |
| pattern | 8 | 0.87 | 0.35 | 1.45 | 0.44 | 2.58 |

Findings:
1. **The method does not reproduce the inputs' full range of local structures on drawings.** Patch recall (the share of the input's patches that reappear in samples) falls from 0.97 on natural images to 0.60–0.73 on circuits, diagrams and floor plans. Rare patches drop out: labels, symbols, a single porch.
2. **Lines break.** Samples have 1.2–1.45× as many line endpoints as the input. Patterns are the worst at 1.45 (seams where the lattice is re-joined).
3. **Local fidelity without global structure.** Patch precision stays ≈ 1, so every local patch is valid. However:
   - floor-plan rooms are re-joined without a consistent layout;
   - circuits keep their symbols but the wiring and labels are scrambled;
   - diagrams duplicate the hub node, with edges that lead nowhere.
4. **Text is the clearest failure.** Letters are copied as patches, so words are cut and repeated mid-word ("DRAWIING ROOM", "RESPOSPOKSPOND").

Set B (16 synthetic drawings):
- Each factor has only 1–2 images per level, so these are qualitative trends, not significant effects.
- Jittered periodicity raises SIFID strongly (3.1 against 0.74 for the perfect grid).
- Denser grids lower diversity: LPIPS div. is 0.05 at density 10 against 0.15 at density 2.
- Table: `track2_setB_factor_effects`.

Figures: `fig_ood_gallery.png`, `fig_ood_metrics.png`, `fig_failures.png`.

## 3. Baselines on the same images (Track 3)

| method | images | SIFID | copied-block ratio | patch precision | LPIPS div. | GPU time, 20 samples of a new image (T4) |
|---|---|---|---|---|---|---|
| **ours** | 3 OOD images, 128 px | **0.39** | 0.34 | 0.97 | 0.13 | **36 s** (1.8 s/sample, no training) |
| SinDDM, 3k steps | same 3 | 18.9 | 0.02 | 0.38 | 0.40 | ≈ 32 min training + 46 s |
| **ours** | 5 OOD images, ≤ 256 px | **2.48** | 0.34 | 0.97 | 0.53 | 11.4 min (34 s/sample) |
| DreamBooth-LoRA | same 5 | 15.6 | 0.03 | 0.39 | 0.40 | 3.8 min training + 107 s |
| SinDDM pretrained (≈ 120k steps) | its 3 natural images | 1.75 | 0.30 | 1.00 | 0.38 | (≈ 1 day of training, from the SinDDM paper) |

- **SinDDM at a matched budget of ≈ 1 GPU-hour is not usable on drawings.** It produces blurred grids, broken p4 tiles and an almost blank floor plan, which got *worse* from 1.5k to 3k steps. Our method needs no training.
- **However, our low SIFID at 128 px is partly memorisation.** On the floor plan and the grid, our samples are near-copies of the input (copied-block ratio 0.52, LPIPS div. 0.08–0.19). Small, simple drawings leave the patch denoiser little to recombine.
- **LoRA is the opposite of ours.**
  - It makes *globally* coherent drawings: whole houses with connected rooms, and a tidy chip schematic.
  - It copies almost nothing (copied-block ratio 0.03).
  - But it drifts from the input: palette and symmetry change on patterns (patch precision 0.001–0.003 on p4 / cmm), and text becomes pseudo-letters. SIFID is 5–10× worse.
  - It is faster than ours at 256 px (5.4 s against 34 s per sample, after a 4-minute training run). Ours is exact attention over all patches, which costs O(N²).
- **Overall:** the training-free method is the most faithful to the input's local statistics, and the cheapest when only a few samples are needed. Its weakness is global structure, which is exactly what drawings need.

Figures: `fig_baselines_sinddm.png`, `fig_baselines_lora.png`, `fig_baseline_metrics.png`, `fig_cost.png`.

## 4. Ablations (4 images: balloons, fp_layout, pat_p4, ckt_24aa; 4 seeds)

Mean change against the paper preset (p 15, ρ 0.2, 4 scales, stride 1, T 10, η 0). Preset values: SIFID 3.58, copied-block ratio 0.48, LPIPS div. 0.54.

| change | Δ SIFID | Δ copied-block ratio | Δ LPIPS div. | reading |
|---|---|---|---|---|
| patch 7 | +2.79 | −0.33 | +0.01 | less copying, worse fidelity |
| patch 11 | +0.62 | −0.20 | 0.00 | |
| patch 23 | **−2.08** | **+0.25** | −0.09 | better SIFID by copying (balloons ≈ the input) |
| ρ 0.1 / 1 / ∞ | −0.04 / +0.39 / +0.70 | −0.06 / +0.03 / +0.07 | ≈ 0 | small effect (the paper reports ρ = ∞ as much worse) |
| η 1 (T 10) | +1.93 | −0.08 | −0.04 | η = 1 does not help at T = 10 |
| 2 / 3 scales | +7.37 / +5.70 | −0.35 / −0.22 | ≈ 0 | coarse layout lost |
| 5 scales (2 images) | **−1.77** | **+0.16** | **−0.34** | better SIFID by collapsing diversity |
| stride 2 / 4 | +13.3 / +33.5 | −0.20 / −0.07 | −0.04 / −0.10 | sparse patches → blur and lost detail |

- **There is a trade-off between fidelity and copying.** Every change that lowers SIFID (patch 23, a 5th scale) does it by copying more or reducing diversity, so SIFID alone rewards memorisation.
- This is why we report copied-block ratio, patch recall and LPIPS diversity next to it.
- A 5th scale cannot run at all on `fp_layout` and `balloons`: the coarsest scale would be smaller than one 15-px patch. The runner logs this as a setup error.

Figure: `fig_ablations.png`; table: `phase9_ablation_deltas`.

## 5. Applications

- **Retargeting to 2× width or height works well on structured drawings:**
  - floor plans become longer houses with extra rooms;
  - patterns extend seamlessly;
  - the circuit chip is repeated, so the result looks plausible but is electrically meaningless.
- **Text stutters** when it is stretched.
- **Tileable textures:** naive tiling of the input shows seams. Tileable samples tile seamlessly for p4, cmm and demi1. The p2 sample tiles but has some broken cells.

Figure: `fig_applications.png`.

## Limitations of this evaluation

- Few images per cell: 3–8 per category, 4 for the ablations, and 3 for SinDDM. Confidence intervals are wide, and per-image results are shown wherever possible.
- SIFID's absolute scale does not match the paper (see §1). Only comparisons within our pipeline are valid.
- SinDDM ran at about 2.5% of its official training budget. It and LoRA are not tuned.
- Phase 4 covers 7 of the paper's 15 natural images, and the T = 40 and ANN rows are missing.
