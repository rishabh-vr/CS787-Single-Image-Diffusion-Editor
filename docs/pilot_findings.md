# Laptop pilot: what it shows and what it does not (2026-11-15)

**Status: preliminary.** Everything here comes from our own implementation (`sidx.core`) running on the laptop CPU at
**64 px, 3 pyramid scales**, with 6–8 samples per image and 2 images per factor level. The real experiments (128 px
Set B, 256 px Set A, 30 samples) run on the Colab T4 and are pending. Numbers: `docs/results.md`,
raw files: `reports/tables/`, `reports/failure/`, figures: `reports/figures/`.

## What was run
| Experiment | Setup | Cost |
|---|---|---|
| Set B64 factor sweep (F1) | 24 synthetic drawings × 8 samples, paper preset, depth 3 | 192 samples, ~1.5 h CPU |
| F2 patch selection, F3 per-scale, F4 blend ablation, F5 background | 3 images (`b0_s0`, `thickness_1_s0`, `text_sparse_s0`) × 6 samples (F5: 3 canvases) | ~14 min CPU |

## Findings, ranked by how much weight they can bear

1. **The two-scale blend is what holds the layout together (F4, strong).** Without it, the correlation between the finished
   coarse scale and the final image drops from 0.97–1.00 to −0.01–0.06 (layout is lost), patch SWD gets 4–13× worse, and the
   base grid gets 3.0× / the 1-px grid 3.0× / the text grid 1.5× more dangling line ends (endpoint excess). Same direction on all
   three images. This is the one result large enough to trust at n = 3 images.
2. **Thin and dense lines are harder (F1, suggestive).** Structural deviation: thickness 8 px 0.010, 4 px 0.019, 2 px 0.066, 1 px 0.103;
   patch precision 0.99 → 0.95 → 0.88 → 0.74. Density (lines every 10 px, shorter than the 15-px patch): deviation 0.141 vs 0.002
   for sparse lines. The sample grids show the failure: bent or shifted line segments, ghosted extra lines. Only 2 images per level, so
   no confidence statements.
3. **At 1-px thickness, breakage shows up at the finest scale, not the coarse layout (F3, weak).** Mean segment length ratio 0.93
   and endpoint excess 1.09 at 64 px, versus 0.98 and 0.97 at 32 px; the plain grid stays ≈ 1.0 at both. One image, 6 samples.
4. **Rare patches are not preferentially dropped (F2; the hypothesis is not supported here).** For the text image, the rarest 10 %
   of input patches are selected at 1.06× the uniform rate; only 14 % of them are never selected, against 35 % of the commonest half.
   For the grids, ~87 % of patches are never selected, but equally for rare and common ones (Spearman 0.07): periodic images
   contain many interchangeable patches. Caveat: at this size the samples are close to copies of the input (below).
5. **More white background did not hurt (F5; the hypothesis is not supported here).** Patch precision rose (0.88 → 0.96) and ink
   fraction stayed ≈ 1. Confounded: the drawing also shrinks relative to the 15-px patch.

## Limits of the pilot (read before quoting any number)
- **Low diversity.** In a 64-px image with 15-px patches the sampler often reproduces the input: LPIPS diversity is 0.03–0.09 (the paper reports 0.49 on
  natural images), and the text labels come back at identical positions. Factor effects are therefore mixed with "how much was
  just copied". The 128-px Colab run reduces this.
- **Uninformative metrics at this size:** block-copy ratio is 0 everywhere (a 48-px block is nearly the whole image); pooled patch recall is
  ≈ 0.03 throughout (dominated by many near-identical background patches).
- **Tiny n.** 2 images per level, 6–8 samples. No tests of significance were run.
- **Our implementation, not the official code.** It matches the official one to ≈ 0.003 grey levels on average (`reports/equivalence/`), but
  both were checked only on natural-image inputs.

## What would change these conclusions
The Colab runs (Set B at 128 px, Set A at 256 px, ablations AB1–AB4) test the same questions with more samples, larger images and
real line art. In particular: whether the thin-line effect survives at 128 px, whether F2/F5 stay null on real floor plans, and
whether the blend result holds with the official code.
