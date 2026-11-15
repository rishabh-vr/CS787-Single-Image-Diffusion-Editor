"""Known-answer tests for the metrics (CPU, small images). Networks (Inception, AlexNet, IQA) are
not needed here except where a test is marked and skipped when weights are unavailable."""

import numpy as np
import pytest
import torch

from sidx.core.resize import resize
from sidx.data.synthetic import BASE, SynthSpec, design, render
from sidx.metrics import copying, paper, patchdist, structure


def _img(spec=None):
    im = render(spec or BASE)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)


@pytest.fixture(scope="module")
def grid():
    return _img()


def _natural(seed=0, n=96):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, n // 4, n // 4, generator=g)
    return resize(base, (n, n)).clamp(0, 1)


# --- paper metrics ----------------------------------------------------------------------------

def test_pixel_diversity_zero_for_identical_samples():
    x = _natural()
    assert paper.pixel_diversity(x, [x, x, x])["pixel_div_raw"] == 0.0


def test_pixel_diversity_grows_with_spread():
    x = _natural()
    lo = paper.pixel_diversity(x, [x + 0.01 * torch.randn_like(x) for _ in range(6)])["pixel_div_raw"]
    hi = paper.pixel_diversity(x, [x + 0.10 * torch.randn_like(x) for _ in range(6)])["pixel_div_raw"]
    assert 0 < lo < hi


def test_near_duplicate_fraction_and_rate():
    x = _natural()
    assert paper.near_duplicate_fraction(x, x) == 1.0
    assert paper.near_duplicate_fraction(x, (x + 0.5) % 1.0) < 0.05
    r = paper.near_duplicate_rate([1.0, 0.9, 0.2, 0.0])
    assert r["near_dup_r50"] == 0.5 and r["near_dup_r85"] == 0.5 and r["near_dup_r95"] == 0.25


def test_frechet_zero_for_same_features_and_positive_otherwise():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 8))
    assert abs(paper._frechet(a, a)) < 1e-6
    assert paper._frechet(a, a + 1.0) == pytest.approx(8.0, rel=0.05)      # shifted mean: |d|^2 = 8


# --- patch distribution -----------------------------------------------------------------------

def test_patch_swd_zero_for_identical_and_orders_corruption():
    x = _natural()
    assert patchdist.patch_swd(x, x) < 1e-6
    noisy = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)
    assert patchdist.patch_swd(x, noisy) > patchdist.patch_swd(x, (x + 0.02 * torch.randn_like(x)).clamp(0, 1))


def test_patch_precision_recall_identity_and_noise():
    x = _natural()
    pr = patchdist.PatchPR(x)
    assert pr.add(x) == 1.0
    assert pr.recall() == 1.0
    pr2 = patchdist.PatchPR(x)
    assert pr2.add(torch.rand_like(x)) < 0.05                               # invented content


def test_patch_recall_drops_when_a_region_is_missing():
    x = _natural()
    half = x.clone()
    half[:, :, 48:] = x[:, :, :48].flip(-1)                                  # right half replaced by a mirror
    pr = patchdist.PatchPR(x)
    pr.add(half)
    assert 0.3 < pr.recall() < 0.8


# --- copying ----------------------------------------------------------------------------------

def test_copy_regions_identity_vs_shuffled_blocks():
    x = _natural(n=128)
    assert copying.copy_regions(x, x)["block_copy_ratio"] == 1.0
    # 16x16 blocks permuted: copied pieces are far smaller than 48 px
    b = 16
    blocks = x.unfold(1, b, b).unfold(2, b, b).reshape(3, -1, b, b)
    perm = torch.randperm(blocks.shape[1], generator=torch.Generator().manual_seed(0))
    sh = blocks[:, perm].reshape(3, 8, 8, b, b).permute(0, 1, 3, 2, 4).reshape(3, 128, 128)
    assert copying.copy_regions(x, sh)["block_copy_ratio"] < 0.2


# --- structure --------------------------------------------------------------------------------

def test_structure_identity_is_neutral(grid):
    ref = structure.StructureReference(grid)
    m = ref.compare(grid, allowed_angles=[0, 90])
    assert m["endpoint_excess"] == pytest.approx(1.0, abs=0.01)
    assert m["junction_hist_l1"] == pytest.approx(0.0, abs=1e-9)
    assert m["grey_ratio"] == pytest.approx(1.0, abs=0.01)
    assert m["mean_seg_len_ratio"] == pytest.approx(1.0, abs=0.01)
    assert m["orientation_conformity"] > 0.95


def test_structure_detects_fragmentation(grid):
    """A grid whose lines are chopped into dashes: more endpoints, shorter segments."""
    ref = structure.StructureReference(grid)
    broken = _img(SynthSpec(length="short"))
    m = ref.compare(broken, allowed_angles=[0, 90])
    assert m["endpoint_excess"] > 3
    assert m["mean_seg_len_ratio"] < 0.7


def test_structure_detects_ghosting(grid):
    ref = structure.StructureReference(grid)
    blurred = torch.nn.functional.avg_pool2d(grid[None], 5, 1, 2)[0]
    assert ref.compare(blurred)["grey_ratio"] > 1.5


def test_structure_detects_tilted_lines(grid):
    ref = structure.StructureReference(grid)
    from PIL import Image

    rot = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).rotate(
        10, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    r = torch.from_numpy(np.asarray(rot, dtype=np.float32) / 255).permute(2, 0, 1)
    assert ref.compare(r, allowed_angles=[0, 90])["orientation_conformity"] < 0.3


def test_spectral_peak_recall_lattice_vs_random(grid):
    g = structure.luminance(grid)
    assert structure.spectral_peak_recall(g, g) == 1.0
    rng = np.random.default_rng(0)
    assert structure.spectral_peak_recall(g, rng.random(g.shape)) < 0.5


# --- synthetic design -----------------------------------------------------------------------------

def test_synthetic_design_is_one_factor_at_a_time():
    d = design()
    ids = [i for i, _, _ in d]
    assert len(ids) == len(set(ids))
    base = {s.seed: s for i, f, s in d if f == "base"}
    for image_id, factor, spec in d:
        if factor == "base":
            continue
        diff = [k for k in spec.__dataclass_fields__ if getattr(spec, k) != getattr(base[spec.seed], k)]
        assert diff == [factor], (image_id, diff)


def test_synthetic_render_is_deterministic_and_binary_ish():
    a, b = render(BASE), render(BASE)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    g = np.asarray(a.convert("L")) / 255
    assert g.min() == 0 and g.max() == 1 and ((g > 0.2) & (g < 0.8)).mean() < 0.25   # anti-aliased edges of 2-px lines
