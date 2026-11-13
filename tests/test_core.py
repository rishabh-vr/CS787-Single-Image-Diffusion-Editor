"""Unit tests of our core (sidx.core) — CPU, tiny images. Plan Phase 4, task 3."""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from sidx.core import CoreConfig, CoreSampler
from sidx.core.denoiser import PatchDenoiser
from sidx.core.hooks import Hooks, ScaleRecorder, WinnerRecorder
from sidx.core.patches import extract_patches, fold_weighted, gaussian_window
from sidx.core.pyramid import build_pyramid, pyramid_shapes, two_scale_blend
from sidx.core.resize import resize, resize_matrix
from sidx.core.schedule import FlowLinearSchedule


@pytest.fixture()
def image() -> torch.Tensor:
    """A 40x52 RGB test image with structure (stripes + a square) and some noise."""
    g = torch.Generator().manual_seed(0)
    y, x = torch.meshgrid(torch.arange(40), torch.arange(52), indexing="ij")
    img = torch.stack([(torch.sin(x / 3.0) + 1) / 2, (y % 8 < 4).float(), ((x > 20) & (y > 10)).float()])
    return (0.8 * img + 0.2 * torch.rand(3, 40, 52, generator=g)).clamp(0, 1)


# --- denoiser -------------------------------------------------------------------------------

def _dataset(n=300, d=27, seed=0):
    return torch.rand(n, d, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def test_alpha_zero_gives_mean_patch():
    """At t = T, alpha = 0: every query is equidistant from all patches -> the mean patch."""
    y = _dataset()
    q = torch.randn(5, y.shape[1])
    out, _ = PatchDenoiser(y)(q, torch.tensor(0.0), torch.tensor(1.0))
    assert torch.allclose(out, y.mean(0).expand_as(out), atol=1e-6)


def test_small_sigma_gives_nearest_patch():
    """As sigma -> 0 the softmax becomes argmax: the output is the nearest clean patch."""
    y = _dataset()
    q = y[[3, 17, 250]] + 0.01 * torch.randn(3, y.shape[1])
    out, idx = PatchDenoiser(y)(q, torch.tensor(1.0), torch.tensor(1e-4), argmax=True)
    assert idx.tolist() == [3, 17, 250]
    assert torch.allclose(out, y[[3, 17, 250]], atol=1e-6)


def test_matches_explicit_formula():
    """Against Eq. 2 written out literally (with the ||x||^2 term, in float64)."""
    y, q = _dataset(), torch.randn(7, 27)
    alpha, sigma = torch.tensor(0.6), torch.tensor(0.4)
    d2 = ((q.double()[:, None, :] - alpha.double() * y.double()[None]) ** 2).sum(-1)
    ref = torch.softmax(-d2 / (2 * sigma.double() ** 2), 1) @ y.double()
    out, _ = PatchDenoiser(y)(q, alpha, sigma)
    assert torch.allclose(out.double(), ref, atol=1e-5)


@pytest.mark.parametrize("alpha,sigma", [(0.0, 1.0), (0.5, 0.5), (0.9, 0.1), (1 - 1e-4, 1e-4)])
def test_backends_agree(alpha, sigma):
    """dense, chunked and SDPA agree in fp32. Plan target 1e-5; we allow 1e-4 because at small
    sigma a near-tie between two patches amplifies ~1e-6 rounding in the logits."""
    y = _dataset(500, 75)
    q = torch.randn(64, 75, generator=torch.Generator().manual_seed(1)) * 0.5
    a, s = torch.tensor(alpha), torch.tensor(sigma)
    dense, i0 = PatchDenoiser(y, "dense")(q, a, s, argmax=True)
    chunked, i1 = PatchDenoiser(y, "chunked", chunks=5)(q, a, s, argmax=True)
    sdpa, i2 = PatchDenoiser(y, "sdpa", chunks=3)(q, a, s, argmax=True)
    assert torch.allclose(dense, chunked, atol=1e-4)
    assert torch.allclose(dense, sdpa, atol=1e-4)
    assert torch.equal(i0, i1) and torch.equal(i0, i2)


# --- patches ----------------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.1, 0.2, 1.0, math.inf])
def test_fold_of_unfold_is_identity(image, rho):
    """fold(unfold(x)) = x. At rho = 0.1 the corner pixels are covered only by patch corners,
    whose weight exp(-100) underflows in fp32 (the official code is identical there), so for
    small rho only pixels at least p//2 from the border are checked."""
    p = 7
    patches = extract_patches(image, p)
    w = gaussian_window(p, rho, 3)
    back = fold_weighted(patches, w, image.shape[-2:], p)[0]
    m = p // 2 if rho < 0.2 else 0
    H, W = image.shape[-2:]
    assert torch.allclose(back[:, m:H - m, m:W - m], image[:, m:H - m, m:W - m], atol=1e-5)


def test_patch_layout_is_channel_major(image):
    patches = extract_patches(image, 5)
    assert patches.shape == ((40 - 4) * (52 - 4), 3 * 25)
    # patch at top-left position (row 2, col 3) is image[:, 2:7, 3:8] flattened C, p, p
    k = 2 * (52 - 4) + 3
    assert torch.equal(patches[k], image[:, 2:7, 3:8].reshape(-1))


def test_gaussian_window_units():
    """rho is the std over normalised [-1, 1] coordinates (notes Q1)."""
    w = gaussian_window(15, 0.2, 1).reshape(15, 15)
    centre_row = w[7]
    # coordinate of pixel 8 is 2/14; weight = exp(-0.5 (2/14 / 0.2)^2)
    assert math.isclose(float(centre_row[8]), math.exp(-0.5 * ((2 / 14) / 0.2) ** 2), rel_tol=1e-5)
    assert torch.all(gaussian_window(7, math.inf, 2) == 1)


# --- resize / pyramid ---------------------------------------------------------------------------

def test_resize_matrix_rows_sum_to_one():
    for n_in, n_out in [(186, 93), (93, 47), (24, 48), (31, 62), (10, 10)]:
        m = resize_matrix(n_in, n_out)
        assert m.shape == (n_out, n_in)
        assert torch.allclose(m.sum(1), torch.ones(n_out), atol=1e-6)


def test_resize_preserves_constants(image):
    c = torch.full((3, 40, 52), 0.3)
    assert torch.allclose(resize(c, (20, 26)), torch.full((3, 20, 26), 0.3), atol=1e-6)
    assert torch.allclose(resize(c, (80, 77)), torch.full((3, 80, 77), 0.3), atol=1e-6)


def test_pyramid_shapes_paper_image():
    """notes §2 step 4: 186x248 -> 93x124 -> 47x62 -> 24x31."""
    assert pyramid_shapes((186, 248), 4) == [(186, 248), (93, 124), (47, 62), (24, 31)]


def test_two_scale_blend_keeps_detail_and_takes_coarse_lowpass(image):
    fine = image.unsqueeze(0)
    coarse = resize(image, (20, 26)).unsqueeze(0)
    # blending a fine image with its own coarser level changes nothing
    assert torch.allclose(two_scale_blend(fine, coarse), fine, atol=1e-5)
    # a constant offset in the coarse scale moves the result by that offset
    shifted = two_scale_blend(fine, coarse + 0.1)
    assert torch.allclose(shifted - fine, torch.full_like(fine, 0.1), atol=1e-5)


# --- schedule ---------------------------------------------------------------------------------

def test_schedule_grid_matches_code_not_paper_text():
    """notes D1: evaluations at T points from sigma = 1 to sigma_min; index 0 is the terminal."""
    s = FlowLinearSchedule(10, 1e-4)
    assert len(s.sigmas) == 11 and float(s.sigmas[0]) == 0.0 and float(s.alphas[0]) == 1.0
    assert math.isclose(float(s.sigmas[1]), 1e-4, rel_tol=1e-3)
    assert math.isclose(float(s.sigmas[-1]), 1.0)
    assert torch.allclose(s.alphas + s.sigmas, torch.ones(11))


def test_step_onto_terminal_returns_x_hat():
    s = FlowLinearSchedule(5)
    x_hat, eps = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
    assert torch.equal(s.step(x_hat, eps, 1, eta=1.0), x_hat)


# --- sampler ----------------------------------------------------------------------------------

def _cfg(**kw):
    base = dict(patch_size=5, num_scales=3, num_steps=5)
    base.update(kw)
    return CoreConfig(**base)


def test_sampler_deterministic_and_seed_dependent(image):
    s = CoreSampler(image, _cfg(), "cpu")
    a, b, c = s.sample(1), s.sample(1), s.sample(2)
    assert a.shape == image.shape
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sampler_backends_agree(image):
    a = CoreSampler(image, _cfg(backend="dense"), "cpu").sample(3)
    b = CoreSampler(image, _cfg(backend="sdpa", query_chunks=2), "cpu").sample(3)
    assert (a - b).abs().mean() * 255 < 0.05


def test_output_patches_come_from_the_input(image):
    """Provenance (paper S5): at sigma_min = 1e-4 every finest-scale patch the output copies is
    an input patch, so with rho -> 0 (centre-pixel copying) and no blend at the last step,
    the output pixels are input pixels (checked away from the border, where no patch centre lies)."""
    s = CoreSampler(image, _cfg(fold_rho=0.01, num_scales=1), "cpu")
    out = s.sample(4)[:, 2:-2, 2:-2]
    in_pixels = set(map(tuple, (image * 255).round().reshape(3, -1).T.tolist()))
    out_pixels = (out * 255).round().reshape(3, -1).T.tolist()
    frac = np.mean([tuple(p) in in_pixels for p in out_pixels])
    assert frac > 0.99


def test_hooks_record_scales_and_winners(image):
    s = CoreSampler(image, _cfg(), "cpu")
    scales, winners = ScaleRecorder(), WinnerRecorder()
    s.sample(5, hook=Hooks(scales, winners), record_argmax=True)
    assert sorted(scales.scales) == [0, 1, 2]
    assert scales.scales[0].shape == image.shape
    w = winners.winners[(0, 1)]
    assert w.shape[0] == (40 - 4) * (52 - 4) and int(w.max()) < s.denoisers[0].N


def test_sid_core_method_interface(tmp_path, image):
    from sidx.methods.base import make_method

    p = tmp_path / "img.png"
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(p)
    m = make_method("sid_core", device="cpu", patch_size=5, num_scales=2, num_steps=3)
    info = m.setup(str(p))
    assert info["pyramid_hw"] == [[40, 52], [20, 26]]
    out = m.sample(10)
    assert out.shape == (3, 40, 52) and 0 <= float(out.min()) and float(out.max()) <= 1.0001
    m.teardown()
    with pytest.raises(ValueError):
        make_method("sid_core", not_a_param=1)
