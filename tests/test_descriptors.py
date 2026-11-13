"""Descriptor sanity checks on images whose answer is known."""

import numpy as np
import torch

from sidx.data.descriptors import compute_descriptors, periodicity_score


def _grid(period=12, width=2, n=96):
    y, x = np.mgrid[0:n, 0:n]
    return 1 - ((x % period < width) | (y % period < width)).astype(np.float32)


def test_periodicity_separates_lattice_line_and_noise():
    assert periodicity_score(_grid()) > 0.9
    line = np.ones((96, 96), np.float32)
    line[40:43, :] = 0
    assert periodicity_score(line) < 0.2                      # repeats only along itself
    assert periodicity_score(np.random.default_rng(0).random((96, 96))) < 0.2


def test_line_width_and_ink():
    img = np.ones((96, 96), np.float32)
    img[30:34, 10:90] = 0                                     # a 4-px stroke
    d = compute_descriptors(torch.from_numpy(img)[None].repeat(3, 1, 1))
    assert abs(d["line_width_px"] - 4.0) <= 1.0
    assert abs(d["ink_fraction"] - 4 * 80 / 96 ** 2) < 1e-3


def test_constant_image_is_all_zero():
    d = compute_descriptors(torch.full((3, 64, 64), 0.5))
    assert all(v == 0 for v in d.values()), d


def test_entropy_orders_structure():
    noise = compute_descriptors(torch.rand(3, 96, 96, generator=torch.Generator().manual_seed(0)))
    grid = compute_descriptors(torch.from_numpy(_grid())[None].repeat(3, 1, 1))
    assert noise["patch_code_entropy"] > grid["patch_code_entropy"]
