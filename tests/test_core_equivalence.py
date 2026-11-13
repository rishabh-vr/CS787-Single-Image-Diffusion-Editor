"""Our core vs. the OFFICIAL code, same seed, fp32, on CPU (small settings so it runs in seconds).

Skipped when the official checkout (external/efficient-SID-code) is not available.
The full-size check at the paper's preset is scripts/check_equivalence.py
(result: reports/equivalence/).
"""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SID = ROOT / "external" / "efficient-SID-code"
pytestmark = pytest.mark.skipif(not (SID / "efficient_sid").exists(), reason="official code not cloned")


@pytest.fixture(scope="module")
def image():
    from sidx.core.resize import resize
    from sidx.core.sampler import load_image

    p = SID / "examples" / "uncond" / "balloons.png"
    return resize(load_image(str(p)), (48, 64))


def test_resampler_matches_official():
    from sidx.core.official_cpu import import_official

    import_official()                          # puts the official repo on sys.path
    from efficient_sid.resize_right import resize as off_resize

    from sidx.core.resize import resize

    x = torch.rand(3, 61, 83, generator=torch.Generator().manual_seed(0))
    for hw in [(31, 42), (16, 21), (61, 83), (122, 166), (20, 120)]:
        a = resize(x, hw)
        b = off_resize(x, scale_factor=None, output_shape=(3, *hw))
        assert (a - b).abs().max() < 1e-5, hw


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_sampler_matches_official(image, eta):
    from sidx.core import CoreConfig, CoreSampler
    from sidx.core.official_cpu import OfficialSampler

    cfg = CoreConfig(patch_size=7, num_scales=3, num_steps=5, eta=eta)
    for seed in (10, 11):
        a = CoreSampler(image, cfg, "cpu").sample(seed)
        b = OfficialSampler(image, cfg).sample(seed)
        d = (a - b).abs().flatten() * 255
        # identical up to float rounding; at sigma = 1e-4 a near-tie can flip one hard patch pick,
        # so allow a few pixels to differ by a few grey levels
        assert d.mean() < 0.02, (eta, seed, float(d.mean()))
        assert (d > 1).float().mean() < 1e-3, (eta, seed)
