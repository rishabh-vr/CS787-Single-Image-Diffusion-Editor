"""Our re-implementation of the training-free single-image diffusion sampler (Qiu et al., 2026).

Pixel space, exact closed-form denoiser. Runs on CPU or CUDA. It is written from the paper and our
audit of the official code (docs/technical_notes.md §2), and checked against the official code
numerically (tests/test_core_equivalence.py, scripts/check_equivalence.py).

Modules:
    resize    antialiased cubic resampling, the same math as the ResizeRight copy the official code uses
    pyramid   pyramid geometry, build, and the two-scale Laplacian blend
    patches   unfold, Gaussian fold window (rho), weighted fold
    denoiser  closed-form patch denoiser: dense / chunked / SDPA backends, optional argmax logging
    schedule  flow-linear noise schedule and the DDIM/eta update
    sampler   CoreSampler: coarse-to-fine unconditional sampling, with hooks
    hooks     recorders for per-scale results and winning-patch indices
"""

from sidx.core.sampler import CoreConfig, CoreSampler

__all__ = ["CoreConfig", "CoreSampler"]
