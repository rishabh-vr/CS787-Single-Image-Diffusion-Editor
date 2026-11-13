"""The closed-form patch denoiser (paper Eq. 2 / S18):

    D(x) = sum_j softmax_j( -||x - alpha*y_j||^2 / (2 sigma^2) ) * y_j

over the clean patch dataset {y_j}. ||x||^2 is the same for every j, so it cancels in the softmax
and is left out (paper Eq. S31; the official code does the same):

    logits_j = -( ||alpha*y_j||^2 - 2 x . alpha*y_j ) / (2 sigma^2)

Backends (identical math, different memory/time):
    dense    one (M x N) logit matrix
    chunked  the queries in ``chunks`` pieces, one (M/chunks x N) matrix at a time
    sdpa     torch's fused scaled-dot-product attention through the homogeneous-coordinate trick
             (paper S3.2): q = [x, 1], k = [alpha*y, -||alpha*y||^2 / 2], v = y, scale = 1/sigma^2.
             Never materialises the logit matrix.

``argmax=True`` also returns, for every query, the index of the dataset patch with the largest
weight (the patch that "wins"; at the last step that is the patch the output copies). It is used
by the patch-selection analysis (plan F2). With the sdpa backend it needs one extra chunked pass.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

BACKENDS = ("dense", "chunked", "sdpa")


class PatchDenoiser:
    def __init__(self, dataset: torch.Tensor, backend: str = "dense", chunks: int = 1) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.dataset = dataset.contiguous()            # (N, D)
        self.N, self.D = dataset.shape
        self.backend = backend
        self.chunks = max(1, int(chunks)) if backend != "dense" else 1

    def _logits(self, q: torch.Tensor, ay: torch.Tensor, ay_norm2: torch.Tensor, sigma) -> torch.Tensor:
        return (ay_norm2 - 2.0 * q @ ay.T) * (-1.0 / (2.0 * sigma ** 2))

    def __call__(self, queries: torch.Tensor, alpha, sigma,
                 argmax: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """queries (M, D) -> (denoised (M, D), winning indices (M,) or None)."""
        y = self.dataset
        ay = alpha * y
        ay_norm2 = (ay ** 2).sum(-1).unsqueeze(0)       # (1, N)
        if self.backend == "sdpa":
            out = self._sdpa(queries, ay, ay_norm2, sigma)
            idx = None
            if argmax:
                idx = torch.cat([self._logits(c, ay, ay_norm2, sigma).argmax(1)
                                 for c in torch.chunk(queries, max(self.chunks, 4))])
            return out, idx
        outs, idxs = [], []
        for q in torch.chunk(queries, self.chunks):
            logits = self._logits(q, ay, ay_norm2, sigma)
            outs.append(F.softmax(logits, dim=1) @ y)
            if argmax:
                idxs.append(logits.argmax(1))
        return torch.cat(outs), (torch.cat(idxs) if argmax else None)

    def _sdpa(self, queries, ay, ay_norm2, sigma) -> torch.Tensor:
        M = queries.shape[0]
        q = torch.cat([queries, torch.ones(M, 1, dtype=queries.dtype, device=queries.device)], 1)
        k = torch.cat([ay, -0.5 * ay_norm2.T], 1)
        outs = []
        scale = float(1.0 / sigma ** 2)
        for qc in torch.chunk(q, self.chunks):
            outs.append(F.scaled_dot_product_attention(
                qc[None, None], k[None, None], self.dataset[None, None], scale=scale)[0, 0])
        return torch.cat(outs)
