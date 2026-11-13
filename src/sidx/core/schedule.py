"""Flow-linear noise schedule and the DDIM / eta update (paper Eqs. 3-6).

x_t = alpha_t * x + sigma_t * eps with sigma_t = 1 - alpha_t.

Index convention (official code, notes D1): index 0 is the clean terminal (alpha, sigma) = (1, 0)
and is never denoised; indices 1..T are the evaluation points, with
alpha = linspace(1 - sigma_min, 0, T). So the least noisy evaluation is at sigma = sigma_min
(1e-4), NOT at 1/T as the paper's text implies. The step from index 1 returns x_hat itself.
"""

from __future__ import annotations

from typing import Optional

import torch


class FlowLinearSchedule:
    def __init__(self, num_steps: int = 10, sigma_min: float = 1e-4) -> None:
        a = torch.linspace(1 - sigma_min, 0.0, num_steps, dtype=torch.float32)
        self.alphas = torch.cat([torch.ones(1), a])
        self.sigmas = torch.cat([torch.zeros(1), 1.0 - a])
        self.num_steps = num_steps

    @property
    def sigma_max(self) -> float:
        return float(self.sigmas[-1])

    def eps_from_x_hat(self, xt: torch.Tensor, x_hat: torch.Tensor, t: int) -> torch.Tensor:
        assert t != 0
        return (xt - self.alphas[t] * x_hat) / self.sigmas[t]

    def step(self, x_hat: torch.Tensor, eps_hat: torch.Tensor, t: int, eta: float = 0.0,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Move from index t to t-1 (Eq. 5, c = eta * sigma_{t-1}). Fresh noise is drawn only for
        eta > 0 and not on the step onto the terminal, in the same order as the official code."""
        assert 0.0 <= eta <= 1.0
        if t - 1 == 0:
            return x_hat
        alpha, sigma = self.alphas[t - 1], self.sigmas[t - 1]
        if eta == 0.0:
            return alpha * x_hat + sigma * eps_hat
        z = torch.randn(x_hat.shape, dtype=x_hat.dtype, device=x_hat.device, generator=generator)
        c = eta * sigma
        return alpha * x_hat + (sigma ** 2 - c ** 2) ** 0.5 * eps_hat + c * z
