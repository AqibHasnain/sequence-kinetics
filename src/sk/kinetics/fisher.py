"""Fisher information of (a, b, c) under each design, and the Jacobian of its observation map."""

from __future__ import annotations

import torch

from sk.kinetics.observations import GeneData, NoiseParams, expected


def _observations(
    theta: torch.Tensor, data: GeneData, noise: NoiseParams, terms: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noise-free observation vector [G, D] and the Fisher weight of each entry.

    Count terms enter as log means, whose Fisher weight under a negative binomial is
    mu r / (r + mu); Gaussian terms enter as means with weight 1 / sd^2. Masked entries get
    weight 0.
    """
    obs, weights = [], []
    for term in expected(theta, data, noise, terms).values():
        if term.kind == "nb":
            obs.append(torch.log(term.mean))
            w = term.mean * term.scale / (term.scale + term.mean)
        else:
            obs.append(term.mean)
            w = 1 / term.scale**2
        weights.append(torch.where(term.mask, w.expand_as(term.mean), torch.zeros_like(term.mean)))
    return torch.cat(obs, -1), torch.cat(weights, -1)


def jacobian(
    theta: torch.Tensor, data: GeneData, noise: NoiseParams, terms: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobian [G, D, 3] of the observation vector with respect to theta [G, 3], and weights."""
    theta = theta.detach().requires_grad_(True)
    obs, weights = _observations(theta, data, noise, terms)
    rows = [
        torch.autograd.grad(obs[:, d].sum(), theta, retain_graph=True)[0]
        for d in range(obs.shape[1])
    ]
    return torch.stack(rows, 1), weights.detach()


def fisher_information(
    theta: torch.Tensor, data: GeneData, noise: NoiseParams, terms: tuple[str, ...]
) -> torch.Tensor:
    """Expected Fisher information [G, 3, 3] at theta, with global parameters held fixed."""
    jac, w = jacobian(theta, data, noise, terms)
    return torch.einsum("gd,gdi,gdj->gij", w, jac, jac)


def jacobian_rank(jac: torch.Tensor, rtol: float = 1e-9) -> torch.Tensor:
    """Numerical rank of each [D, 3] Jacobian: singular values above rtol x the largest."""
    s = torch.linalg.svdvals(jac)
    return (s > rtol * s[..., :1]).sum(-1)


def laplace_posterior_cov(info: torch.Tensor, prior_cov: torch.Tensor) -> torch.Tensor:
    """Posterior covariance (I + Sigma_prior^-1)^-1 of the Gaussian approximation."""
    return torch.linalg.inv(info + torch.linalg.inv(prior_cov))


def contraction(post_cov: torch.Tensor, prior_cov: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """1 - posterior variance / prior variance of the linear combination v . theta."""
    return 1 - (v @ post_cov @ v) / (v @ prior_cov @ v)
