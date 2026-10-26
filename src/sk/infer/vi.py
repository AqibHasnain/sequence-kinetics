"""Per-gene variational posteriors under a learned hierarchical Gaussian prior (section 6.5).

Every gene has a free Gaussian q(a, b, c). The prior N(mu, Sigma) and the assay noise
parameters are global and fitted by maximizing the evidence lower bound summed over genes
(empirical Bayes). The expected log-likelihood is estimated with reparameterized Monte Carlo
samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import MultivariateNormal, kl_divergence

from sk.kinetics.observations import GeneData, NoiseParams, log_likelihood

FAMILIES = ("diag", "full")
_TRIL = torch.tril_indices(3, 3, offset=-1)


def _scale_tril(log_diag: torch.Tensor, offdiag: torch.Tensor) -> torch.Tensor:
    lower = torch.zeros(*log_diag.shape, 3, dtype=log_diag.dtype)
    lower[..., _TRIL[0], _TRIL[1]] = offdiag
    return torch.diag_embed(torch.exp(log_diag)) + lower


class GaussianPrior(nn.Module):
    def __init__(self, mean: torch.Tensor, scale_tril: torch.Tensor | None = None):
        super().__init__()
        scale_tril = torch.eye(3, dtype=mean.dtype) if scale_tril is None else scale_tril
        self.mean = nn.Parameter(mean.clone())
        self.log_diag = nn.Parameter(torch.log(torch.diagonal(scale_tril)).clone())
        self.offdiag = nn.Parameter(scale_tril[_TRIL[0], _TRIL[1]].clone())

    def scale_tril(self) -> torch.Tensor:
        return _scale_tril(self.log_diag, self.offdiag)

    def cov(self) -> torch.Tensor:
        tril = self.scale_tril()
        return tril @ tril.T

    def dist(self) -> MultivariateNormal:
        return MultivariateNormal(self.mean, scale_tril=self.scale_tril())


class GenePosteriors(nn.Module):
    """One Gaussian q per gene; "diag" fixes the off-diagonal Cholesky entries at zero."""

    def __init__(self, mean: torch.Tensor, family: str, log_sd: float = -1.0):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.family = family
        self.mean = nn.Parameter(mean.clone())
        self.log_sd = nn.Parameter(torch.full_like(mean, log_sd))
        self.offdiag = nn.Parameter(torch.zeros_like(mean), requires_grad=family == "full")

    def scale_tril(self) -> torch.Tensor:
        return _scale_tril(self.log_sd, self.offdiag)

    def dist(self) -> MultivariateNormal:
        return MultivariateNormal(self.mean, scale_tril=self.scale_tril())

    def rsample(self, n: int) -> torch.Tensor:
        eps = torch.randn(n, *self.mean.shape, dtype=self.mean.dtype)
        return self.mean + torch.einsum("gij,ngj->ngi", self.scale_tril(), eps)


def elbo_per_gene(
    q: GenePosteriors,
    prior: GaussianPrior,
    noise: NoiseParams,
    data: GeneData,
    terms: tuple[str, ...],
    n_samples: int,
) -> torch.Tensor:
    theta = q.rsample(n_samples)
    expected_ll = log_likelihood(theta, data, noise, terms).mean(0)
    return expected_ll - kl_divergence(q.dist(), prior.dist())


@dataclass
class FitConfig:
    steps: int = 4000
    lr: float = 0.03
    lr_final: float = 1e-3
    n_samples: int = 5
    fit_prior: bool = True
    # Names of NoiseParams parameters to optimize; None means every one the design touches.
    noise_trainable: tuple[str, ...] | None = None
    log_every: int = 250


def fit(
    q: GenePosteriors,
    prior: GaussianPrior,
    noise: NoiseParams,
    data: GeneData,
    terms: tuple[str, ...],
    cfg: FitConfig,
) -> list[dict]:
    """Maximize the ELBO over q, and over the prior and noise parameters when they are free."""
    params = [q.mean, q.log_sd] + ([q.offdiag] if q.family == "full" else [])
    if cfg.fit_prior:
        params += list(prior.parameters())
    for name, p in noise.named_parameters():
        if cfg.noise_trainable is None or name.split(".")[0] in cfg.noise_trainable:
            params.append(p)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    history = []
    for step in range(cfg.steps):
        # Cosine decay from lr to lr_final.
        frac = step / max(cfg.steps - 1, 1)
        for group in opt.param_groups:
            group["lr"] = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (
                1 + math.cos(math.pi * frac)
            )
        opt.zero_grad()
        elbo = elbo_per_gene(q, prior, noise, data, terms, cfg.n_samples)
        loss = -elbo.sum() / len(data)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.append({"step": step, "elbo_per_gene": -loss.item()})
    return history


def initial_means(data: GeneData, log_size_factor: torch.Tensor, c0: float) -> torch.Tensor:
    """Moment-based starting point: a - b and a - c from the steady-state counts, c = c0.

    Every gene starts at the same c, so designs without labeling data keep no gene-specific
    decay information beyond what their own likelihood supplies.
    """
    s = torch.exp(log_size_factor)
    u = ((data.wc_intron + 0.5) / s).mean(-1) / data.len_intron
    total = ((data.wc_exon + 0.5) / s).mean(-1) / data.len_exon
    m = torch.clamp(total - u, min=0.1 * total)
    c = torch.full_like(u, c0)
    a = c + torch.log(m)
    b = a - torch.log(u)
    return torch.stack([a, b, c], -1)
