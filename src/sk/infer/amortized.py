"""Amortized variational inference: the sequence network outputs q(a, b, c | sequence).

Training maximizes the ELBO summed over genes (section 6.5): the expected log-likelihood of each
gene's observations under q, by reparameterized Monte Carlo, minus KL(q || prior). The prior and
the assay noise parameters are each arm's Phase 1 empirical-Bayes values, held fixed
(sk.model.priors). The network runs on the accelerator in float32; the likelihood and KL run on
CPU in float64, the same code as Phase 1, with gradients flowing back across devices.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from sk.kinetics.observations import GeneData, NoiseParams, log_likelihood

LABELING_FIELDS = (
    "tl_y",
    "tl_sd",
    "tl_mask",
    "tl_bg",
    "tt_y",
    "tt_total",
    "tt_mask",
    "tt_len_u",
    "tt_len_m",
)


def expression_deciles(data: GeneData) -> np.ndarray:
    """Decile (0-9) of each gene's mean whole-cell exonic count per kb."""
    level = (data.wc_exon.mean(-1) / data.len_exon).numpy()
    ranks = level.argsort().argsort()
    return (ranks * 10 // len(level)).astype(int)


def shuffle_labeling(data: GeneData, seed: int) -> tuple[GeneData, np.ndarray]:
    """KIN-SHUF: move every gene's labeling data to another gene of the same expression decile.

    The labeling block (TimeLapse-seq fractions, their SDs, mask and background; TT-seq labeled
    counts, the matching total counts and bin lengths) moves as a unit, so each gene keeps a real
    gene's labeling profile and the marginal within each decile is unchanged, but the profile
    no longer belongs to the gene whose sequence and steady state the network sees.
    Returns the shuffled data and the source index of each gene's labeling block.
    """
    rng = np.random.default_rng(seed)
    decile = expression_deciles(data)
    source = np.arange(len(data))
    for d in np.unique(decile):
        idx = np.flatnonzero(decile == d)
        source[idx] = rng.permutation(idx)
    src = torch.as_tensor(source)
    moved = {name: getattr(data, name)[src] for name in LABELING_FIELDS}
    return replace(data, **moved), source


@dataclass
class Batch:
    idx: np.ndarray
    codes: torch.Tensor
    flags: torch.Tensor
    covariates: torch.Tensor


def gaussian_elbo(
    mean: torch.Tensor,
    tril: torch.Tensor,
    prior: MultivariateNormal,
    noise: NoiseParams,
    data: GeneData,
    terms: tuple[str, ...],
    n_samples: int,
) -> torch.Tensor:
    """Per-gene ELBO [B] for network outputs on any device; computed on CPU in float64."""
    mean = mean.cpu().double()
    tril = tril.cpu().double()
    eps = torch.randn(n_samples, *mean.shape, dtype=torch.float64)
    theta = mean + torch.einsum("gij,ngj->ngi", tril, eps)
    expected_ll = log_likelihood(theta, data, noise, terms).mean(0)
    q = MultivariateNormal(mean, scale_tril=tril)
    return expected_ll - kl_divergence(q, prior)


def data_subset(data: GeneData, idx: np.ndarray) -> GeneData:
    t = torch.as_tensor(idx)
    return GeneData(**{f.name: getattr(data, f.name)[t] for f in fields(data)})
