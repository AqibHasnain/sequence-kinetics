"""Closed-form solution of the two-compartment kinetic model.

    du/dt = alpha - beta u,    dm/dt = beta u - gamma m

with log rates a = log alpha, b = log beta, c = log gamma. Steady state is u* = alpha/beta and
m* = alpha/gamma. A labeling pulse of duration t started from steady state labels the fractions

    f_u(t) = 1 - exp(-beta t)
    f_m(t) = 1 - (beta exp(-gamma t) - gamma exp(-beta t)) / (beta - gamma)

of each pool. All functions take and return torch tensors and broadcast.
"""

from __future__ import annotations

import torch

_SERIES_CUTOFF = 1e-4


def log_steady_state(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """log u* and log m*, stacked on the last axis."""
    return torch.stack([a - b, a - c], dim=-1)


def _one_minus_exp_neg_over_x(x: torch.Tensor) -> torch.Tensor:
    """(1 - exp(-x)) / x, equal to 1 at x = 0 and finite with finite gradients near it."""
    small = x.abs() < _SERIES_CUTOFF
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-safe) / safe)


def labeled_fraction_u(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-torch.exp(b) * t)


def labeled_fraction_m(b: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Labeled fraction of mature RNA, stable at and near beta = gamma.

    Rewritten as 1 - exp(-gamma t) - gamma t exp(-gamma t) (1 - exp(-(beta - gamma) t)) /
    ((beta - gamma) t), whose last factor tends to 1 as beta -> gamma, giving the limit
    1 - (1 + gamma t) exp(-gamma t).
    """
    beta, gamma = torch.exp(b), torch.exp(c)
    gt = gamma * t
    decay = torch.exp(-gt)
    return -torch.expm1(-gt) - gt * decay * _one_minus_exp_neg_over_x((beta - gamma) * t)


def premrna_weight(
    b: torch.Tensor, c: torch.Tensor, len_u: torch.Tensor, len_m: torch.Tensor
) -> torch.Tensor:
    """Share of a region's steady-state reads that come from pre-mRNA.

    A region spans len_u nucleotides of pre-mRNA and len_m nucleotides of mature mRNA, so its
    reads are proportional to len_u u* + len_m m*, and u*/m* = gamma/beta.
    """
    return torch.sigmoid(torch.log(len_u) - torch.log(len_m) + c - b)


def region_labeled_fraction(
    b: torch.Tensor,
    c: torch.Tensor,
    t: torch.Tensor,
    len_u: torch.Tensor,
    len_m: torch.Tensor,
) -> torch.Tensor:
    """Labeled share of the reads in a region after a pulse of duration t.

    len_m = 0 gives the pre-mRNA fraction (an intronic region).
    """
    w = premrna_weight(b, c, len_u, len_m)
    return w * labeled_fraction_u(b, t) + (1 - w) * labeled_fraction_m(b, c, t)
