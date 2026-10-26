import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from sk.kinetics.closed_form import labeled_fraction_m, labeled_fraction_u
from sk.kinetics.fisher import jacobian, jacobian_rank
from sk.kinetics.observations import DESIGNS, GeneData, NoiseParams, expected

T = np.array([0.01, 0.083, 0.25, 1.0, 4.0, 24.0])
D = torch.float64


@pytest.mark.parametrize(
    "beta,gamma",
    [(6.0, 0.2), (0.2, 6.0), (1.0, 1.0), (1.0, 1.0 + 1e-9), (0.05, 0.05), (30.0, 0.01)],
)
def test_labeled_fractions_match_numerical_integration(beta, gamma):
    """Catches a wrong closed form, including the beta = gamma limit and cancellation near it."""

    def rhs(_, y):
        u, m = y
        return [1.0 - beta * u, beta * u - gamma * m]

    sol = solve_ivp(rhs, (0, T[-1]), [0.0, 0.0], t_eval=T, rtol=1e-12, atol=1e-15, method="DOP853")
    f_u = sol.y[0] * beta  # labeled u over u* = alpha / beta, alpha = 1
    f_m = sol.y[1] * gamma
    b, c, t = (
        torch.tensor(math.log(beta), dtype=D),
        torch.tensor(math.log(gamma), dtype=D),
        torch.tensor(T, dtype=D),
    )
    np.testing.assert_allclose(labeled_fraction_u(b, t).numpy(), f_u, rtol=1e-6)
    np.testing.assert_allclose(labeled_fraction_m(b, c, t).numpy(), f_m, rtol=1e-6)


def _gene_data(n: int) -> GeneData:
    """Covariates of a typical gene with every observation present; counts are unused."""
    ones = torch.ones(n, dtype=D)
    return GeneData(
        wc_exon=torch.zeros(n, 4, dtype=D),
        wc_intron=torch.zeros(n, 4, dtype=D),
        len_exon=4.0 * ones,
        len_intron=40.0 * ones,
        tl_y=torch.zeros(n, 4, 2, dtype=D),
        tl_sd=torch.full((n, 4, 2), 0.02, dtype=D),
        tl_mask=torch.ones(n, 4, 2, dtype=torch.bool),
        tl_bg=0.02 * ones,
        tt_y=torch.zeros(n, 2, 2, dtype=D),
        tt_total=torch.full((n, 2), 100.0, dtype=D),
        tt_mask=torch.ones(n, 2, dtype=torch.bool),
        tt_len_u=torch.tensor([[5.0, 20.0]], dtype=D).expand(n, 2),
        tt_len_m=torch.tensor([[3.0, 0.0]], dtype=D).expand(n, 2),
    )


def _theta(n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.tensor([0.0, 0.5, -1.7], dtype=D) + torch.randn(n, 3, generator=g, dtype=D)


@pytest.mark.parametrize(
    "terms,direction",
    [
        (("timelapse", "ttseq"), (1.0, 0.0, 0.0)),  # labeled fractions do not depend on alpha
        (DESIGNS["SS-EI"], (1.0, 1.0, 1.0)),  # steady state is invariant to joint rate scaling
    ],
)
def test_observations_invariant_along_unidentified_direction(terms, direction):
    """Catches a likelihood that leaks information along a direction section 2.2 says is free."""
    noise = NoiseParams(dtype=D)
    theta = _theta(50)
    shifted = theta + 0.7 * torch.tensor(direction, dtype=D)
    data = _gene_data(50)
    base, moved = expected(theta, data, noise, terms), expected(shifted, data, noise, terms)
    for name in base:
        torch.testing.assert_close(moved[name].mean, base[name].mean, rtol=1e-12, atol=0)


@pytest.mark.parametrize("design,rank", [("SS-E", 1), ("SS-EI", 2), ("KIN", 3), ("KIN-1T", 3)])
def test_jacobian_rank_matches_identifiability_table(design, rank):
    """Catches a design whose observation map identifies more or fewer directions than claimed."""
    jac, _ = jacobian(_theta(100), _gene_data(100), NoiseParams(dtype=D), DESIGNS[design])
    assert (jacobian_rank(jac) == rank).all()
