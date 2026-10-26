"""Hierarchical prior and assay noise parameters for each arm of the Phase 2 grid.

Every arm uses the global parameters of its design's Phase 1 empirical-Bayes fit
(runs/phase1/fits_full.pt), held fixed during network training:

- KIN, KIN-SHUF: the KIN prior and noise.
- KIN-1T: the KIN-1T prior and noise (its TT-seq scale and contamination are KIN's).
- SS-EI, SS-E: the independent prior (DECISIONS, "SS-arm priors carry no information along
  unidentified directions"). The identified components keep the mean and covariance of the
  design's fit; the orthonormal complement is independent of them with unit variance, and its
  mean is the least-norm shift that puts the prior mean of c at log(ln 2 / 4 h).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from sk.kinetics.observations import DESIGNS, NoiseParams
from sk.paths import RUNS

ARMS = ("SS-E", "SS-EI", "KIN", "KIN-1T", "KIN-SHUF")
DESIGN_OF = {"SS-E": "SS-E", "SS-EI": "SS-EI", "KIN": "KIN", "KIN-1T": "KIN-1T", "KIN-SHUF": "KIN"}
C_PRIOR_MEDIAN = math.log(math.log(2) / 4.0)
SS_IDENTIFIED = {
    "SS-E": np.array([[1.0, 0.0, -1.0]]),
    "SS-EI": np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]),
}


def terms(arm: str) -> tuple[str, ...]:
    return DESIGNS[DESIGN_OF[arm]]


@lru_cache(maxsize=1)
def _phase1_fits() -> dict:
    return torch.load(RUNS / "phase1" / "fits_full.pt", weights_only=False)["fits"]


def orthonormal_complement(rows: np.ndarray) -> np.ndarray:
    """Orthonormal basis [3, 3 - k] of the orthogonal complement of the row space of rows."""
    _, _, vt = np.linalg.svd(rows)
    return vt[rows.shape[0] :].T


def independent_prior(
    identified: np.ndarray, fitted_mean: np.ndarray, fitted_cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    comp = orthonormal_complement(identified)
    basis = np.vstack([identified, comp.T])
    k = identified.shape[0]
    block = np.eye(3)
    block[:k, :k] = identified @ fitted_cov @ identified.T
    inv = np.linalg.inv(basis)
    cov = inv @ block @ inv.T
    # Shift fitted_mean along the complement only, by the least-norm amount that sets mean c.
    gap = C_PRIOR_MEDIAN - fitted_mean[2]
    s = np.linalg.pinv(comp[2:3]) @ np.array([gap])
    mean = fitted_mean + comp @ s
    return mean, cov


def arm_prior(arm: str, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Prior mean [3] and covariance [3, 3] in (a, b, c)."""
    fit = _phase1_fits()[DESIGN_OF[arm]]
    mean, cov = fit["prior_mean"].numpy(), fit["prior_cov"].numpy()
    if arm in SS_IDENTIFIED:
        mean, cov = independent_prior(SS_IDENTIFIED[arm], mean, cov)
    return torch.tensor(mean, dtype=dtype), torch.tensor(cov, dtype=dtype)


def arm_noise(arm: str, dtype: torch.dtype = torch.float64) -> NoiseParams:
    noise = NoiseParams(dtype=dtype)
    state = {k: v.to(dtype) for k, v in _phase1_fits()[DESIGN_OF[arm]]["noise"].items()}
    noise.load_state_dict(state)
    for p in noise.parameters():
        p.requires_grad_(False)
    return noise
