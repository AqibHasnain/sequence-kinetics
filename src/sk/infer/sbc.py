"""Simulation-based calibration (Talts et al. 2018): rank statistics and interval coverage."""

from __future__ import annotations

import numpy as np
from scipy.stats import chisquare, norm


def ranks(truth: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Rank of each true value among its L posterior draws, in 0..L.

    truth has shape [N] and draws [N, L]. Under a calibrated posterior the rank is uniform on
    0..L. Ties, which have probability zero for continuous draws, count as below.
    """
    return (draws < truth[:, None]).sum(axis=1)


def uniformity(rank: np.ndarray, n_draws: int, n_bins: int) -> dict:
    """Chi-square test of the rank histogram against uniform, over n_bins equal-width bins.

    The L + 1 possible ranks must divide evenly into the bins, so every bin has the same
    expected count.
    """
    if (n_draws + 1) % n_bins:
        raise ValueError("n_draws + 1 must be a multiple of n_bins")
    counts = np.bincount(rank * n_bins // (n_draws + 1), minlength=n_bins)
    test = chisquare(counts)
    return {"chi2": float(test.statistic), "p": float(test.pvalue), "counts": counts.tolist()}


def coverage(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray, level: float = 0.9) -> float:
    """Share of true values inside the central Gaussian interval at the given level."""
    z = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(truth - mean) <= z * sd))
