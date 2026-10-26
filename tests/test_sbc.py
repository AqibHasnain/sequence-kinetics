import numpy as np
import pytest

from sk.infer.sbc import coverage, ranks, uniformity

N, L, BINS = 2000, 99, 20


@pytest.mark.parametrize(
    "sd_scale,shift,calibrated",
    [(1.0, 0.0, True), (0.6, 0.0, False), (1.6, 0.0, False), (1.0, 0.3, False)],
)
def test_sbc_on_conjugate_gaussian(sd_scale, shift, calibrated):
    """Catches rank or coverage code that passes a miscalibrated posterior or fails an exact one.

    theta ~ N(0, 1), y ~ N(theta, 0.5^2): the posterior is N(0.8 y, 0.2) exactly. Posteriors
    that are too narrow, too wide or shifted must fail.
    """
    rng = np.random.default_rng(1)
    theta = rng.normal(size=N)
    y = theta + 0.5 * rng.normal(size=N)
    mean, sd = 0.8 * y + shift, np.sqrt(0.2) * sd_scale
    draws = mean[:, None] + sd * rng.normal(size=(N, L))
    p = uniformity(ranks(theta, draws), L, BINS)["p"]
    cov = coverage(theta, mean, np.full(N, sd))
    passed = p > 0.01 and 0.85 <= cov <= 0.95
    assert passed == calibrated
