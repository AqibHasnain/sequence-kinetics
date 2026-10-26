import numpy as np
import pytest

from sk.eval.design_sim import SS_IDENTIFIED, conditional_map, independent_prior_cov, route

# A correlated population covariance, so an independent-prior construction that leaks the
# fitted cross-covariance into the unidentified direction changes the predicted deltas.
FITTED = np.array([[1.3, 0.7, 0.26], [0.7, 0.78, 0.19], [0.26, 0.19, 0.76]])
THIRD = 1 / 3


@pytest.mark.parametrize(
    "design,delta,correct,expected,ok,tie",
    [
        ("SS-EI", (1, 0, 0), 0, (2 * THIRD, -THIRD, -THIRD), True, False),
        ("SS-EI", (0, 0, 1), 2, (-THIRD, -THIRD, 2 * THIRD), True, False),
        ("SS-E", (1, 0, 0), 0, (0.5, 0.0, -0.5), False, True),
        ("SS-E", (0, 0, 1), 2, (-0.5, 0.0, 0.5), False, True),
    ],
)
def test_independent_prior_routing(design, delta, correct, expected, ok, tie):
    """Catches an SS-arm map that does not hold the unidentified directions at their prior
    conditional mean, and a router that scores an exact tie as correct.

    Under a prior whose unidentified directions are independent of the identified components,
    the posterior-mean delta is the orthogonal projection of the true delta onto the identified
    row space: SS-EI keeps a single-rate change with margin 0.25, SS-E splits it evenly
    between a and c.
    """
    rows = SS_IDENTIFIED[design]
    m = conditional_map(independent_prior_cov(rows, FITTED), rows)
    pred = m @ np.array(delta, dtype=float)
    np.testing.assert_allclose(pred, expected, atol=1e-12)
    got_ok, got_tie = route(pred[None], correct)
    assert (bool(got_ok[0]), bool(got_tie[0])) == (ok, tie)
