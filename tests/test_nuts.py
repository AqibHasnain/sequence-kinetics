import numpy as np
import pytest
import torch

from sk.paths import PROCESSED, RUNS

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "k562").exists() or not (RUNS / "phase1" / "fits_full.pt").exists(),
    reason="processed data or Phase 1 fits not built",
)


@pytest.mark.parametrize("arm", ["KIN", "SS-EI", "KIN-1T"])
def test_jax_log_density_matches_torch(arm):
    """Catches a JAX log density that diverges from the training likelihood plus prior, which
    would make the NUTS reference posterior answer a different question."""
    from torch.distributions import MultivariateNormal

    from sk.data import kinetic_inputs
    from sk.infer import nuts
    from sk.kinetics.observations import log_likelihood
    from sk.model.priors import arm_noise, arm_prior, terms

    gene_ids = nuts.selected_genes()[:4]
    data = kinetic_inputs.build(gene_ids)
    mean, cov = arm_prior(arm)
    noise = arm_noise(arm)
    arm_terms = terms(arm)
    torch.manual_seed(0)
    theta = MultivariateNormal(mean, 4 * cov).sample((20, len(gene_ids)))  # [20, G, 3]
    # b = c exercises the series branch of the mature labeled fraction.
    theta[0, :, 2] = theta[0, :, 1]
    expected = log_likelihood(theta, data, noise, arm_terms) + MultivariateNormal(
        mean, cov
    ).log_prob(theta)

    arrays, noise_np = nuts.gene_arrays(data), nuts.noise_arrays(noise)
    got = np.array(
        [
            [
                nuts.log_density(
                    theta[s, g].numpy(),
                    {k: v[g] for k, v in arrays.items()},
                    noise_np,
                    mean.numpy(),
                    cov.numpy(),
                    arm_terms,
                )
                for g in range(len(gene_ids))
            ]
            for s in range(theta.shape[0])
        ]
    )
    np.testing.assert_allclose(got, expected.numpy(), rtol=1e-8)
