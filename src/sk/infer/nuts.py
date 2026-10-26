"""Per-gene NUTS reference posteriors for the amortization gap (section 6.5, item 2).

For each arm, the exact posterior of theta = (a, b, c) of every gene in a fixed sample of 200
validation (chr7) genes, under the arm's fixed prior N(arm_prior(arm)), noise arm_noise(arm)
and likelihood terms terms(arm). The log density is a JAX transcription of
sk.kinetics.observations.log_likelihood, so NUTS samples the posterior the networks are trained
against. Writes results/phase3/nuts/nuts_{arm}.npz and nuts_summary.json.

    python -m sk.infer.nuts --arms KIN SS-EI KIN-1T
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from jax.scipy.stats import norm
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS
from scipy.stats import norm as sp_norm
from scipy.stats import rankdata

from sk.data import kinetic_inputs
from sk.kinetics.closed_form import _SERIES_CUTOFF
from sk.kinetics.observations import TIMELAPSE_PULSES_H, TTSEQ_PULSE_H, GeneData, NoiseParams
from sk.paths import RESULTS

jax.config.update("jax_enable_x64", True)

OUT = RESULTS / "phase3" / "nuts"
N_GENES = 200
N_CHAINS = 4
N_WARMUP = 1000
N_SAMPLES = 1000
RHAT_FLAG = 1.01


def gene_arrays(data: GeneData) -> dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in vars(data).items()}


def noise_arrays(noise: NoiseParams) -> dict[str, np.ndarray]:
    return {k: v.detach().numpy() for k, v in noise.state_dict().items()}


def _one_minus_exp_neg_over_x(x):
    small = jnp.abs(x) < _SERIES_CUTOFF
    safe = jnp.where(small, 1.0, x)
    return jnp.where(small, 1 - x / 2 + x * x / 6, -jnp.expm1(-safe) / safe)


def _region_labeled_fraction(b, c, t, len_u, len_m):
    w = jax.nn.sigmoid(jnp.log(len_u) - jnp.log(len_m) + c - b)
    beta, gamma = jnp.exp(b), jnp.exp(c)
    f_u = -jnp.expm1(-beta * t)
    gt = gamma * t
    f_m = -jnp.expm1(-gt) - gt * jnp.exp(-gt) * _one_minus_exp_neg_over_x((beta - gamma) * t)
    return w * f_u + (1 - w) * f_m


def _nb_log_prob(y, mu, size):
    log_mu_r = jnp.log(mu + size)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1)
        + size * (jnp.log(size) - log_mu_r)
        + y * (jnp.log(mu) - log_mu_r)
    )


def log_likelihood(theta, gene: dict, noise: dict, terms: tuple[str, ...]):
    """log p(y | theta) of one gene; gene holds that gene's GeneData fields without the gene
    axis and noise holds the NoiseParams state dict."""
    a, b, c = theta[0], theta[1], theta[2]
    log_sf = noise["log_size_factor"]
    total = 0.0
    if "wc_exon" in terms:
        mu = jnp.exp(jnp.log(gene["len_exon"]) + jnp.logaddexp(a - b, a - c) + log_sf)
        size = jnp.exp(noise["log_nb_size.wc_exon"])
        total = total + _nb_log_prob(gene["wc_exon"], mu, size).sum()
    if "wc_intron" in terms:
        mu = jnp.exp(jnp.log(gene["len_intron"]) + a - b + log_sf)
        size = jnp.exp(noise["log_nb_size.wc_intron"])
        total = total + _nb_log_prob(gene["wc_intron"], mu, size).sum()
    if "timelapse" in terms:
        t = jnp.asarray(TIMELAPSE_PULSES_H)
        h = _region_labeled_fraction(b, c, t, gene["len_exon"], gene["len_exon"])
        bg = gene["tl_bg"]
        mean = (bg + (1 - bg) * h)[:, None]
        sd = jnp.sqrt(
            (jnp.exp(noise["log_tl_sd_scale"]) * gene["tl_sd"]) ** 2
            + jnp.exp(2 * noise["log_tl_extra_sd"])
        )
        lp = norm.logpdf(gene["tl_y"], mean, sd)
        total = total + jnp.where(gene["tl_mask"], lp, 0.0).sum()
    if "ttseq" in terms:
        h = _region_labeled_fraction(b, c, TTSEQ_PULSE_H, gene["tt_len_u"], gene["tt_len_m"])
        eps = jax.nn.sigmoid(noise["logit_tt_contam"])
        captured = h + eps * (1 - h)
        mu = (gene["tt_total"] * captured)[:, None] * jnp.exp(noise["log_tt_scale"])
        size = jnp.concatenate(
            [jnp.exp(noise["log_nb_size.tt_exon"]), jnp.exp(noise["log_nb_size.tt_intron"])]
        )[:, None]
        lp = _nb_log_prob(gene["tt_y"], mu, size)
        total = total + jnp.where(gene["tt_mask"][:, None], lp, 0.0).sum()
    return total


def log_density(theta, gene, noise, prior_mean, prior_cov, terms):
    prior = dist.MultivariateNormal(prior_mean, covariance_matrix=prior_cov).log_prob(theta)
    return prior + log_likelihood(theta, gene, noise, terms)


def _model(gene, noise, prior_mean, prior_cov, terms):
    theta = numpyro.sample(
        "theta", dist.MultivariateNormal(prior_mean, covariance_matrix=prior_cov)
    )
    numpyro.factor("log_lik", log_likelihood(theta, gene, noise, terms))


def _split_chains(x: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average").reshape(x.shape)
    return sp_norm.ppf((r - 0.375) / (x.size + 0.25))


def diagnostics(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-normalized split R-hat (the max of bulk and folded) and bulk ESS per coordinate
    (Vehtari et al. 2021), from draws [chains, draws, 3]."""
    r_hat, ess = np.empty(draws.shape[-1]), np.empty(draws.shape[-1])
    for k in range(draws.shape[-1]):
        x = _split_chains(draws[..., k])
        z = _rank_normalize(x)
        folded = _rank_normalize(np.abs(x - np.median(x)))
        r_hat[k] = max(
            split_gelman_rubin(z[..., None])[0], split_gelman_rubin(folded[..., None])[0]
        )
        ess[k] = effective_sample_size(z[..., None])[0]
    return r_hat, ess


def selected_genes() -> list[str]:
    ids = kinetic_inputs.modeled_genes("validation")
    idx = np.random.default_rng(0).choice(len(ids), N_GENES, replace=False)
    return sorted(ids[i] for i in idx)


def run_arm(arm: str, gene_ids: list[str], data: dict[str, np.ndarray], seed: int = 0) -> dict:
    from sk.model.priors import arm_noise, arm_prior, terms

    prior_mean, prior_cov = (x.numpy() for x in arm_prior(arm))
    noise = noise_arrays(arm_noise(arm))
    arm_terms = terms(arm)
    kernel = NUTS(lambda *args: _model(*args, arm_terms), dense_mass=True)
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=N_CHAINS,
        chain_method="parallel",
        progress_bar=False,
        jit_model_args=True,
    )
    chol = np.linalg.cholesky(prior_cov)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    g = len(gene_ids)
    out = {
        "gene_ids": np.array(gene_ids),
        "mean": np.empty((g, 3)),
        "cov": np.empty((g, 3, 3)),
        "r_hat": np.empty((g, 3)),
        "ess": np.empty((g, 3)),
        "divergences": np.empty(g, dtype=np.int64),
    }
    for i in range(g):
        gene = {k: v[i] for k, v in data.items()}
        # Chains start within 0.1 prior standard deviations of the prior mean, apart from each
        # other so that R-hat can detect chains stuck in different modes.
        init = prior_mean + 0.1 * rng.standard_normal((N_CHAINS, 3)) @ chol.T
        key, sub = jax.random.split(key)
        mcmc.run(
            sub,
            gene,
            noise,
            prior_mean,
            prior_cov,
            init_params={"theta": jnp.asarray(init)},
            extra_fields=("diverging",),
        )
        draws = np.asarray(mcmc.get_samples(group_by_chain=True)["theta"])
        flat = draws.reshape(-1, 3)
        out["mean"][i] = flat.mean(0)
        out["cov"][i] = np.cov(flat, rowvar=False)
        out["r_hat"][i], out["ess"][i] = diagnostics(draws)
        out["divergences"][i] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return out


def _summary(out: dict, seconds: float, start: str, sha: str) -> dict:
    flagged = out["r_hat"].max(axis=1) > RHAT_FLAG
    return {
        "n_genes": len(out["gene_ids"]),
        "max_r_hat": float(out["r_hat"].max()),
        "min_ess": float(out["ess"].min()),
        "total_divergences": int(out["divergences"].sum()),
        "genes_r_hat_gt_1.01": [str(g) for g in out["gene_ids"][flagged]],
        "wall_clock_s": round(seconds, 1),
        "start_utc": start,
        "git_sha": sha,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["KIN"])
    args = ap.parse_args()
    numpyro.set_host_device_count(N_CHAINS)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    gene_ids = selected_genes()
    data = gene_arrays(kinetic_inputs.build(gene_ids))
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "nuts_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for arm in args.arms:
        start = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.perf_counter()
        out = run_arm(arm, gene_ids, data)
        seconds = time.perf_counter() - t0
        np.savez(OUT / f"nuts_{arm}.npz", **out)
        summary[arm] = _summary(out, seconds, start, sha)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(arm, json.dumps(summary[arm]), flush=True)


if __name__ == "__main__":
    main()
