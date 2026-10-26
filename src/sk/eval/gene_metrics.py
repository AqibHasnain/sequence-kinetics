"""Gene-level fit (section 8.1) and the amortization gap (section 6.5), from out-of-fold posteriors.

Every gene's posterior comes from the fold model that held its chromosome out.

- Spearman of posterior-mean c with log SLAM-seq decay rate (an assay no arm trains on) and with
  log TimeLapse-seq decay rate; of posterior-mean a with a TT-seq synthesis proxy (labeled
  fragments per kb of bins, size-factor normalized); and of the predicted log exonic level with
  the observed whole-cell level.
- 90% posterior predictive coverage of held-out observations of modeled genes: draws of theta
  from q, observations simulated under the arm's noise model, coverage of the central 90%.
- Amortization gap on the 200 chr7 genes of results/phase3/nuts: KL from the NUTS posterior
  (Gaussian moment match) to the amortized posterior of the fold1 model, the same KL to the
  prior, and the shift of posterior means in NUTS posterior SDs.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from sk.data import kinetic_inputs
from sk.data.splits import load_cv_folds, load_splits
from sk.eval.phase3 import FOLDS, TRAINED, completed_seeds
from sk.kinetics.observations import simulate
from sk.model.priors import arm_noise, arm_prior
from sk.paths import PROCESSED, RESULTS, RUNS
from sk.utils.logging import get_logger

log = get_logger(__name__)
OUT = RESULTS / "phase3"
LN2 = math.log(2)


def posteriors(arm: str, seed: int) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(RUNS / "phase2" / arm / f"seed{seed}" / f / "genes.parquet")
            for f in FOLDS
        ]
    ).set_index("gene_id")


def _cov(df: pd.DataFrame) -> np.ndarray:
    c = np.empty((len(df), 3, 3))
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        c[:, i, j] = c[:, j, i] = df[f"cov_{'abc'[i]}{'abc'[j]}"].to_numpy()
    return c


def references() -> pd.DataFrame:
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet").set_index("gene_id")
    hl = pd.read_parquet(PROCESSED / "k562" / "half_lives.parquet")
    k562 = PROCESSED / "k562"
    wc = pd.read_parquet(k562 / "whole_cell_counts.parquet").set_index("gene_id")
    exon = wc.filter(like="exon_wc")
    sf = exon.sum() / exon.sum().mean()
    tt = pd.read_parquet(k562 / "ttseq_counts.parquet").set_index("gene_id")
    lab = tt[["exon_L_with_1", "exon_L_with_2", "intron_L_with_1", "intron_L_with_2"]]
    lab_sf = lab.sum() / lab.sum().mean()
    kb = 0.2 * (tt["n_exon_bins"] + tt["n_intron_bins"])
    from sk.data.genome import gene_regions

    regions = gene_regions()
    len_exon = pd.Series(
        {g: regions[g].exon_length / 1e3 for g in genes.index if g in regions}, dtype=float
    )
    return pd.DataFrame(
        {
            "chrom": genes["chrom"],
            "log_decay_slam": np.log(LN2 / hl["half_life_slamseq_h"]),
            "log_decay_timelapse": np.log(LN2 / hl["half_life_timelapse_h"]),
            "log_ttseq_synthesis_proxy": np.log(((lab / lab_sf).mean(axis=1) + 0.5) / kb),
            "log_exon_level": np.log((exon / sf).mean(axis=1) + 0.5),
            "len_exon_kb": len_exon,
        }
    )


def _rho(x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    return {"rho": float(spearmanr(x[ok], y[ok]).statistic), "n": int(ok.sum())}


def correlations(post: pd.DataFrame, ref: pd.DataFrame, test_only: bool) -> dict:
    df = post.join(ref, how="inner")
    if test_only:
        df = df[df["chrom"].isin(load_splits()["test"])]
    pred_level = np.log(df["len_exon_kb"]) + np.logaddexp(
        df["mean_a"] - df["mean_b"], df["mean_a"] - df["mean_c"]
    )
    return {
        "c_vs_slamseq_decay": _rho(df["mean_c"], df["log_decay_slam"]),
        "c_vs_timelapse_decay": _rho(df["mean_c"], df["log_decay_timelapse"]),
        "a_vs_ttseq_synthesis_proxy": _rho(df["mean_a"], df["log_ttseq_synthesis_proxy"]),
        "predicted_vs_observed_exonic_level": _rho(pred_level, df["log_exon_level"]),
    }


def coverage(arm: str, post: pd.DataFrame, n_draws: int = 200, seed: int = 0) -> dict:
    """Central 90% posterior predictive coverage of held-out observations, per term."""
    torch.manual_seed(seed)
    ids = [g for s in ("train", "validation", "test") for g in kinetic_inputs.modeled_genes(s)]
    ids = [g for g in ids if g in post.index]
    data = kinetic_inputs.build(ids)
    p = post.loc[ids]
    mean = torch.tensor(p[["mean_a", "mean_b", "mean_c"]].to_numpy())
    cov = torch.tensor(_cov(p))
    q = torch.distributions.MultivariateNormal(mean, cov)
    noise = arm_noise(arm)
    sims = {"wc_exon": [], "wc_intron": [], "timelapse": []}
    for _ in range(n_draws):
        s = simulate(q.sample(), data, noise)
        sims["wc_exon"].append(s.wc_exon)
        sims["wc_intron"].append(s.wc_intron)
        sims["timelapse"].append(s.tl_y)
    obs = {"wc_exon": data.wc_exon, "wc_intron": data.wc_intron, "timelapse": data.tl_y}
    out = {}
    for k, v in sims.items():
        v = torch.stack(v)
        lo, hi = torch.quantile(v.double(), 0.05, 0), torch.quantile(v.double(), 0.95, 0)
        inside = (obs[k] >= lo) & (obs[k] <= hi)
        if k == "timelapse":
            inside = inside[data.tl_mask]
        out[k] = float(inside.double().mean())
    out["n_genes"] = len(ids)
    return out


def _gauss_kl(m0, s0, m1, s1) -> np.ndarray:
    """KL(N(m0, s0) || N(m1, s1)) per gene."""
    s1_inv = np.linalg.inv(s1)
    d = m1 - m0
    return 0.5 * (
        np.einsum("gij,gji->g", s1_inv, s0)
        + np.einsum("gi,gij,gj->g", d, s1_inv, d)
        - 3
        + np.linalg.slogdet(s1)[1]
        - np.linalg.slogdet(s0)[1]
    )


def amortization_gap(arm: str, seed: int) -> dict | None:
    path = RESULTS / "phase3" / "nuts" / f"nuts_{arm}.npz"
    run = RUNS / "phase2" / arm / f"seed{seed}" / "fold1" / "genes.parquet"
    if not path.exists() or not run.exists():
        return None
    assert "chr7" in load_cv_folds()["fold1"]
    nuts = np.load(path, allow_pickle=True)
    ids = [str(g) for g in nuts["gene_ids"]]
    post = pd.read_parquet(run).set_index("gene_id").loc[ids]
    m_a = post[["mean_a", "mean_b", "mean_c"]].to_numpy()
    s_a = _cov(post)
    m_n, s_n = nuts["mean"], nuts["cov"]
    pm, pc = arm_prior(arm)
    pm = np.broadcast_to(pm.numpy(), m_n.shape)
    pc = np.broadcast_to(pc.numpy(), s_n.shape)
    sd_n = np.sqrt(np.diagonal(s_n, axis1=1, axis2=2))
    kl_amort = _gauss_kl(m_n, s_n, m_a, s_a)
    kl_prior = _gauss_kl(m_n, s_n, pm, pc)
    shift = np.abs(m_a - m_n) / sd_n
    rho = {r: float(spearmanr(m_a[:, i], m_n[:, i]).statistic) for i, r in enumerate("abc")}
    return {
        "n_genes": len(ids),
        "kl_nuts_to_amortized_mean": float(kl_amort.mean()),
        "kl_nuts_to_amortized_median": float(np.median(kl_amort)),
        "kl_nuts_to_prior_mean": float(kl_prior.mean()),
        "kl_nuts_to_prior_median": float(np.median(kl_prior)),
        "mean_shift_in_nuts_sd_median_abc": np.median(shift, 0).tolist(),
        "mean_shift_in_nuts_sd_mean_abc": shift.mean(0).tolist(),
        "amortized_sd_median_abc": np.median(
            np.sqrt(np.diagonal(s_a, axis1=1, axis2=2)), 0
        ).tolist(),
        "nuts_sd_median_abc": np.median(sd_n, 0).tolist(),
        "spearman_amortized_vs_nuts_mean_abc": rho,
        "_kl_amort": kl_amort,
        "_kl_prior": kl_prior,
        "_shift": shift,
    }


def main() -> None:
    ref = references()
    out: dict = {"arms": {}}
    for arm in TRAINED:
        for seed in completed_seeds(arm):
            post = posteriors(arm, seed)
            entry = {
                "test_chromosomes": correlations(post, ref, True),
                "all_genes": correlations(post, ref, False),
            }
            if seed == 0:
                entry["coverage90"] = coverage(arm, post)
                gap = amortization_gap(arm, seed)
                if gap is not None:
                    entry["amortization_gap"] = {
                        k: v for k, v in gap.items() if not k.startswith("_")
                    }
            out["arms"].setdefault(arm, {})[f"seed{seed}"] = entry
            log.info("%s seed%d: %s", arm, seed, json.dumps(entry["test_chromosomes"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gene_metrics.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
