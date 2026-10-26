"""Post hoc: checks on the gene-level decay result and on the amortization gap.

    python -m sk.eval.gene_diagnostics

Added after the gene-level and amortization results were seen (DECISIONS, 2026-10-26). Writes
results/phase3/gene_diagnostics.json.

Decay result, seed 0, the primary test-chromosome genes with a SLAM-seq rate:
- integrity: every gene predicted once, by the fold model that held its chromosome out, and the
  same genes for every arm;
- paired gene bootstrap of Spearman rho differences between arms;
- a covariate-only baseline: per fold, ridge regression of the Phase 1 KIN posterior-mean c on
  the ten gene covariates, fitted on that fold's training genes and predicting its held-out
  genes, and each arm's partial rank correlation with SLAM-seq given that prediction;
- the arms on genes stratified by the share of their 32-mers found in their fold's training
  and early-stopping inputs;
- KIN and KIN-SHUF on a sample of their own training genes against held-out genes (folds 1, 3);
- the TimeLapse-seq against SLAM-seq ceiling.

Amortization gap, fold1 seed 0: posterior SDs on training, early-stopping and held-out genes;
for KIN, the KL from the Phase 1 per-gene posterior and exonic 90% predictive coverage on
training and early-stopping genes; and the held-out KL and coverage after widening the network
posterior by the empirical covariance of its mean errors against NUTS.
"""

from __future__ import annotations

import argparse
import json
from functools import cache, lru_cache

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from sk.data import kinetic_inputs
from sk.data.leakage import kmer_codes
from sk.data.splits import load_cv_folds, load_splits
from sk.eval import gene_metrics as gm
from sk.eval.phase3 import FOLDS, interval
from sk.eval.score import predict
from sk.kinetics.observations import simulate
from sk.model import inputs as mi
from sk.model.priors import arm_noise
from sk.paths import RESULTS, RUNS
from sk.train import fold_genes, load_config, load_trained
from sk.utils.logging import get_logger

log = get_logger(__name__)

ARMS = ("KIN", "KIN-SHUF", "SS-EI", "SS-E", "KIN-1T")
SEED = 0
N_BOOT = 2000
RIDGE = 1e-3
GAP_FOLD = "fold1"
GAP_ARMS = ("KIN", "SS-EI", "KIN-1T")
N_TRAIN_SAMPLE = 400
N_EARLYSTOP_SAMPLE = 200
MEMORIZATION_FOLDS = ("fold1", "fold3")
N_MEMORIZATION_SAMPLE = 700


@cache
def _fold_genes(fold: str) -> tuple[list[str], list[str]]:
    return fold_genes(fold, load_config())


@lru_cache(maxsize=1)
def _phase1_kin() -> tuple[list[str], np.ndarray, np.ndarray]:
    fits = torch.load(RUNS / "phase1" / "fits_full.pt", weights_only=False)
    f = fits["fits"]["KIN"]
    tril = f["q_scale_tril"].double()
    return list(fits["gene_ids"]), f["q_mean"].double().numpy(), (tril @ tril.mT).numpy()


def _rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def _partial(x, y, z) -> float:
    """Rank correlation of x and y after removing a linear fit on the ranks of z from each."""
    rz = rankdata(z)

    def resid(v):
        rv = rankdata(v)
        return rv - np.polyval(np.polyfit(rz, rv, 1), rz)

    return float(np.corrcoef(resid(x), resid(y))[0, 1])


def _run(net, ids: list[str], device: str) -> tuple[np.ndarray, np.ndarray]:
    all_ids, enc = mi.gene_inputs()
    rows = all_ids.get_indexer(ids)
    order = np.argsort(rows)
    r = rows[order]
    m, c = predict(net, enc.codes[r], enc.flags[r], enc.covariates[r], device)
    back = np.argsort(order)
    return m[back], c[back]


# ------------------------------------------------------------------- gene-level decay


def integrity(ref: pd.DataFrame) -> dict:
    folds = load_cv_folds()
    off_fold, overlap, duplicates = 0, 0, 0
    for arm in ARMS:
        seen: list[str] = []
        for fold in FOLDS:
            g = pd.read_parquet(RUNS / "phase2" / arm / f"seed{SEED}" / fold / "genes.parquet")
            off_fold += int((~ref["chrom"].reindex(g["gene_id"]).isin(folds[fold])).sum())
            tr, va = _fold_genes(fold)
            overlap += len(set(g["gene_id"]) & (set(tr) | set(va)))
            seen += list(g["gene_id"])
        duplicates += len(seen) - len(set(seen))
    return {
        "genes_off_their_fold": off_fold,
        "genes_also_in_training_or_early_stopping": overlap,
        "genes_predicted_twice": duplicates,
    }


def covariate_baseline(ids: pd.Index) -> pd.Series:
    """Out-of-fold ridge prediction of Phase 1 KIN posterior-mean c from the gene covariates."""
    all_ids, enc = mi.gene_inputs()
    x = pd.DataFrame(mi.transform_covariates(np.asarray(enc.covariates)), index=all_ids)
    p1_ids, p1_mean, _ = _phase1_kin()
    target = pd.Series(p1_mean[:, 2], index=p1_ids)
    chrom = gm.references()["chrom"]
    folds = load_cv_folds()
    out = pd.Series(np.nan, index=ids)
    for fold in FOLDS:
        tr = [g for g in _fold_genes(fold)[0] if g in target.index]
        x1 = np.c_[np.ones(len(tr)), x.loc[tr].to_numpy()]
        w = np.linalg.solve(x1.T @ x1 + RIDGE * np.eye(x1.shape[1]), x1.T @ target.loc[tr])
        held = ids[chrom.reindex(ids).isin(folds[fold]).to_numpy()]
        out[held] = np.c_[np.ones(len(held)), x.loc[held].to_numpy()] @ w
    return out


def kmer_share(ids: pd.Index) -> pd.Series:
    """Share of each gene's input 32-mers found in its fold's training or early-stopping inputs."""
    all_ids, enc = mi.gene_inputs()
    row = {g: i for i, g in enumerate(all_ids)}
    chrom = gm.references()["chrom"]
    folds = load_cv_folds()
    share = pd.Series(np.nan, index=ids)
    for fold in FOLDS:
        held = ids[chrom.reindex(ids).isin(folds[fold]).to_numpy()]
        if not len(held):
            continue
        tr, va = _fold_genes(fold)
        pool = np.unique(
            np.concatenate([np.unique(kmer_codes(enc.codes[row[g]])) for g in tr + va])
        )
        for g in held:
            k = np.unique(kmer_codes(enc.codes[row[g]]))
            pos = np.clip(np.searchsorted(pool, k), 0, len(pool) - 1)
            share[g] = float((pool[pos] == k).mean()) if len(k) else 0.0
    return share


def memorization(ref: pd.DataFrame, device: str) -> dict:
    """c against SLAM-seq on a sample of a fold model's training genes and on its held-out genes."""
    folds = load_cv_folds()
    slam = ref["log_decay_slam"]
    out = {}
    for arm in ("KIN", "KIN-SHUF"):
        post = gm.posteriors(arm, SEED).join(ref, how="inner")
        for fold in MEMORIZATION_FOLDS:
            tr = [g for g in _fold_genes(fold)[0] if np.isfinite(slam.get(g, np.nan))]
            rng = np.random.default_rng(SEED)
            tr = sorted(rng.choice(tr, min(N_MEMORIZATION_SAMPLE, len(tr)), replace=False))
            m, _ = _run(load_trained(arm, SEED, fold, device), tr, device)
            held = post[post["chrom"].isin(folds[fold]) & np.isfinite(post["log_decay_slam"])]
            out[f"{arm}/{fold}"] = {
                "training_genes": {"rho": _rho(m[:, 2], slam.loc[tr]), "n": len(tr)},
                "held_out_genes": {
                    "rho": _rho(held["mean_c"], held["log_decay_slam"]),
                    "n": len(held),
                },
            }
    return out


def decay_headline(device: str) -> dict:
    ref = gm.references()
    test = load_splits()["test"]
    post = {a: gm.posteriors(a, SEED) for a in ARMS}
    ids = post["KIN"].index
    ids = ids[ref["chrom"].reindex(ids).isin(test).to_numpy()]
    ids = ids[np.isfinite(ref["log_decay_slam"].reindex(ids)).to_numpy()]
    slam = ref.loc[ids, "log_decay_slam"].to_numpy()
    c = {a: post[a]["mean_c"].reindex(ids).to_numpy() for a in ARMS}
    assert all(np.isfinite(v).all() for v in c.values()), "arms differ in their test genes"
    base = covariate_baseline(ids).to_numpy()
    c["covariates only"] = base

    pairs = [
        ("KIN", "KIN-SHUF"),
        ("KIN", "SS-EI"),
        ("KIN-SHUF", "SS-EI"),
        ("KIN", "covariates only"),
    ]
    rng = np.random.default_rng(SEED)
    boot = {f"{a} - {b}": [] for a, b in pairs}
    boot_partial = {a: [] for a in ("KIN", "KIN-SHUF", "SS-EI")}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(ids), len(ids))
        r = {a: _rho(c[a][i], slam[i]) for a in c}
        for a, b in pairs:
            boot[f"{a} - {b}"].append(r[a] - r[b])
        for a in boot_partial:
            boot_partial[a].append(_partial(c[a][i], slam[i], base[i]))
    point = {a: _rho(c[a], slam) for a in c}

    share = kmer_share(ids)
    strata = {
        "no shared 32-mer": (share == 0).to_numpy(),
        "share <= 0.01": (share <= 0.01).to_numpy(),
        "share > 0.01": (share > 0.01).to_numpy(),
    }
    tl = ref.loc[ids, "log_decay_timelapse"].to_numpy()
    ok = np.isfinite(tl)
    return {
        "n_genes": len(ids),
        "integrity": integrity(ref),
        "rho": point,
        "paired_difference": {
            k: {
                "estimate": point[k.split(" - ")[0]] - point[k.split(" - ")[1]],
                "ci95": interval(v),
            }
            for k, v in boot.items()
        },
        "partial_rho_given_covariate_baseline": {
            a: {"estimate": _partial(c[a], slam, base), "ci95": interval(np.array(v))}
            for a, v in boot_partial.items()
        },
        "by_kmer_share": {
            label: {
                "n": int(m.sum()),
                **{a: _rho(c[a][m], slam[m]) for a in ("KIN", "KIN-SHUF", "SS-EI")},
            }
            for label, m in strata.items()
        },
        "kmer_share_quantiles_50_90_99": share.quantile([0.5, 0.9, 0.99]).tolist(),
        "memorization": memorization(ref, device),
        "ceiling_timelapse_vs_slam": {"rho": _rho(tl[ok], slam[ok]), "n": int(ok.sum())},
    }


# ------------------------------------------------------------------- amortization gap


def _sd(cov: np.ndarray) -> list[float]:
    return np.median(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), 0).tolist()


def exon_coverage(arm: str, ids: list[str], mean: np.ndarray, cov: np.ndarray) -> float:
    """Central 90% posterior predictive coverage of whole-cell exonic counts."""
    torch.manual_seed(SEED)
    data = kinetic_inputs.build(ids)
    q = torch.distributions.MultivariateNormal(torch.tensor(mean), torch.tensor(cov))
    noise = arm_noise(arm)
    sims = torch.stack([simulate(q.sample(), data, noise).wc_exon for _ in range(200)]).double()
    lo, hi = torch.quantile(sims, 0.05, 0), torch.quantile(sims, 0.95, 0)
    return float(((data.wc_exon >= lo) & (data.wc_exon <= hi)).double().mean())


def amortization(device: str) -> dict:
    p1_ids, p1_mean, p1_cov = _phase1_kin()
    p1_index = {g: i for i, g in enumerate(p1_ids)}
    rng = np.random.default_rng(SEED)
    tr, va = _fold_genes(GAP_FOLD)
    tr = sorted(rng.choice([g for g in tr if g in p1_index], N_TRAIN_SAMPLE, replace=False))
    va = sorted(rng.choice([g for g in va if g in p1_index], N_EARLYSTOP_SAMPLE, replace=False))
    out = {}
    for arm in GAP_ARMS:
        nuts = np.load(RESULTS / "phase3" / "nuts" / f"nuts_{arm}.npz", allow_pickle=True)
        held = [str(g) for g in nuts["gene_ids"]]
        post = gm.posteriors(arm, SEED).loc[held]
        m_h, s_h = post[["mean_a", "mean_b", "mean_c"]].to_numpy(), gm._cov(post)
        m_n, s_n = nuts["mean"], nuts["cov"]
        widened = s_h + np.cov((m_h - m_n).T)
        net = load_trained(arm, SEED, GAP_FOLD, device)
        m_t, s_t = _run(net, tr, device)
        m_v, s_v = _run(net, va, device)
        entry = {
            "posterior_sd_median_abc": {
                "training_genes": _sd(s_t),
                "early_stopping_genes": _sd(s_v),
                "held_out_genes": _sd(s_h),
                "held_out_nuts": _sd(s_n),
            },
            "held_out": {
                "sd_of_mean_error_vs_nuts_abc": (m_h - m_n).std(0).tolist(),
                "kl_nuts_to_amortized_median": float(np.median(gm._gauss_kl(m_n, s_n, m_h, s_h))),
                "kl_amortized_to_nuts_median": float(np.median(gm._gauss_kl(m_h, s_h, m_n, s_n))),
                "kl_nuts_to_widened_median": float(np.median(gm._gauss_kl(m_n, s_n, m_h, widened))),
                "exon_coverage90_amortized": exon_coverage(arm, held, m_h, s_h),
                "exon_coverage90_widened": exon_coverage(arm, held, m_h, widened),
                "exon_coverage90_nuts": exon_coverage(arm, held, m_n, s_n),
            },
        }
        if arm == "KIN":
            for label, ids, m, s in (("training", tr, m_t, s_t), ("early_stopping", va, m_v, s_v)):
                idx = [p1_index[g] for g in ids]
                m_r, s_r = p1_mean[idx], p1_cov[idx]
                entry[f"{label}_vs_phase1_posterior"] = {
                    "n": len(ids),
                    "sd_of_mean_error_abc": (m - m_r).std(0).tolist(),
                    "kl_phase1_to_amortized_median": float(np.median(gm._gauss_kl(m_r, s_r, m, s))),
                    "exon_coverage90_amortized": exon_coverage(arm, ids, m, s),
                    "exon_coverage90_phase1": exon_coverage(arm, ids, m_r, s_r),
                }
        out[arm] = entry
        log.info("%s: %s", arm, json.dumps(entry["held_out"]))
    return {"fold": GAP_FOLD, "seed": SEED, "arms": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = {
        "post_hoc": True,
        "decay": decay_headline(args.device),
        "amortization": amortization(args.device),
    }
    path = RESULTS / "phase3" / "gene_diagnostics.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
