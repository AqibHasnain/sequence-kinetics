"""Gate G0: is the K562 kinetic ground truth good enough to support a routing claim?

Two criteria from the project plan, section 9:
1. At least 5,000 genes have exonic and intronic counts above 10 in every total RNA-seq sample
   and labeled-fraction data at two or more pulse or chase times.
2. Half-lives from two independent K562 assays agree at Spearman rho >= 0.5 on shared genes.
"""

from __future__ import annotations

import json

import pandas as pd
from scipy.stats import spearmanr

from sk.paths import PROCESSED, RESULTS
from sk.utils.logging import get_logger

log = get_logger(__name__)

MIN_GENES = 5000
MIN_COUNT = 10
MIN_TIMES = 2
MIN_RHO = 0.5
K562 = PROCESSED / "k562"
VARIANTS = PROCESSED / "variants"


def gene_criterion() -> dict:
    counts = pd.read_parquet(K562 / "whole_cell_counts.parquet").set_index("gene_id")
    fractions = pd.read_parquet(K562 / "labeled_fractions.parquet")
    times = fractions.dropna(subset=["fraction_new"]).groupby("gene_id")["pulse_min"].nunique()
    with_counts = set(counts.index[(counts > MIN_COUNT).all(axis=1)])
    with_times = set(times.index[times >= MIN_TIMES])
    genes = sorted(with_counts & with_times)
    return {
        "genes_with_counts": len(with_counts),
        "genes_with_labeled_times": len(with_times),
        "genes_meeting_criterion": len(genes),
        "threshold": MIN_GENES,
        "pass": len(genes) >= MIN_GENES,
        "gene_ids": genes,
    }


def half_life_criterion(gene_ids: list[str] | None = None) -> dict:
    hl = pd.read_parquet(K562 / "half_lives.parquet").dropna()
    rho = spearmanr(hl.iloc[:, 0], hl.iloc[:, 1]).statistic
    out = {
        "assays": list(hl.columns),
        "n_shared_genes": len(hl),
        "spearman_rho": float(rho),
        "threshold": MIN_RHO,
        "pass": bool(rho >= MIN_RHO),
    }
    if gene_ids:
        sub = hl.loc[hl.index.intersection(gene_ids)]
        out["n_shared_genes_in_criterion_1_set"] = len(sub)
        out["spearman_rho_in_criterion_1_set"] = float(
            spearmanr(sub.iloc[:, 0], sub.iloc[:, 1]).statistic
        )
    return out


def variant_counts() -> dict:
    out = {}
    for name in ("promoter_enhancer", "utr3"):
        v = pd.read_parquet(VARIANTS / f"{name}.parquet")
        # A variant tested in several cell lines or at several time points counts once, and
        # counts as significant when any of its measurements is.
        keys = ["chrom", "pos0", "alt"]
        flags = v.groupby(keys)[["significant", "null_variant"]].max()
        unique = (
            v.drop_duplicates(keys)
            .set_index(keys)
            .drop(columns=["significant", "null_variant"])
            .join(flags)
        )
        sig = unique[unique["significant"]]
        out[name] = {
            "rows": len(v),
            "unique_variants": len(unique),
            "unique_significant": len(sig),
            "unique_significant_by_split": sig.groupby("split").size().to_dict(),
            "unique_null": int(unique["null_variant"].sum()),
            "by_source": v.groupby("source" if "source" in v else "element").size().to_dict(),
        }
    return out


def main() -> None:
    genes = gene_criterion()
    half_lives = half_life_criterion(genes["gene_ids"])
    gene_ids = genes.pop("gene_ids")
    report = {
        "criterion_1_genes": genes,
        "criterion_2_half_lives": half_lives,
        "pass": genes["pass"] and half_lives["pass"],
        "variant_sets": variant_counts(),
    }
    out = RESULTS / "phase0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_g0.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    pd.Series(gene_ids, name="gene_id").to_csv(out / "g0_genes.txt", index=False, header=False)
    log.info("Gate G0 %s: %s", "passes" if report["pass"] else "fails", json.dumps(report)[:400])


if __name__ == "__main__":
    main()
