"""Gate G2: are the routing variant sets large enough under chromosome cross-validation?

Project plan, section 9: at least 150 significant promoter or enhancer variants and at least 150
significant 3' UTR variants are scoreable. Routing uses 5-fold chromosome cross-validation, so a
variant is scoreable when its chromosome is held out by one of the folds. The Phase 4 LCL
criterion is evaluated only if Phase 4 runs.

A substitution measured several times (time points, replicates, cell lines) counts once, and
counts as significant or null when any of its measurements is, as in Gate G0.
"""

from __future__ import annotations

import json

import pandas as pd

from sk.paths import PROCESSED, RESULTS
from sk.utils.logging import get_logger

log = get_logger(__name__)

MIN_SIGNIFICANT = 150
VARIANTS = PROCESSED / "variants"


def unique_variants(v: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    flags = v.groupby("variant_id")[["significant", "null_variant"]].max()
    first = v.drop_duplicates("variant_id").set_index("variant_id")[keep]
    return first.join(flags)


def _per_fold(u: pd.DataFrame) -> dict:
    return {
        fold: {"significant": int(g["significant"].sum()), "null": int(g["null_variant"].sum())}
        for fold, g in u.groupby("fold")
    }


def promoter_enhancer() -> dict:
    v = pd.read_parquet(VARIANTS / "promoter_enhancer.parquet")
    if (v.groupby("variant_id")["element_id"].nunique() > 1).any():
        raise ValueError("a variant is assigned to more than one element")
    cell_lines = v.groupby("element_id")["cell_line"].unique().map(sorted)
    u = unique_variants(v, ["element_id", "chrom", "fold"])
    u = u[u["fold"] != "excluded"]
    elements = {
        element: {
            "cell_lines": list(cell_lines[element]),
            "chrom": g["chrom"].iloc[0],
            "fold": g["fold"].iloc[0],
            "significant": int(g["significant"].sum()),
            "null": int(g["null_variant"].sum()),
        }
        for element, g in u.groupby("element_id")
    }
    n_sig = int(u["significant"].sum())
    return {
        "unique_significant_scoreable": n_sig,
        "unique_null_scoreable": int(u["null_variant"].sum()),
        "unique_significant_and_null": int((u["significant"] & u["null_variant"]).sum()),
        "n_elements": len(elements),
        "elements_by_fold": u.groupby("fold")["element_id"].nunique().to_dict(),
        "by_fold": _per_fold(u),
        "by_element": elements,
        "threshold": MIN_SIGNIFICANT,
        "pass": n_sig >= MIN_SIGNIFICANT,
    }


def utr3() -> dict:
    v = pd.read_parquet(VARIANTS / "utr3.parquet")
    u = unique_variants(v, ["fold"])
    excluded = int(u.loc[u["fold"] == "excluded", "significant"].sum())
    u = u[u["fold"] != "excluded"]
    n_sig = int(u["significant"].sum())
    return {
        "unique_significant_scoreable": n_sig,
        "unique_null_scoreable": int(u["null_variant"].sum()),
        "unique_significant_and_null": int((u["significant"] & u["null_variant"]).sum()),
        "unique_significant_on_excluded_chromosomes": excluded,
        "by_fold": _per_fold(u),
        "threshold": MIN_SIGNIFICANT,
        "pass": n_sig >= MIN_SIGNIFICANT,
    }


def main() -> None:
    report = {
        "regime": "5-fold chromosome cross-validation (configs/splits.yaml cv_folds)",
        "promoter_enhancer": promoter_enhancer(),
        "utr3": utr3(),
        "lcl_discordant_lead_variants": "not evaluated; Phase 4 is conditional",
    }
    report["pass"] = report["promoter_enhancer"]["pass"] and report["utr3"]["pass"]
    out = RESULTS / "phase0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_g2.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log.info("Gate G2 %s", "passes" if report["pass"] else "fails")


if __name__ == "__main__":
    main()
