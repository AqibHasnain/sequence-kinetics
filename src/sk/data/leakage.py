"""Sequence-overlap leakage between scored variants and each fold's training sequences.

Chromosome hold-out keeps every scored variant off its fold's training chromosomes, but a
paralog, a retrocopy or a shared repeat on a training chromosome can carry the same sequence. A
variant is flagged for a fold when any 32-mer that contains the variant position (reference
allele) occurs exactly, on either strand, in the input of a gene that fold trains on (gradient
or early-stopping genes). Flagged variants are excluded from that fold's scoring. An MPRA
element's overlap is the share of its 32-mers found in those inputs; it is reported per element.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sk.data import kinetic_inputs
from sk.data.genome import encode, fetch, revcomp
from sk.data.splits import load_cv_folds
from sk.model import inputs as mi
from sk.paths import PROCESSED, RESULTS
from sk.utils.logging import get_logger

log = get_logger(__name__)
K = 32
OUT = RESULTS / "phase2" / "leakage.json"


def kmer_codes(codes: np.ndarray) -> np.ndarray:
    """uint64 code of every K-mer of a base-code array; K-mers with N or padding get no code."""
    codes = np.asarray(codes, dtype=np.uint64)
    n = len(codes) - K + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    bad = codes > 3
    out = np.zeros(n, dtype=np.uint64)
    for j in range(K):
        out = (out << np.uint64(2)) | (codes[j : j + n] & np.uint64(3))
    has_bad = np.convolve(bad.astype(np.int32), np.ones(K, dtype=np.int32), "valid") > 0
    return out[~has_bad]


def _seq_kmers(seq: str) -> np.ndarray:
    return np.union1d(kmer_codes(encode(seq)), kmer_codes(encode(revcomp(seq))))


def gene_kmers(gene_ids: list[str]) -> dict[str, np.ndarray]:
    ids, enc = mi.gene_inputs()
    rows = ids.get_indexer(gene_ids)
    return {g: np.unique(kmer_codes(enc.codes[r])) for g, r in zip(gene_ids, rows, strict=True)}


def variant_windows(v: pd.DataFrame) -> list[np.ndarray]:
    """K-mers (both strands) of the reference sequence that contain each variant position."""
    out = []
    for row in v.itertuples():
        out.append(_seq_kmers(fetch(row.chrom, row.pos0 - K + 1, row.pos0 + K)))
    return out


def main() -> None:
    import yaml

    from sk.paths import CONFIGS

    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet").set_index("gene_id")
    modeled = [g for s in ("train", "validation", "test") for g in kinetic_inputs.modeled_genes(s)]
    kmers = gene_kmers(modeled)
    chrom = genes.loc[modeled, "chrom"]
    variants = {
        name: pd.read_parquet(PROCESSED / "variants" / f"{name}.parquet").drop_duplicates(
            "variant_id"
        )
        for name in ("promoter_enhancer", "utr3")
    }
    report: dict = {"k": K, "folds": {}}
    for fold, held in load_cv_folds().items():
        train = [g for g in modeled if chrom[g] not in held]
        assert cfg["validation_chrom"][fold] not in held
        pool = np.unique(np.concatenate([kmers[g] for g in train]))
        fold_report = {"n_training_genes": len(train), "flagged": {}, "element_share": {}}
        for name, v in variants.items():
            v = v[v["fold"] == fold]
            windows = variant_windows(v)
            owner = np.repeat(np.arange(len(windows)), [len(w) for w in windows])
            found = np.isin(np.concatenate(windows), pool, assume_unique=False)
            hit = np.bincount(owner[found], minlength=len(windows)) > 0
            fold_report["flagged"][name] = sorted(v.loc[np.array(hit, dtype=bool), "variant_id"])
            log.info("%s %s: %d of %d variants flagged", fold, name, int(hit.sum()), len(v))
            if name == "promoter_enhancer":
                for element_id, g in v.groupby("element_id"):
                    lo = int((g["pos0"]).min())
                    hi = int((g["pos0"]).max()) + 1
                    km = _seq_kmers(fetch(g["chrom"].iloc[0], lo, hi))
                    fold_report["element_share"][element_id] = float(np.isin(km, pool).mean())
        report["folds"][fold] = fold_report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")


def flagged() -> set[tuple[str, str]]:
    """(fold, variant_id) pairs excluded for sequence overlap."""
    report = json.loads(OUT.read_text())
    return {
        (fold, vid)
        for fold, r in report["folds"].items()
        for ids in r["flagged"].values()
        for vid in ids
    }


if __name__ == "__main__":
    main()
