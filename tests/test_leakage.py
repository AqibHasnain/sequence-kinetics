import numpy as np
import pandas as pd
import pytest

from sk.data.genome import encode, revcomp
from sk.data.leakage import OUT, K, _seq_kmers, kmer_codes
from sk.paths import PROCESSED, RUNS

DATA = PROCESSED / "inputs" / "codes.npy"


def test_kmer_scan_finds_a_shared_window_on_either_strand_and_skips_n():
    """A broken K-mer encoding would let a training copy of a variant's context go unflagged."""
    gen = np.random.default_rng(0)
    context = "".join(gen.choice(list("ACGT"), 2 * K - 1))
    other = "".join(gen.choice(list("ACGT"), 300))
    training = kmer_codes(encode(other[:100] + revcomp(context) + other[100:]))
    assert np.isin(_seq_kmers(context), training).any()
    assert not np.isin(_seq_kmers(other[: 2 * K - 1]), kmer_codes(encode(context))).any()
    assert len(kmer_codes(encode("A" * K + "N" + "C" * (K - 1)))) == 1


@pytest.mark.skipif(not DATA.exists(), reason="processed data not built")
def test_no_scored_variant_or_element_overlaps_its_fold_training_genes():
    """A gene or element shared between a fold's training set and its scored variants would let
    the model score variants in sequence it was fitted to."""
    from sk.data.splits import load_cv_folds
    from sk.train import fold_genes, load_config

    cfg = load_config()
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet").set_index("gene_id")
    for fold, held in load_cv_folds().items():
        train, valid = fold_genes(fold, cfg)
        fitted = set(train) | set(valid)
        assert not set(genes.loc[list(fitted), "chrom"]) & held
        for name in ("promoter_enhancer", "utr3"):
            v = pd.read_parquet(PROCESSED / "variants" / f"{name}.parquet")
            v = v[v["fold"] == fold]
            assert set(v["chrom"]) <= held
            assert not set(v["gene_id"].dropna()) & fitted
    if OUT.exists():
        import json

        report = json.loads(OUT.read_text())
        for r in report["folds"].values():
            assert all(share == 0.0 for share in r["element_share"].values())


@pytest.mark.skipif(not (RUNS / "phase2").exists() or not OUT.exists(), reason="no runs")
def test_phase3_drops_variants_flagged_for_sequence_overlap():
    from sk.data.leakage import flagged
    from sk.eval.phase3 import TRAINED, _load, completed_seeds

    bad = flagged()
    for arm in TRAINED:
        seeds = completed_seeds(arm)
        if seeds:
            df = _load(arm, seeds[0], "reporter")
            assert not {(f, k) for f, k in zip(df["fold"], df["key"], strict=True)} & bad
            return
    pytest.skip("no completed seed")
