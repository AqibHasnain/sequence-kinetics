import pandas as pd
import pytest

from sk.data.splits import fold_of, load_cv_folds, load_splits, split_of
from sk.paths import PROCESSED

GENES = PROCESSED / "genes" / "genes.parquet"
VARIANT_SETS = ("promoter_enhancer", "utr3")


def test_config_assigns_every_autosome_once_and_excludes_sex_and_mito():
    splits = load_splits()
    assigned = [c for name in ("train", "validation", "test") for c in splits[name]]
    assert sorted(assigned) == sorted(f"chr{i}" for i in range(1, 23))
    assert {split_of(c) for c in ("chrX", "chrY", "chrM", "chr1_KI270706v1_random")} == {None}
    assert split_of("2") == "test"


def test_cv_folds_partition_the_assigned_chromosomes():
    folds = load_cv_folds()
    assert len(folds) == 5
    assert all(fold_of(c) is not None for c in load_splits()["train"])
    assert fold_of("chrX") is None


@pytest.mark.skipif(not GENES.exists(), reason="processed data not built")
def test_gene_and_variant_splits_follow_the_config():
    splits = load_splits()
    genes = pd.read_parquet(GENES)
    assert set(genes.loc[genes["split"] == "train", "chrom"]) <= splits["train"]
    assert set(genes.loc[genes["split"] == "test", "chrom"]) == splits["test"]
    for name in VARIANT_SETS:
        path = PROCESSED / "variants" / f"{name}.parquet"
        if not path.exists():
            continue
        v = pd.read_parquet(path)
        expected = v["chrom"].map(lambda c: split_of(c) or "excluded")
        assert (v["split"] == expected).all()
        # Routing uses cross-validation: a variant is scored only by the fold model that held
        # its chromosome out, so the recorded fold must be that fold.
        expected_fold = v["chrom"].map(lambda c: fold_of(c) or "excluded")
        assert (v["fold"] == expected_fold).all()
        assert (v.loc[v["split"] != "excluded", "fold"] != "excluded").all()
