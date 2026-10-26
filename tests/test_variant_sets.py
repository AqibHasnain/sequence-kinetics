import json

import pandas as pd
import pytest

from sk.data.genome import fetch
from sk.data.splits import normalize_chrom
from sk.data.variants import SATMUT
from sk.paths import PROCESSED

VARIANTS = PROCESSED / "variants"
pytestmark = pytest.mark.skipif(
    not (VARIANTS / "utr3.parquet").exists(), reason="processed data not built"
)


@pytest.mark.parametrize("name", ["promoter_enhancer", "utr3"])
def test_reference_allele_matches_hg38(name):
    """A build or strand error in a source table would put the wrong allele in the input."""
    v = pd.read_parquet(VARIANTS / f"{name}.parquet")
    bases = [fetch(c, p, p + 1) for c, p in zip(v["chrom"], v["pos0"], strict=True)]
    assert (pd.Series(bases, index=v.index) == v["hg38_base"]).all()
    assert v["ref_matches_hg38"].all()


def test_alternate_insert_differs_at_exactly_the_variant_position():
    v = pd.read_parquet(VARIANTS / "utr3.parquet")
    ref = v["insert_ref"].to_numpy()
    alt = v["insert_alt"].to_numpy()
    offsets = v["insert_offset"].to_numpy()
    for r, a, o in zip(ref, alt, offsets, strict=True):
        assert len(r) == len(a)
        assert [i for i in range(len(r)) if r[i] != a[i]] == [o]


def test_reporter_elements_reproduce_the_published_insert():
    """The assayed element must equal hg38 at its recorded coordinates.

    satMutMPRA variant positions are offsets into this element, so a coordinate or build
    error here would shift every variant in the set.
    """
    v = pd.read_parquet(VARIANTS / "promoter_enhancer.parquet")
    for path in sorted(SATMUT.glob("*.meta.json")):
        meta = json.loads(path.read_text())
        coords = meta["extraMetadata"]
        element = meta["targetGenes"][0]["targetSequence"]["sequence"]
        chrom = normalize_chrom(str(coords["chr"]))
        assert fetch(chrom, coords["start"] - 1, coords["end"]) == element
        rows = v[v["source"] == path.name.split(".")[0]]
        assert len(rows)
        for row in rows.itertuples():
            assert element[row.element_offset] == row.ref
            assert row.pos0 == coords["start"] - 1 + row.element_offset
