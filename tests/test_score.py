import numpy as np
import pandas as pd
import pytest
import torch

from sk.eval import score
from sk.model import inputs as mi
from sk.model.inputs import COVARIATES
from sk.model.network import NetConfig, SequenceNet
from sk.paths import PROCESSED

UTR3 = PROCESSED / "variants" / "utr3.parquet"


@pytest.mark.skipif(
    not UTR3.exists() or not (PROCESSED / "inputs").exists(), reason="processed data not built"
)
def test_utr_reporter_scores_each_insert_with_its_own_covariates(tmp_path, monkeypatch):
    """Each insert's reference prediction must use that insert's construct, so a batch of
    inserts with different GC gives the same means as scoring each insert alone."""
    v = pd.read_parquet(UTR3)
    v = v[v["fold"] == "fold0"].drop_duplicates(["variant_id", "insert_ref"])
    gc = v["insert_ref"].str.count("[GC]") / v["insert_ref"].str.len()
    v = v.loc[[gc.idxmin(), gc.idxmax()]]
    v.to_parquet(tmp_path / "utr3.parquet")
    monkeypatch.setattr(score, "VARIANTS", tmp_path)

    torch.manual_seed(0)
    net = SequenceNet(NetConfig(n_covariates=len(COVARIATES)), torch.zeros(3)).eval()
    out = score.utr_reporter(net, "fold0", "cpu")

    for row, insert in zip(out.itertuples(), v["insert_ref"], strict=True):
        codes, flags, cov, _ = mi.utr_reporter_input(insert)
        alone, _ = score.predict(net, codes[None], flags[None], cov[None], "cpu")
        np.testing.assert_allclose([row.ref_a, row.ref_b, row.ref_c], alone[0], rtol=0, atol=1e-6)
