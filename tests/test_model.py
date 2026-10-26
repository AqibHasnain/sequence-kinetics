import numpy as np
import pytest
import torch

from sk.infer.amortized import LABELING_FIELDS, expression_deciles, shuffle_labeling
from sk.kinetics.observations import GeneData
from sk.model.inputs import COVARIATES, INPUT_LEN, PROMOTER_LEN
from sk.model.network import NetConfig, SequenceNet


@pytest.mark.parametrize(
    ("position", "rate"),
    [
        (INPUT_LEN - 300, 0),  # a 3'UTR base must be able to move delta a
        (PROMOTER_LEN - 700, 2),  # a promoter base must be able to move delta c
    ],
)
def test_both_segments_reach_every_rate_head(position, rate):
    """DESIGN_CHECK item 2: an architecture that wired a segment to one head would fix routing."""
    torch.manual_seed(0)
    net = SequenceNet(NetConfig(n_covariates=len(COVARIATES)), torch.zeros(3)).eval()
    gen = np.random.default_rng(0)
    codes = torch.as_tensor(gen.integers(0, 4, (1, INPUT_LEN)), dtype=torch.uint8)
    flags = torch.zeros_like(codes)
    cov = torch.zeros(1, len(COVARIATES))
    alt = codes.clone()
    alt[0, position] = (alt[0, position] + 1) % 4
    with torch.no_grad():
        delta = net(alt, flags, cov)[0] - net(codes, flags, cov)[0]
    assert delta[0, rate].abs() > 0


def _gene_data(n: int, seed: int) -> GeneData:
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.rand(*shape, generator=g, dtype=torch.float64)

    return GeneData(
        wc_exon=torch.floor(1000 * r(n, 4)),
        wc_intron=torch.floor(100 * r(n, 4)),
        len_exon=1 + r(n),
        len_intron=1 + r(n),
        tl_y=r(n, 4, 2),
        tl_sd=0.1 * r(n, 4, 2),
        tl_mask=r(n, 4, 2) > 0.1,
        tl_bg=0.05 * r(n),
        tt_y=torch.floor(50 * r(n, 2, 2)),
        tt_total=100 * r(n, 2),
        tt_mask=r(n, 2) > 0.1,
        tt_len_u=1 + r(n, 2),
        tt_len_m=r(n, 2),
    )


def test_kin_shuf_preserves_labeling_marginals_within_expression_deciles():
    """DESIGN_CHECK item 5: a shuffle that changed the labeled-fraction distribution, crossed
    expression deciles, split a gene's labeling block, or touched steady state would let KIN-SHUF
    differ from KIN for reasons other than gene-specific labeling information."""
    data = _gene_data(500, 0)
    shuffled, source = shuffle_labeling(data, seed=1)
    decile = expression_deciles(data)
    assert (decile[source] == decile).all()
    assert (source != np.arange(500)).mean() > 0.8
    for d in range(10):
        idx = np.flatnonzero(decile == d)
        before = np.sort(data.tl_y[idx].flatten().numpy())
        after = np.sort(shuffled.tl_y[idx].flatten().numpy())
        np.testing.assert_array_equal(before, after)
    src = torch.as_tensor(source)
    for name in LABELING_FIELDS:
        assert torch.equal(getattr(shuffled, name), getattr(data, name)[src])
    assert torch.equal(shuffled.wc_exon, data.wc_exon)
    assert torch.equal(shuffled.wc_intron, data.wc_intron)
