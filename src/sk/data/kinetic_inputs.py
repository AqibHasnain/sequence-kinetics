"""Assemble the per-gene K562 observations into the tensors the kinetic likelihood reads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sk.data.genome import gene_regions
from sk.kinetics.observations import TIMELAPSE_PULSES_H, GeneData
from sk.paths import PROCESSED, RESULTS

K562 = PROCESSED / "k562"
BIN_KB = 0.2
# The deposited 0.05 and 0.95 posterior quantiles span 2 x 1.645 standard deviations of a
# Gaussian.
QUANTILE_SPAN_SD = 2 * 1.6448536269514722


def modeled_genes(split: str) -> list[str]:
    """Genes of one split that meet Gate G0 criterion 1 and have annotated intronic sequence.

    A gene whose basic exons cover its whole span has no intronic length to put intronic
    reads on, so it is left out.
    """
    ids = (RESULTS / "phase0" / "g0_genes.txt").read_text().split()
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet").set_index("gene_id")
    ids = [g for g in ids if genes.at[g, "split"] == split]
    _, len_intron = _lengths(ids)
    return [g for g, li in zip(ids, len_intron, strict=True) if li > 0]


def _lengths(gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    regions = gene_regions()
    exon = np.array([regions[g].exon_length for g in gene_ids], dtype=float)
    span = np.array([regions[g].end - regions[g].start for g in gene_ids], dtype=float)
    return exon / 1e3, (span - exon) / 1e3


def _timelapse(gene_ids: list[str]) -> tuple[np.ndarray, ...]:
    lf = pd.read_parquet(K562 / "labeled_fractions.parquet")
    lf = lf[lf["gene_id"].isin(set(gene_ids))]
    wide = lf.pivot_table(
        index="gene_id",
        columns=["pulse_min", "replicate"],
        values=["fraction_new", "lo", "hi"],
        dropna=False,
    ).reindex(gene_ids)
    bg = wide["fraction_new"][0.0].mean(axis=1)
    bg = bg.fillna(bg.median()).to_numpy()
    pulses_min = [60 * t for t in TIMELAPSE_PULSES_H]
    reps = sorted(lf["replicate"].unique())
    shape = (len(gene_ids), len(pulses_min), len(reps))
    y, sd = np.empty(shape), np.empty(shape)
    for i, t in enumerate(pulses_min):
        for j, r in enumerate(reps):
            y[:, i, j] = wide["fraction_new"][(t, r)].to_numpy()
            sd[:, i, j] = ((wide["hi"][(t, r)] - wide["lo"][(t, r)]) / QUANTILE_SPAN_SD).to_numpy()
    mask = np.isfinite(y) & np.isfinite(sd) & (sd > 0)
    return np.where(mask, y, 0.0), np.where(mask, sd, 1.0), mask, bg


def _median_of_ratios(counts: np.ndarray) -> np.ndarray:
    ok = (counts > 0).all(axis=1)
    log_c = np.log(counts[ok])
    return np.exp(np.median(log_c - log_c.mean(axis=1, keepdims=True), axis=0))


def _ttseq(gene_ids: list[str]) -> tuple[np.ndarray, ...]:
    tt = pd.read_parquet(K562 / "ttseq_counts.parquet").set_index("gene_id")
    regions = ("exon", "intron")
    total_cols = [[f"{r}_T_with_{k}" for k in (1, 2)] for r in regions]
    sf = _median_of_ratios(tt[total_cols[0]].to_numpy(dtype=float))
    tt = tt.reindex(gene_ids)
    y = np.stack(
        [tt[[f"{r}_L_with_{k}" for k in (1, 2)]].to_numpy(dtype=float) for r in regions], axis=1
    )
    total = np.stack([(tt[cols].to_numpy(dtype=float) / sf).mean(axis=1) for cols in total_cols], 1)
    len_u = BIN_KB * tt[["n_exon_bins", "n_intron_bins"]].to_numpy(dtype=float)
    len_m = np.stack([tt["exon_bp_in_exon_bins"].to_numpy(dtype=float) / 1e3, np.zeros(len(tt))], 1)
    mask = np.isfinite(total) & (total > 0) & (len_u > 0) & np.isfinite(y).all(axis=2)
    return (
        np.where(mask[..., None], y, 0.0),
        np.where(mask, total, 1.0),
        mask,
        np.where(mask, len_u, 1.0),
        np.where(mask, len_m, 1.0) * np.array([1.0, 0.0]),
    )


def build(gene_ids: list[str], dtype: torch.dtype = torch.float64) -> GeneData:
    wc = pd.read_parquet(K562 / "whole_cell_counts.parquet").set_index("gene_id").reindex(gene_ids)
    len_exon, len_intron = _lengths(gene_ids)
    tl_y, tl_sd, tl_mask, bg = _timelapse(gene_ids)
    tt_y, tt_total, tt_mask, tt_len_u, tt_len_m = _ttseq(gene_ids)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=dtype)

    data = GeneData(
        wc_exon=t(wc.filter(like="exon_wc").to_numpy(dtype=float)),
        wc_intron=t(wc.filter(like="intron_wc").to_numpy(dtype=float)),
        len_exon=t(len_exon),
        len_intron=t(len_intron),
        tl_y=t(tl_y),
        tl_sd=t(tl_sd),
        tl_mask=torch.as_tensor(tl_mask),
        tl_bg=t(bg),
        tt_y=t(tt_y),
        tt_total=t(tt_total),
        tt_mask=torch.as_tensor(tt_mask),
        tt_len_u=t(tt_len_u),
        tt_len_m=t(tt_len_m),
    )
    for name, value in vars(data).items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"non-finite values in {name}")
    if (data.len_intron <= 0).any() or (data.wc_exon < 0).any():
        raise ValueError("gene without introns or negative counts")
    return data
