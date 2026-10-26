"""Per-gene K562 tables: exonic and intronic counts, labeled fractions and half-lives.

Three assays contribute:
- GSE264556 (Churchman lab): exonic and whole-gene counts per gene for four unperturbed K562
  whole-cell replicates, the steady-state measurements.
- GSE75792 (Schwalb 2016): strand-specific fragment counts in 200 bp bins for total RNA
  (2 fragmented, 2 unfragmented) and 4sU-labeled RNA (TT-seq and 4sU-seq, 5 min pulse),
  summed here over each gene's exonic and intronic bins.
- GSE207924 (Smalec, Ietswaart 2024): subcellular TimeLapse-seq fraction new per gene at 4sU
  pulses of 0, 15, 30, 60 and 120 min, with 0.05 and 0.95 posterior quantiles, and whole-cell
  half-lives from Table S1.
- Wu 2019 (eLife 45396): published K562 SLAM-seq decay rates, the second half-life assay.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from sk.data.genome import GeneRegion, gene_regions
from sk.paths import PROCESSED, RAW
from sk.utils.logging import get_logger

log = get_logger(__name__)

SCHWALB = RAW / "gse75792"
SUBCELLULAR = RAW / "gse264556"
TIMELAPSE = RAW / "gse207924"
WU = RAW / "wu2019_elife" / "elife-45396-fig1-data1-v2.xlsx"
OUT = PROCESSED / "k562"

# GSE75792 libraries: L is 4sU-labeled RNA and T total RNA; "with" and "none" are with and
# without fragmentation, so L_with is TT-seq and T_with its matched fragmented total RNA
# (inferred from the GEO sample titles, which list the same four library types in this order).
TTSEQ_PULSE_MIN = 5.0
# GSE264556 whole-cell libraries, the steady-state total RNA-seq samples.
WHOLE_CELL = ("1A_WC_S1", "1B_WC_S5", "2B_WC_S9", "2C_WC_S13")
TIMELAPSE_PULSE_MIN = (0.0, 15.0, 30.0, 60.0, 120.0)

BIN = 200
# Column order of the binned count table, from the GSE75792 README; each library has a plus- and
# a minus-strand column, and the four ChIP columns that follow are not used.
SCHWALB_BINNED = tuple(
    f"{lib}_{rep}" for lib in ("L_with", "L_none", "T_with", "T_none") for rep in (1, 2)
)


def _bin_labels(
    regions: list[GeneRegion], modeled: dict[str, int], n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Owner gene, exon flag and exonic bp of each 200 bp bin on one chromosome strand.

    A bin is owned by a modeled gene when that gene's span is the only annotated span on the
    strand covering it, so overlapping genes contribute no reads. A bin is exonic when it
    overlaps any exon of its owner; the remaining owned bins are intronic.
    """
    depth = np.zeros(n_bins + 1, dtype=np.int32)
    owner = np.full(n_bins, -1, dtype=np.int64)
    exon_bp = np.zeros(n_bins, dtype=np.int64)
    for g in regions:
        lo, hi = g.start // BIN, min((g.end - 1) // BIN + 1, n_bins)
        depth[lo] += 1
        depth[hi] -= 1
    depth = np.cumsum(depth)[:n_bins]
    for g in regions:
        idx = modeled.get(g.gene_id)
        if idx is None:
            continue
        lo, hi = g.start // BIN, min((g.end - 1) // BIN + 1, n_bins)
        sl = slice(lo, hi)
        owner[sl] = np.where(depth[sl] == 1, idx, owner[sl])
        for s, e in g.exons:
            for b in range(s // BIN, min((e - 1) // BIN + 1, n_bins)):
                exon_bp[b] += min(e, (b + 1) * BIN) - max(s, b * BIN)
    return owner, exon_bp > 0, exon_bp


def ttseq_counts(genes: pd.DataFrame) -> pd.DataFrame:
    """Exonic and intronic read counts per gene and GSE75792 library, from the 200 bp bins.

    The deposited per-transcription-unit count tables are keyed by an index that matches
    neither the deposited annotation's row order nor any other deposited file, so per-gene
    counts are rebuilt from the genome-wide binned table (hg38, strand-specific). Each gene
    reports, per library, the counts in its exonic and its intronic bins, together with the
    number of bins of each kind and the exonic bp in its exonic bins, which the labeled-fraction
    likelihood needs to weight pre-mRNA and mature reads.
    """
    anno = pd.read_csv(
        SCHWALB / "GSE75792_binning_anno.txt.gz", sep="\t", header=None, usecols=[0]
    )[0].to_numpy()
    counts = pd.read_csv(
        SCHWALB / "GSE75792_binned_count_table.txt.gz",
        sep="\t",
        header=None,
        usecols=range(2 * len(SCHWALB_BINNED)),
        dtype=np.int64,
    ).to_numpy()
    regions = gene_regions()
    modeled = {gid: i for i, gid in enumerate(genes["gene_id"])}
    n = len(genes)
    out = {f"{r}_{lib}": np.zeros(n) for r in ("exon", "intron") for lib in SCHWALB_BINNED}
    n_exon_bins, n_intron_bins, exon_bp = np.zeros(n), np.zeros(n), np.zeros(n)
    boundaries = np.flatnonzero(np.r_[True, anno[1:] != anno[:-1], True])
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        chrom = anno[lo]
        for k, strand in enumerate("+-"):
            on = [g for g in regions.values() if g.chrom == chrom and g.strand == strand]
            owner, is_exon, bp = _bin_labels(on, modeled, hi - lo)
            keep = owner >= 0
            o, ex = owner[keep], is_exon[keep]
            n_exon_bins += np.bincount(o[ex], minlength=n)
            n_intron_bins += np.bincount(o[~ex], minlength=n)
            exon_bp += np.bincount(o[ex], weights=bp[keep][ex], minlength=n)
            block = counts[lo:hi][keep]
            for j, lib in enumerate(SCHWALB_BINNED):
                c = block[:, 2 * j + k]
                out[f"exon_{lib}"] += np.bincount(o[ex], weights=c[ex], minlength=n)
                out[f"intron_{lib}"] += np.bincount(o[~ex], weights=c[~ex], minlength=n)
    df = pd.DataFrame(out).astype(np.int64)
    df.insert(0, "gene_id", genes["gene_id"].to_numpy())
    df["n_exon_bins"] = n_exon_bins.astype(np.int64)
    df["n_intron_bins"] = n_intron_bins.astype(np.int64)
    df["exon_bp_in_exon_bins"] = exon_bp.astype(np.int64)
    return df[df["n_exon_bins"] > 0].reset_index(drop=True)


def whole_cell_counts(genes: pd.DataFrame) -> pd.DataFrame:
    """Exonic and intronic counts per gene for the four GSE264556 whole-cell replicates.

    The deposited tables give exonic counts and whole-gene counts; the difference is the
    intronic count. Genes whose exonic count exceeds the gene count in any replicate are
    dropped, since the difference is then not a count.
    """
    tables = {
        level: pd.read_csv(SUBCELLULAR / f"GSE264556_counts_{level}.tsv.gz", sep="\t", index_col=0)[
            list(WHOLE_CELL)
        ]
        for level in ("exon", "gene")
    }
    exon = tables["exon"]
    intron = tables["gene"] - exon
    keep = intron.min(axis=1) >= 0
    exon, intron = exon.loc[keep], intron.loc[keep]
    exon.columns = [f"exon_wc{i + 1}" for i in range(len(WHOLE_CELL))]
    intron.columns = [f"intron_wc{i + 1}" for i in range(len(WHOLE_CELL))]
    out = pd.concat([exon, intron], axis=1).rename_axis("gene_id").reset_index()
    out["gene_id"] = out["gene_id"].str.split(".").str[0]
    return out[out["gene_id"].isin(set(genes["gene_id"]))].reset_index(drop=True)


def _timelapse_fraction_new(path: Path) -> pd.DataFrame:
    """Fraction new per gene and pulse time from one replicate's deposited table.

    The two replicates label their time columns with different prefixes (T and U), so the
    prefix is read from the header.
    """
    df = pd.read_csv(path, sep="\t")
    prefix = df.columns[2].split()[0][0]
    rows = []
    for i, t in enumerate(TIMELAPSE_PULSE_MIN, start=1):
        rows.append(
            pd.DataFrame(
                {
                    "gene_id": df["Gene"].str.split(".").str[0],
                    "pulse_min": t,
                    "fraction_new": df[f"{prefix}{i} Mean"],
                    "lo": df[f"{prefix}{i} 0.05 quantile"],
                    "hi": df[f"{prefix}{i} 0.95 quantile"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _open_nested_tar(path: Path) -> tarfile.TarFile:
    """Open a tar archive, peeling the gzip layers it was deposited with.

    GSE207924's processed-data archive is gzipped twice, which Python's tarfile does not
    unwrap on its own.
    """
    stream: io.BufferedReader | gzip.GzipFile = open(path, "rb")
    while stream.peek(2)[:2] == b"\x1f\x8b":
        stream = io.BufferedReader(gzip.GzipFile(fileobj=stream, mode="rb"))
    return tarfile.open(fileobj=stream, mode="r")


def labeled_fractions() -> pd.DataFrame:
    """TimeLapse-seq fraction new per gene, pulse time and replicate (total RNA fraction)."""
    tar = TIMELAPSE / "GSE207924_processed_data_files.tar.gz"
    work = OUT / "timelapse"
    work.mkdir(parents=True, exist_ok=True)
    frames = []
    with _open_nested_tar(tar) as t:
        for rep in (1, 2):
            name = f"processed_data_files/K562_total_WT_rep{rep}.tsv"
            t.extract(name, path=work, filter="data")
            df = _timelapse_fraction_new(work / name)
            df["replicate"] = rep
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def half_lives() -> pd.DataFrame:
    """Half-lives in hours from two independent K562 assays, per gene."""
    tl = [
        pd.read_excel(TIMELAPSE / "GSE207924_Table_S1.xlsx", sheet_name=f"K562_rates_rep{r}")
        .assign(gene_id=lambda d: d["Gene"].str.split(".").str[0])
        .set_index("gene_id")["half_life_whole_cell.MAP"]
        .apply(pd.to_numeric, errors="coerce")
        for r in (1, 2)
    ]
    timelapse = np.exp(np.log(pd.concat(tl, axis=1)).mean(axis=1)) / 60.0
    slam = pd.read_excel(WU, sheet_name="k562-SLAM-seq").set_index("Name")
    slam_hl = np.log(2) / -slam["decay_rate"].where(slam["decay_rate"] < 0)
    return pd.DataFrame(
        {"half_life_timelapse_h": timelapse, "half_life_slamseq_h": slam_hl}
    ).rename_axis("gene_id")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet")
    whole_cell = whole_cell_counts(genes)
    whole_cell.to_parquet(OUT / "whole_cell_counts.parquet", index=False)
    counts = ttseq_counts(genes)
    counts.to_parquet(OUT / "ttseq_counts.parquet", index=False)
    fractions = labeled_fractions()
    fractions.to_parquet(OUT / "labeled_fractions.parquet", index=False)
    hl = half_lives()
    hl.to_parquet(OUT / "half_lives.parquet")
    log.info(
        "whole-cell genes %d, TT-seq genes %d, labeled fractions %d, both half-lives %d",
        len(whole_cell),
        len(counts),
        fractions["gene_id"].nunique(),
        int(hl.dropna().shape[0]),
    )


if __name__ == "__main__":
    main()
