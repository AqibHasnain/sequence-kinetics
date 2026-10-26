"""hg38 sequence access and the GENCODE gene model used by every stage.

One transcript represents each gene: the APPRIS principal transcript of the lowest rank, and
among ties, and for genes without an APPRIS tag, the longest basic transcript. The rule is
fixed here so promoter windows, transcript sequences and covariates are reproducible.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx

from sk.data.splits import split_of
from sk.paths import PROCESSED, RAW

FASTA = RAW / "genome" / "hg38.fa"
GTF = RAW / "genome" / "gencode.v44.basic.annotation.gtf.gz"
GENES = PROCESSED / "genes"

_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i
BASES = np.frombuffer(b"ACGTN", dtype=np.uint8)
_ATTR = re.compile(r'(\S+) "([^"]*)"')
_APPRIS = re.compile(r"appris_principal_(\d)")


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    return BASES[np.asarray(codes, dtype=np.int64)].tobytes().decode("ascii")


@lru_cache(maxsize=1)
def genome() -> pyfaidx.Fasta:
    if not FASTA.exists():
        gz = FASTA.with_suffix(".fa.gz")
        with gzip.open(gz, "rb") as src, FASTA.open("wb") as dst:
            while block := src.read(1 << 24):
                dst.write(block)
    return pyfaidx.Fasta(str(FASTA), as_raw=True, sequence_always_upper=True)


def fetch(chrom: str, start: int, end: int) -> str:
    """Sequence of the 0-based half-open interval, N-padded past either chromosome end."""
    fa = genome()[chrom]
    lo, hi = max(start, 0), min(end, len(fa))
    body = fa[lo:hi] if hi > lo else ""
    return "N" * (lo - start) + body + "N" * (end - hi)


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


@dataclass
class Transcript:
    gene_id: str
    symbol: str
    gene_type: str
    chrom: str
    strand: str
    tx_id: str
    appris: int
    exons: list[tuple[int, int]]
    cds: list[tuple[int, int]]

    @property
    def length(self) -> int:
        return sum(e - s for s, e in self.exons)


def _attrs(field: str) -> dict[str, str]:
    return dict(_ATTR.findall(field))


def parse_gtf(path: Path = GTF) -> dict[str, Transcript]:
    """One Transcript per gene, chosen by the APPRIS-then-length rule."""
    best: dict[str, Transcript] = {}
    current: dict[str, Transcript] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon", "CDS"):
                continue
            a = _attrs(f[8])
            tx_id = a["transcript_id"]
            if f[2] == "transcript":
                m = _APPRIS.search(f[8])
                current[tx_id] = Transcript(
                    gene_id=a["gene_id"].split(".")[0],
                    symbol=a.get("gene_name", ""),
                    gene_type=a["gene_type"],
                    chrom=f[0],
                    strand=f[6],
                    tx_id=tx_id,
                    appris=int(m.group(1)) if m else 9,
                    exons=[],
                    cds=[],
                )
            else:
                tx = current.get(tx_id)
                if tx is None:
                    continue
                (tx.exons if f[2] == "exon" else tx.cds).append((int(f[3]) - 1, int(f[4])))
    for tx in current.values():
        tx.exons.sort()
        tx.cds.sort()
        prev = best.get(tx.gene_id)
        if prev is None or (tx.appris, -tx.length) < (prev.appris, -prev.length):
            best[tx.gene_id] = tx
    return best


@dataclass
class GeneRegion:
    gene_id: str
    chrom: str
    strand: str
    start: int
    end: int
    exons: list[tuple[int, int]]

    @property
    def exon_length(self) -> int:
        return sum(e - s for s, e in self.exons)


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


@lru_cache(maxsize=1)
def gene_regions(path: Path = GTF) -> dict[str, GeneRegion]:
    """Every annotated gene of any type, with its span and the union of its basic exons.

    Read counts that are not split by transcript (whole-gene and binned counts) are modeled
    against these regions, so the union over transcripts is used rather than the
    representative transcript.
    """
    genes: dict[str, GeneRegion] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            # chrY PAR copies share their gene_id prefix with the chrX copy.
            if f[2] not in ("gene", "exon") or f[0] == "chrY":
                continue
            gene_id = _attrs(f[8])["gene_id"].split(".")[0]
            if f[2] == "gene":
                genes[gene_id] = GeneRegion(gene_id, f[0], f[6], int(f[3]) - 1, int(f[4]), [])
            else:
                genes[gene_id].exons.append((int(f[3]) - 1, int(f[4])))
    for g in genes.values():
        g.exons = _merge(g.exons)
    return genes


def _gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")


def mature_sequence(tx: Transcript) -> str:
    seq = "".join(fetch(tx.chrom, s, e) for s, e in tx.exons)
    return seq if tx.strand == "+" else revcomp(seq)


def build_gene_table(gene_types: tuple[str, ...] = ("protein_coding",)) -> pd.DataFrame:
    """Per-gene covariates, coordinates and split, for genes on assigned autosomes."""
    rows = []
    for tx in parse_gtf().values():
        if tx.gene_type not in gene_types or split_of(tx.chrom) is None or not tx.exons:
            continue
        span_start, span_end = tx.exons[0][0], tx.exons[-1][1]
        tss = span_start if tx.strand == "+" else span_end - 1
        introns = [
            (tx.exons[i][1], tx.exons[i + 1][0])
            for i in range(len(tx.exons) - 1)
            if tx.exons[i + 1][0] > tx.exons[i][1]
        ]
        mature = mature_sequence(tx)
        cds_len = sum(e - s for s, e in tx.cds)
        if tx.cds:
            up = sum(max(0, min(e, tx.cds[0][0]) - s) for s, e in tx.exons)
            down = sum(max(0, e - max(s, tx.cds[-1][1])) for s, e in tx.exons)
            utr5_len, utr3_len = (up, down) if tx.strand == "+" else (down, up)
        else:
            utr5_len = utr3_len = 0
        rows.append(
            {
                "gene_id": tx.gene_id,
                "symbol": tx.symbol,
                "chrom": tx.chrom,
                "strand": tx.strand,
                "split": split_of(tx.chrom),
                "tx_id": tx.tx_id,
                "tss": tss,
                "span_start": span_start,
                "span_end": span_end,
                "n_introns": len(introns),
                "intron_length": sum(e - s for s, e in introns),
                "tx_length": len(mature),
                "cds_length": cds_len,
                "utr5_length": utr5_len,
                "utr3_length": utr3_len,
                "gc_transcript": _gc(mature),
                "gc_promoter": _gc(promoter_sequence(tx)),
            }
        )
    return pd.DataFrame(rows).sort_values("gene_id").reset_index(drop=True)


PROMOTER_UP, PROMOTER_DOWN = 1536, 512


def promoter_sequence(tx: Transcript) -> str:
    """2,048 bp from TSS - 1,536 to TSS + 512 on the gene's strand."""
    if tx.strand == "+":
        seq = fetch(tx.chrom, tx.exons[0][0] - PROMOTER_UP, tx.exons[0][0] + PROMOTER_DOWN)
    else:
        seq = revcomp(
            fetch(tx.chrom, tx.exons[-1][1] - PROMOTER_DOWN, tx.exons[-1][1] + PROMOTER_UP)
        )
    return seq


def main() -> None:
    GENES.mkdir(parents=True, exist_ok=True)
    table = build_gene_table()
    table.to_parquet(GENES / "genes.parquet", index=False)
    print(f"{len(table)} genes, splits: {table['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
