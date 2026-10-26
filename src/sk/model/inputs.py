"""Network inputs (section 6.1): a promoter segment and a transcript segment per sequence.

Each input is 14,336 positions: the promoter segment (2,048 bp, TSS - 1,536 to TSS + 512) then
the transcript segment (mature mRNA, 12,288 nt). Positions carry a base code (0-3 ACGT, 4 N,
5 padding) and flag bits (bit 0: first nucleotide of a codon, bit 1: first nucleotide after an
exon-exon junction). Transcripts longer than 12,288 nt keep their 3' end, so every 3'UTR base
of a scored variant stays in the input; shorter ones are right-padded.

Gene-level covariates (log1p lengths and intron count, GC of each region) are standardized with
the mean and SD over all protein-coding genes, which carry no label information.

Reporter constructs (section 6.3) use public vector sequences in data/raw/vectors:
- satMutMPRA elements: pGL4.23[luc2/minP] (GenBank DQ904455). Promoter segment = element, then
  the vector's minimal promoter up to the transcription start, right-aligned at TSS; then 512 bp
  of the transcribed vector. Transcript = vector from the minimal-promoter TSS through the SV40
  late poly(A) signal (5' leader, luc2 CDS, 3' region).
- 3'UTR inserts: pEGFP-N1 (GenBank U55762) stands in for the MapUTR master plasmid, whose full
  sequence is not deposited. Promoter segment = CMV promoter around its TSS. Transcript =
  5' leader and EGFP CDS, then the 164 nt insert, then the vector 3' region through the SV40
  poly(A) signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from sk.data.genome import PROMOTER_DOWN, PROMOTER_UP, encode, fetch, parse_gtf, revcomp
from sk.paths import PROCESSED, RAW

PROMOTER_LEN = PROMOTER_UP + PROMOTER_DOWN
TRANSCRIPT_LEN = 12_288
INPUT_LEN = PROMOTER_LEN + TRANSCRIPT_LEN
PAD = 5
FRAME, JUNCTION = 1, 2
COVARIATES = (
    "n_introns",
    "intron_length",
    "tx_length",
    "cds_length",
    "utr5_length",
    "utr3_length",
    "gc_promoter",
    "gc_utr5",
    "gc_cds",
    "gc_utr3",
)
LOG_COVARIATES = COVARIATES[:6]
INPUTS = PROCESSED / "inputs"
VECTORS = RAW / "vectors"


@dataclass
class Encoded:
    codes: np.ndarray  # [N, INPUT_LEN] uint8
    flags: np.ndarray  # [N, INPUT_LEN] uint8
    covariates: np.ndarray  # [N, len(COVARIATES)] float32, raw (not standardized)


def _gc(seq: str) -> float:
    acgt = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / acgt if acgt else 0.5


def encode_one(
    promoter: str, transcript: str, cds: tuple[int, int] | None, junctions: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codes, flags and raw covariates of one input.

    promoter has PROMOTER_LEN characters (padding as "-"). cds is the [start, end) of the CDS in
    transcript coordinates; junctions are transcript offsets of the first base of each exon after
    the first.
    """
    if len(promoter) != PROMOTER_LEN:
        raise ValueError("promoter segment has the wrong length")
    codes = np.full(INPUT_LEN, PAD, dtype=np.uint8)
    flags = np.zeros(INPUT_LEN, dtype=np.uint8)
    p = np.frombuffer(promoter.encode(), dtype=np.uint8)
    pcodes = encode(promoter.replace("-", "N"))
    codes[:PROMOTER_LEN] = np.where(p == ord("-"), PAD, pcodes)
    shift = max(len(transcript) - TRANSCRIPT_LEN, 0)
    kept = transcript[shift:]
    codes[PROMOTER_LEN : PROMOTER_LEN + len(kept)] = encode(kept)
    tflags = np.zeros(len(transcript), dtype=np.uint8)
    if cds is not None:
        tflags[np.arange(cds[0], cds[1] - 2, 3)] |= FRAME
    if junctions:
        tflags[np.asarray(junctions)] |= JUNCTION
    flags[PROMOTER_LEN : PROMOTER_LEN + len(kept)] = tflags[shift:]
    utr5 = transcript[: cds[0]] if cds else ""
    cds_seq = transcript[cds[0] : cds[1]] if cds else ""
    utr3 = transcript[cds[1] :] if cds else transcript
    cov = np.array(
        [
            len(junctions),
            0.0,  # intron length, set by the caller
            len(transcript),
            len(cds_seq),
            len(utr5),
            len(utr3),
            _gc(promoter),
            _gc(utr5),
            _gc(cds_seq),
            _gc(utr3),
        ],
        dtype=np.float32,
    )
    return codes, flags, cov


def transcript_offset(length: int) -> int:
    """Index of transcript position 0 in the input (negative when the 5' end is truncated)."""
    return PROMOTER_LEN - max(length - TRANSCRIPT_LEN, 0)


@dataclass
class GeneLayout:
    """Coordinates needed to place an endogenous variant into a gene's input."""

    chrom: str
    strand: str
    tss: int
    exons: list[tuple[int, int]]  # genomic, sorted ascending
    tx_length: int


def gene_input(tx) -> tuple[np.ndarray, np.ndarray, np.ndarray, GeneLayout]:
    from sk.data.genome import mature_sequence, promoter_sequence

    mature = mature_sequence(tx)
    lengths = [e - s for s, e in tx.exons]
    if tx.strand == "-":
        lengths = lengths[::-1]
    junctions = list(np.cumsum(lengths)[:-1])
    cds = None
    if tx.cds:
        cds_len = sum(e - s for s, e in tx.cds)
        if tx.strand == "+":
            up = sum(max(0, min(e, tx.cds[0][0]) - s) for s, e in tx.exons)
        else:
            up = sum(max(0, e - max(s, tx.cds[-1][1])) for s, e in tx.exons)
        cds = (up, up + cds_len)
    codes, flags, cov = encode_one(promoter_sequence(tx), mature, cds, junctions)
    span = tx.exons[-1][1] - tx.exons[0][0]
    cov[1] = span - len(mature)
    tss = tx.exons[0][0] if tx.strand == "+" else tx.exons[-1][1] - 1
    return codes, flags, cov, GeneLayout(tx.chrom, tx.strand, tss, list(tx.exons), len(mature))


def input_positions(layout: GeneLayout, pos0: int) -> list[int]:
    """Input indices that hold genomic position pos0 (promoter and/or transcript segment)."""
    out = []
    rel = pos0 - layout.tss if layout.strand == "+" else layout.tss - pos0
    if -PROMOTER_UP <= rel < PROMOTER_DOWN:
        out.append(PROMOTER_UP + rel)
    t = 0
    exons = layout.exons if layout.strand == "+" else layout.exons[::-1]
    for s, e in exons:
        if s <= pos0 < e:
            off = t + (pos0 - s if layout.strand == "+" else e - 1 - pos0)
            idx = transcript_offset(layout.tx_length) + off
            if idx >= PROMOTER_LEN:
                out.append(idx)
            break
        t += e - s
    return out


def build_gene_inputs() -> None:
    """Encode every protein-coding gene in genes.parquet and cache the arrays."""
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet")
    txs = parse_gtf()
    n = len(genes)
    codes = np.empty((n, INPUT_LEN), dtype=np.uint8)
    flags = np.empty((n, INPUT_LEN), dtype=np.uint8)
    cov = np.empty((n, len(COVARIATES)), dtype=np.float32)
    for i, gid in enumerate(genes["gene_id"]):
        codes[i], flags[i], cov[i], _ = gene_input(txs[gid])
    INPUTS.mkdir(parents=True, exist_ok=True)
    np.save(INPUTS / "codes.npy", codes)
    np.save(INPUTS / "flags.npy", flags)
    np.save(INPUTS / "covariates.npy", cov)
    genes[["gene_id"]].to_parquet(INPUTS / "gene_ids.parquet", index=False)


@lru_cache(maxsize=1)
def gene_inputs() -> tuple[pd.Index, Encoded]:
    ids = pd.Index(pd.read_parquet(INPUTS / "gene_ids.parquet")["gene_id"])
    enc = Encoded(
        np.load(INPUTS / "codes.npy", mmap_mode="r"),
        np.load(INPUTS / "flags.npy", mmap_mode="r"),
        np.load(INPUTS / "covariates.npy"),
    )
    return ids, enc


@lru_cache(maxsize=1)
def covariate_scaler() -> tuple[np.ndarray, np.ndarray]:
    _, enc = gene_inputs()
    x = transform_covariates(enc.covariates, None)
    return x.mean(0), x.std(0) + 1e-6


def transform_covariates(cov: np.ndarray, scaler: tuple | None = "default") -> np.ndarray:
    x = cov.astype(np.float32).copy()
    x[:, : len(LOG_COVARIATES)] = np.log1p(x[:, : len(LOG_COVARIATES)])
    if scaler is None:
        return x
    mu, sd = covariate_scaler() if scaler == "default" else scaler
    return (x - mu) / sd


# ------------------------------------------------------------------- reporter constructs


def _genbank_sequence(name: str) -> str:
    text = (VECTORS / f"{name}.gb").read_text().split("ORIGIN")[1]
    return re.sub(r"[^acgtn]", "", text.lower()).upper()


# 0-based coordinates in each vector, from the GenBank features: pGL4.23 minimal promoter ends at
# 108 and luc2 is 140..1793, SV40 late poly(A) ends at 2049. pEGFP-N1: the CMV transcription
# start (TCAGATC) is at 582, EGFP is 678..1398, and the SV40 early poly(A) signal ends by 1630.
PGL423 = {"tss": 108, "cds": (140, 1793), "end": 2049}
PEGFP = {"tss": 582, "cds": (678, 1398), "end": 1630}


def _left_pad(seq: str, n: int) -> str:
    seq = seq[-n:]
    return "-" * (n - len(seq)) + seq


def _right_pad(seq: str, n: int) -> str:
    seq = seq[:n]
    return seq + "-" * (n - len(seq))


@lru_cache(maxsize=1)
def pgl423() -> str:
    return _genbank_sequence("DQ904455")


@lru_cache(maxsize=1)
def pegfp() -> str:
    return _genbank_sequence("U55762")


def mpra_element_input(element: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Input for an MPRA element in pGL4.23, and the input index of the element's first base.

    element is in the orientation it drives the reporter.
    """
    v = pgl423()
    upstream = element + v[70 : PGL423["tss"]]
    promoter = _left_pad(upstream, PROMOTER_UP) + _right_pad(v[PGL423["tss"] :], PROMOTER_DOWN)
    transcript = v[PGL423["tss"] : PGL423["end"]]
    cds = (PGL423["cds"][0] - PGL423["tss"], PGL423["cds"][1] - PGL423["tss"])
    codes, flags, cov = encode_one(promoter, transcript, cds, [])
    return codes, flags, cov, PROMOTER_UP - len(upstream)


def utr_reporter_input(insert: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Input for a 3'UTR insert in the EGFP reporter, and the input index of its first base."""
    v = pegfp()
    promoter = _left_pad(v[: PEGFP["tss"]], PROMOTER_UP) + _right_pad(
        v[PEGFP["tss"] :], PROMOTER_DOWN
    )
    head = v[PEGFP["tss"] : PEGFP["cds"][1]]
    transcript = head + insert + v[PEGFP["cds"][1] : PEGFP["end"]]
    cds = (PEGFP["cds"][0] - PEGFP["tss"], PEGFP["cds"][1] - PEGFP["tss"])
    codes, flags, cov = encode_one(promoter, transcript, cds, [])
    return codes, flags, cov, transcript_offset(len(transcript)) + len(head)


def element_orientation(chrom: str, start: int, end: int, genes: pd.DataFrame) -> str:
    """Strand of the protein-coding gene whose TSS lies within 1 kb of the element, else '+'."""
    g = genes[
        (genes["chrom"] == chrom) & (genes["tss"] >= start - 1000) & (genes["tss"] < end + 1000)
    ]
    if not len(g):
        return "+"
    mid = (start + end) / 2
    return g.iloc[(g["tss"] - mid).abs().argmin()]["strand"]


def oriented(seq: str, strand: str) -> str:
    return seq if strand == "+" else revcomp(seq)


def fetch_element(chrom: str, start: int, end: int) -> str:
    return fetch(chrom, start, end)


if __name__ == "__main__":
    build_gene_inputs()


def decode_base(code: int) -> str:
    return "ACGTN-"[int(code)]
