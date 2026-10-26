"""Routing variant sets with their reporter context.

Two sets, each with a known mechanism:
- Synthesis side: satMutMPRA saturation mutagenesis of promoters and enhancers assayed in human
  cell lines. Effects are log2 reporter expression of the alternate allele.
- Decay side: MapUTR rare 3' UTR variants (HEK293 and HeLa) and Su et al. disease-database UTR
  variants (HEK293T and SH-SY5Y), whose effects are on mRNA abundance and half-life.

Every variant carries hg38 coordinates, the reference allele verified against hg38, the gene it
falls in, the chromosome split and the cross-validation fold that holds its chromosome out. A
variant is significant when FDR < 0.05 and the absolute log2 effect exceeds 0.2.

`variant_id` identifies a substitution across the measurements of it. In the synthesis-side set,
`element_id` names the assayed element and `source` the score set; the Phase 3 bootstrap
resamples elements, then variants within each element.
"""

from __future__ import annotations

import gzip
import json
import re

import numpy as np
import pandas as pd
from pyliftover import LiftOver

from sk.data.genome import fetch
from sk.data.sources import SATMUT as SATMUT_SCORE_SETS
from sk.data.splits import fold_of, normalize_chrom, split_of
from sk.paths import PROCESSED, RAW
from sk.utils.logging import get_logger

log = get_logger(__name__)

SATMUT = RAW / "satmutmpra"
MAPUTR = RAW / "maputr" / "41467_2024_46795_MOESM4_ESM.xlsx"
SU = RAW / "su2025" / "elife-97682-supp1-v1.xlsx"
CHAIN = RAW / "genome" / "hg19ToHg38.over.chain.gz"
OUT = PROCESSED / "variants"

FDR_SIGNIFICANT = 0.05
FDR_NULL = 0.5
MIN_ABS_LOG2 = 0.2
LN2 = float(np.log(2))
_HGVS = re.compile(r"^n\.(\d+)([ACGT])>([ACGT])$")
# MapUTR oligos carry 164 nt of genomic context centered on the variant; Su et al. assay the
# UTR of the affected transcript. The reporter flank recorded here is MapUTR's.
MAPUTR_FLANK = 164


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.clip(q, 0, 1)
    return pd.Series(out, index=p.index)


def _annotate(df: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    """Add variant id, split, fold, hg38 reference base and the overlapping gene."""
    df = df.copy()
    df["chrom"] = df["chrom"].map(normalize_chrom)
    df["variant_id"] = (
        df["chrom"] + ":" + (df["pos0"] + 1).astype(str) + ":" + df["ref"] + ">" + df["alt"]
    )
    df["split"] = df["chrom"].map(lambda c: split_of(c) or "excluded")
    df["fold"] = df["chrom"].map(lambda c: fold_of(c) or "excluded")
    df["hg38_base"] = [fetch(c, p, p + 1) for c, p in zip(df["chrom"], df["pos0"], strict=True)]
    df["ref_matches_hg38"] = df["hg38_base"] == df["ref"]
    gene_id, gene_strand = [], []
    by_chrom = {c: g.sort_values("span_start") for c, g in genes.groupby("chrom")}
    for chrom, pos in zip(df["chrom"], df["pos0"], strict=True):
        g = by_chrom.get(chrom)
        hit = None if g is None else g[(g["span_start"] <= pos) & (pos < g["span_end"])].head(1)
        gene_id.append(hit["gene_id"].iloc[0] if hit is not None and len(hit) else pd.NA)
        gene_strand.append(hit["strand"].iloc[0] if hit is not None and len(hit) else pd.NA)
    df["gene_id"] = gene_id
    df["gene_strand"] = gene_strand
    return df


def satmut_variants(genes: pd.DataFrame) -> pd.DataFrame:
    """Single-nucleotide substitutions from the satMutMPRA score sets."""
    rows = []
    for name, element_id, cell_line in SATMUT_SCORE_SETS.values():
        path = SATMUT / f"{name}.csv"
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        coords = meta["extraMetadata"]
        element = meta["targetGenes"][0]["targetSequence"]["sequence"]
        start0 = coords["start"] - 1
        if fetch(normalize_chrom(str(coords["chr"])), start0, coords["end"]) != element:
            raise ValueError(f"{path.name}: element sequence does not match hg38")
        df = pd.read_csv(path)
        parsed = df["hgvs_nt"].str.extract(_HGVS)
        keep = parsed[0].notna()
        df, parsed = df[keep], parsed[keep]
        rows.append(
            pd.DataFrame(
                {
                    "variant_set": "promoter_enhancer",
                    "source": name,
                    "element_id": element_id,
                    "cell_line": cell_line,
                    "chrom": str(coords["chr"]),
                    "pos0": start0 + parsed[0].astype(int) - 1,
                    "ref": parsed[1].to_numpy(),
                    "alt": parsed[2].to_numpy(),
                    "element_offset": parsed[0].astype(int).to_numpy() - 1,
                    "log2_effect": df["score"].to_numpy(),
                    "pval": df["p-value"].to_numpy(),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out["fdr"] = out.groupby("source")["pval"].transform(benjamini_hochberg)
    return _annotate(out, genes)


def _lift(df: pd.DataFrame) -> pd.DataFrame:
    """Lift hg19 0-based positions to hg38, dropping variants without a unique mapping."""
    with gzip.open(CHAIN, "rb") as fh:
        lo = LiftOver(fh)
    pos = []
    for chrom, p in zip(df["chrom"], df["pos0"], strict=True):
        hit = lo.convert_coordinate(normalize_chrom(chrom), int(p))
        pos.append((hit[0][0], hit[0][1]) if hit else (None, None))
    df = df.assign(chrom=[c for c, _ in pos], pos0=[p for _, p in pos])
    return df.dropna(subset=["chrom", "pos0"]).astype({"pos0": int})


def maputr_variants(genes: pd.DataFrame) -> pd.DataFrame:
    """Rare 3' UTR variants from MapUTR, both cell lines, lifted from hg19 to hg38."""
    sheets = {"HEK293": "Sheet 1", "HeLa": "Sheet 3"}
    frames = []
    for cell_line, sheet in sheets.items():
        df = pd.read_excel(MAPUTR, sheet_name=sheet, skiprows=3)
        frames.append(
            pd.DataFrame(
                {
                    "variant_set": "utr3",
                    "source": "MapUTR",
                    "cell_line": cell_line,
                    "chrom": df["chromosome"].astype(str),
                    "pos0": df["position (hg19, 1-based)"].astype(int) - 1,
                    "ref": df["ref"].astype(str),
                    "alt": df["alt"].astype(str),
                    "symbol": df["gene"].astype(str),
                    "log2_effect": df["lnFC"].astype(float) / LN2,
                    "pval": df["pval"].astype(float),
                    "fdr": df["FDR"].astype(float),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out = out[out["ref"].isin(list("ACGT")) & out["alt"].isin(list("ACGT"))]
    return _annotate(_lift(out), genes)


def su_variants(genes: pd.DataFrame) -> pd.DataFrame:
    """Disease-database UTR variants from Su et al., scored by change in reporter half-life."""
    df = pd.read_excel(SU, sheet_name="Sup_T2_metaTable")
    # Alleles are reported on the transcript strand, which the variant id's fourth field gives.
    strand = df["Mutant"].astype(str).str.split("_").str[3]
    complement = str.maketrans("ACGT", "TGCA")
    ref = df["ReferenceAllele"].astype(str)
    alt = df["AlternateAllele"].astype(str)
    df = df.assign(
        strand=strand,
        genomic_ref=np.where(strand == "-", [r.translate(complement) for r in ref], ref),
        genomic_alt=np.where(strand == "-", [a.translate(complement) for a in alt], alt),
    )
    frames = []
    for cell_line, suffix in (("SH-SY5Y", "SH"), ("HEK293T", "HEK")):
        wt = pd.to_numeric(df[f"t05_WT_{suffix}"], errors="coerce")
        mt = pd.to_numeric(df[f"t05_mt_{suffix}"], errors="coerce")
        pval = pd.to_numeric(df[f"pval_{suffix}"], errors="coerce")
        part = pd.DataFrame(
            {
                "variant_set": "utr3",
                "source": "Su2025",
                "cell_line": cell_line,
                "chrom": df["Chromosome"].astype(str),
                "pos0": pd.to_numeric(df["Start"], errors="coerce"),
                "ref": df["genomic_ref"],
                "alt": df["genomic_alt"],
                "strand": df["strand"],
                "symbol": df["GeneSymbol"].astype(str),
                "utr_group": df["UTR_Group"].astype(str),
                "log2_effect": np.log2(mt / wt),
                "pval": pval,
            }
        ).dropna(subset=["pos0", "log2_effect", "pval"])
        part["fdr"] = benjamini_hochberg(part["pval"])
        frames.append(part)
    out = pd.concat(frames, ignore_index=True).astype({"pos0": int})
    out = out[out["ref"].isin(list("ACGT")) & out["alt"].isin(list("ACGT"))]
    out = out[out["utr_group"] == "3'UTR"]
    return _annotate(out, genes)


def reporter_context(variants: pd.DataFrame, flank: int = MAPUTR_FLANK) -> pd.DataFrame:
    """Reference and alternate reporter inserts for the 3' UTR sets.

    The insert is the genomic window centered on the variant, on the gene's strand, the length
    MapUTR synthesized. The alternate insert differs at exactly one position.
    """
    half = flank // 2
    ref_seq, alt_seq, offsets = [], [], []
    strand_of = variants.get("strand", variants["gene_strand"]).fillna(variants["gene_strand"])
    for row, strand in zip(variants.itertuples(), strand_of, strict=True):
        start = row.pos0 - half
        seq = fetch(row.chrom, start, start + flank)
        offset = row.pos0 - start
        alt = seq[:offset] + row.alt + seq[offset + 1 :]
        if strand == "-":
            seq, alt = _rc(seq), _rc(alt)
            offset = flank - 1 - offset
        ref_seq.append(seq)
        alt_seq.append(alt)
        offsets.append(offset)
    return variants.assign(insert_ref=ref_seq, insert_alt=alt_seq, insert_offset=offsets)


def _rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def significance(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        significant=(df["fdr"] < FDR_SIGNIFICANT) & (df["log2_effect"].abs() > MIN_ABS_LOG2),
        null_variant=df["fdr"] > FDR_NULL,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet")
    promoter = significance(satmut_variants(genes))
    utr3 = significance(pd.concat([maputr_variants(genes), su_variants(genes)], ignore_index=True))
    dropped = int((~utr3["ref_matches_hg38"]).sum())
    utr3 = reporter_context(utr3[utr3["ref_matches_hg38"]].reset_index(drop=True))
    log.info("dropped %d 3' UTR variants whose reference allele does not match hg38", dropped)
    promoter.to_parquet(OUT / "promoter_enhancer.parquet", index=False)
    utr3.to_parquet(OUT / "utr3.parquet", index=False)
    for name, df in (("promoter_enhancer", promoter), ("utr3", utr3)):
        log.info(
            "%s: %d variants, %d significant, ref matches hg38 for %d",
            name,
            len(df),
            int(df["significant"].sum()),
            int(df["ref_matches_hg38"].sum()),
        )


if __name__ == "__main__":
    main()
