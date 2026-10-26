"""Motif baseline for the H4 magnitude correlations (project plan, section 4.5).

The baseline routes by region (promoter and enhancer variants -> a, 3'UTR variants -> c), so
only its signed score is used, as the predicted effect on the correct rate. Definitions, fixed
before any comparison with measured effects:

Promoter and enhancer variants (satMutMPRA; hg38 plus strand, ref and alt are single bases):
  - Motifs: JASPAR 2024 CORE vertebrates non-redundant PFMs (MEME bundle). Counts are
    probability x nsites; a pseudocount of 0.8 is split evenly over the four bases (0.2 each).
    PWM = log2((count + 0.2) / (nsites + 0.8) / 0.25), uniform background.
  - Match score: for each variant, allele and motif, the best PWM score over every window of
    the motif's width that contains the variant, on both strands (the reverse-complement PWM on
    the plus strand). delta = best(alt) - best(ref). An N in the hg38 context takes the
    column's lowest score.
  - TF weight: log1p of K562 expression in fragments per kb per million. For genes in
    data/processed/k562/whole_cell_counts.parquet: exonic counts (exon_wc*) divided by each
    sample's library size (counts per million), averaged over samples, divided by the gene's
    union exon length in kb. That table holds only the split autosomes, so every other gene
    (chrX, which carries GATA1, among them) takes the mean FPKM of the two ENCODE K562 RSEM
    quantifications (ENCFF136UBW, ENCFF171FQU), which is the same unit (Spearman 0.90 and
    median log1p difference 0 on shared genes). JASPAR names are upper-cased, a "(var.N)"
    suffix is dropped, and the symbol is looked up among GENCODE v44 gene_name values (any
    gene type; a symbol with several gene_ids takes the largest weight). Dimers A::B take the
    smaller weight of the two partners. A TF whose symbol is not found (fusions such as
    EWSR1-FLI1, names without a human gene) or that has no expression value gets weight 0.
  - Score: the delta of the motif maximizing weight x |delta|, signed, so a gained match
    predicts a positive effect. Motifs with weight 0 (TFs not expressed in K562 or not found)
    can never be selected; when weight x |delta| is 0 for every motif the score is 0.

3'UTR variants (MapUTR and Su et al.; insert_ref and insert_alt, the 164-nt insert in
transcript orientation):
  - ARE: the number of ATTTA occurrences, overlapping allowed.
  - Seed matches: TargetScan 8 miR_Family_Info.txt rows with Species ID 9606 and Family
    Conservation? == 2 (broadly conserved), deduplicated on Seed+m8. With S the reverse
    complement of Seed+m8 in DNA (7 nt, pairing with miRNA nt 8 to 2) and s = S[1:] (nt 7 to
    2), a canonical site is an occurrence of s that is preceded by S[0] (7mer-m8), followed by
    A (7mer-A1), or both (8mer). Each insert position carrying a site for any family counts
    once, so an 8mer is one site and families sharing a 6mer do not double count.
  - Score: -(delta ARE + delta seed sites), delta = alt - ref, so gaining a destabilizing
    element predicts lower abundance.

Outputs: runs/phase3/motif_baseline.parquet (one row per input row, keyed by variant_id,
source, cell_line and the input row number, since some 3'UTR variants are measured more than
once under the same key) and results/phase3/motif_baseline/summary.json with the Spearman rho
between score and log2_effect on significant variants per variant set, per element (promoter
set) and per cell line, and the share of significant variants with a nonzero score.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr

from sk.data.genome import GTF, encode, fetch, gene_regions
from sk.data.sources import JASPAR_URL, TARGETSCAN_URL
from sk.paths import PROCESSED, RAW, RESULTS, RUNS
from sk.utils.logging import get_logger, json_default

log = get_logger(__name__)

JASPAR = RAW / "jaspar" / "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
TARGETSCAN_ZIP = RAW / "targetscan" / "miR_Family_Info.txt.zip"
TARGETSCAN = RAW / "targetscan" / "miR_Family_Info.txt"
VARIANTS = PROCESSED / "variants"
ENCODE = [
    RAW / "encode" / "ENCSR000AEL_rep1_ENCFF136UBW.tsv",
    RAW / "encode" / "ENCSR000AEL_rep2_ENCFF171FQU.tsv",
]
OUT_TABLE = RUNS / "phase3" / "motif_baseline.parquet"
OUT_DIR = RESULTS / "phase3" / "motif_baseline"
PSEUDOCOUNT = 0.8
HUMAN = 9606
BROADLY_CONSERVED = 2
ARE = "ATTTA"
MIN_N_RHO = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_meme(path: Path) -> list[tuple[str, str, np.ndarray]]:
    """(motif id, TF name, log-odds PWM of shape (width, 5) with an N column) per motif."""
    motifs = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("MOTIF"):
            i += 1
            continue
        _, motif_id, name = lines[i].split(maxsplit=2)
        header = lines[i + 1]
        width = int(re.search(r"w= *(\d+)", header).group(1))
        nsites = float(re.search(r"nsites= *(\d+)", header).group(1))
        probs = np.array([lines[i + 2 + j].split() for j in range(width)], dtype=np.float64)
        counts = probs * nsites + PSEUDOCOUNT / 4
        pwm = np.log2(counts / counts.sum(axis=1, keepdims=True) / 0.25)
        motifs.append((motif_id, name.strip(), np.column_stack([pwm, pwm.min(axis=1)])))
        i += 2 + width
    return motifs


def gene_symbols(path: Path = GTF) -> pd.Series:
    """gene_id (unversioned) to gene_name for every gene in the GENCODE annotation."""
    rows = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.split("\t", 8)
            if len(f) == 9 and f[2] == "gene":
                attrs = dict(re.findall(r'(\S+) "([^"]*)"', f[8]))
                rows[attrs["gene_id"].split(".")[0]] = attrs["gene_name"]
    return pd.Series(rows, name="symbol")


def k562_weights() -> pd.Series:
    """log1p of K562 fragments per kb per million, indexed by gene symbol."""
    counts = pd.read_parquet(PROCESSED / "k562" / "whole_cell_counts.parquet")
    counts = counts.set_index("gene_id")[[c for c in counts.columns if "exon_wc" in c]]
    cpm = (counts / counts.sum(axis=0) * 1e6).mean(axis=1)
    regions = gene_regions()
    kb = pd.Series({g: regions[g].exon_length / 1e3 for g in cpm.index if g in regions})
    fpkm = cpm.loc[kb.index] / kb
    reps = []
    for path in ENCODE:
        q = pd.read_csv(path, sep="\t", usecols=["gene_id", "FPKM"])
        q = q[q.gene_id.str.startswith("ENSG")]
        reps.append(q.groupby(q.gene_id.str.split(".").str[0])["FPKM"].sum())
    encode_fpkm = pd.concat(reps, axis=1).mean(axis=1)
    fpkm = pd.concat([fpkm, encode_fpkm.drop(fpkm.index, errors="ignore")])
    symbols = gene_symbols()
    expr = np.log1p(fpkm).rename(index=symbols.to_dict())
    return expr[expr.index.isin(symbols.values)].groupby(level=0).max()


def tf_weight(name: str, by_symbol: pd.Series) -> tuple[float, bool]:
    """Weight of a JASPAR TF name and whether every partner was found."""
    parts = [re.sub(r"\(var\.\d+\)$", "", p).upper() for p in name.split("::")]
    found = all(p in by_symbol.index for p in parts)
    return (min(float(by_symbol.get(p, 0.0)) for p in parts), found)


def promoter_scores(variants: pd.DataFrame, motifs, weights: np.ndarray) -> pd.DataFrame:
    """Score per unique (chrom, pos0, ref, alt): the selected motif's delta and the motif."""
    sites = variants[["chrom", "pos0", "ref", "alt"]].drop_duplicates().reset_index(drop=True)
    flank = max(pwm.shape[0] for _, _, pwm in motifs) - 1
    seqs = [
        fetch(c, p - flank, p + flank + 1) for c, p in zip(sites.chrom, sites.pos0, strict=True)
    ]
    ref = np.stack([encode(s) for s in seqs])
    alt = ref.copy()
    ref[:, flank] = encode("".join(sites.ref))
    alt[:, flank] = encode("".join(sites.alt))
    both = np.concatenate([ref, alt])
    n = len(sites)
    delta = np.empty((n, len(motifs)), dtype=np.float32)
    # Reverse complement of a PWM over ACGTN: reverse positions, swap A<->T and C<->G.
    rc_cols = [3, 2, 1, 0, 4]
    for m, (_, _, pwm) in enumerate(motifs):
        w = pwm.shape[0]
        # Window starts flank - w + 1 .. flank all contain the variant at index flank.
        ctx = both[:, flank - w + 1 : flank + w]
        win = sliding_window_view(ctx, w, axis=1)
        best = np.full(2 * n, -np.inf)
        for p in (pwm, pwm[::-1][:, rc_cols]):
            score = p[np.arange(w), win].sum(axis=2)
            best = np.maximum(best, score.max(axis=1))
        delta[:, m] = best[n:] - best[:n]
    impact = np.abs(delta) * weights[None, :]
    pick = impact.argmax(axis=1)
    chosen = delta[np.arange(n), pick]
    has = impact[np.arange(n), pick] > 0
    sites["baseline_score"] = np.where(has, chosen, 0.0)
    sites["motif"] = np.where(has, np.array([f"{i} {t}" for i, t, _ in motifs])[pick], None)
    return sites


def mirna_seed_table(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Lookup over 6-mer codes: is a seed 6mer, and the bitmask of allowed m8 bases."""
    fam = pd.read_csv(path, sep="\t")
    keep = fam[(fam["Species ID"] == HUMAN) & (fam["Family Conservation?"] == BROADLY_CONSERVED)]
    seeds = keep["Seed+m8"].str.upper().str.replace("U", "T").unique()
    is_seed = np.zeros(4**6, dtype=bool)
    m8_mask = np.zeros(4**6, dtype=np.uint8)
    for seed in seeds:
        site = encode(seed[::-1].translate(str.maketrans("ACGT", "TGCA")))
        code = int(np.dot(site[1:], 4 ** np.arange(5, -1, -1)))
        is_seed[code] = True
        m8_mask[code] |= 1 << int(site[0])
    return is_seed, m8_mask, keep["miR family"].nunique(), len(seeds)


def count_sites(seqs: pd.Series, is_seed: np.ndarray, m8_mask: np.ndarray) -> np.ndarray:
    x = np.stack([encode(s) for s in seqs]).astype(np.int64)
    k6 = sliding_window_view(x, 6, axis=1)
    valid = (k6 < 4).all(axis=2)
    code = np.where(valid, (k6 * 4 ** np.arange(5, -1, -1)).sum(axis=2), 0)
    hit = valid & is_seed[code]
    prev = np.pad(x, ((0, 0), (1, 0)), constant_values=4)[:, : code.shape[1]]
    nxt = np.pad(x, ((0, 0), (0, 1)), constant_values=4)[:, 6 : 6 + code.shape[1]]
    prev_ok = ((m8_mask[code] >> np.minimum(prev, 3)) & 1).astype(bool) & (prev < 4)
    return (hit & (prev_ok | (nxt == 0))).sum(axis=1)


def count_are(seqs: pd.Series) -> np.ndarray:
    return np.array([len(re.findall(f"(?={ARE})", s)) for s in seqs])


def rho(df: pd.DataFrame) -> dict:
    if len(df) < MIN_N_RHO or df.baseline_score.nunique() < 2:
        return {"n": len(df), "rho": None}
    return {"n": len(df), "rho": float(spearmanr(df.baseline_score, df.log2_effect).statistic)}


def correlations(table: pd.DataFrame) -> dict:
    out = {}
    for vset, df in table[table.significant].groupby("variant_set"):
        entry = {
            "all": rho(df),
            "share_nonzero": float((df.baseline_score != 0).mean()),
            "per_cell_line": {k: rho(g) for k, g in df.groupby("cell_line")},
        }
        if vset == "promoter_enhancer":
            entry["per_element"] = {k: rho(g) for k, g in df.groupby("element_id")}
        else:
            entry["per_source"] = {k: rho(g) for k, g in df.groupby("source")}
        out[vset] = entry
    return out


def main() -> None:
    started = datetime.now(UTC)
    t0 = time.monotonic()
    pe = pd.read_parquet(VARIANTS / "promoter_enhancer.parquet")
    utr = pd.read_parquet(VARIANTS / "utr3.parquet")

    motifs = read_meme(JASPAR)
    by_symbol = k562_weights()
    tf = [tf_weight(name, by_symbol) for _, name, _ in motifs]
    weights = np.array([w for w, _ in tf])
    log.info("scanning %d motifs over %d promoter rows", len(motifs), len(pe))
    sites = promoter_scores(pe, motifs, weights)
    pe_out = pe.merge(sites, on=["chrom", "pos0", "ref", "alt"], how="left")
    pe_out["row"] = np.arange(len(pe))

    is_seed, m8_mask, n_families, n_seeds = mirna_seed_table(TARGETSCAN)
    d_are = count_are(utr.insert_alt) - count_are(utr.insert_ref)
    d_seed = count_sites(utr.insert_alt, is_seed, m8_mask) - count_sites(
        utr.insert_ref, is_seed, m8_mask
    )
    utr_out = utr.assign(
        row=np.arange(len(utr)),
        delta_are=d_are,
        delta_seed=d_seed,
        baseline_score=-(d_are + d_seed).astype(np.float64),
    )

    keep = ["variant_set", "row", "variant_id", "source", "cell_line", "significant"]
    keep += ["null_variant", "log2_effect", "fdr", "baseline_score"]
    table = pd.concat(
        [pe_out[keep + ["element_id", "motif"]], utr_out[keep + ["delta_are", "delta_seed"]]],
        ignore_index=True,
    )
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_TABLE, index=False)

    summary = {
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_clock_s": round(time.monotonic() - t0, 1),
        "definitions": __doc__,
        "downloads": {
            "jaspar": {"url": JASPAR_URL, "file": JASPAR.name, "sha256": sha256(JASPAR)},
            "targetscan": {
                "url": TARGETSCAN_URL,
                "file": TARGETSCAN_ZIP.name,
                "sha256": sha256(TARGETSCAN_ZIP),
                "extracted": TARGETSCAN.name,
                "extracted_sha256": sha256(TARGETSCAN),
            },
        },
        "counts": {
            "motifs": len(motifs),
            "motifs_tf_found": int(sum(found for _, found in tf)),
            "motifs_weight_positive": int((weights > 0).sum()),
            "mirna_families_broadly_conserved_human": int(n_families),
            "unique_seed_m8": int(n_seeds),
            "rows": table.groupby("variant_set").size().to_dict(),
            "significant": table[table.significant].groupby("variant_set").size().to_dict(),
            "promoter_unique_sites": len(sites),
        },
        "significant_variants": correlations(table),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n"
    )
    log.info(
        "wrote %s and %s in %.0f s", OUT_TABLE, OUT_DIR / "summary.json", summary["wall_clock_s"]
    )


if __name__ == "__main__":
    main()
