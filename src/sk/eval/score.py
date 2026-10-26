"""Gene posteriors and variant deltas from one trained fold model (section 6.3).

A fold model scores only genes and variants on the chromosomes its fold holds out. Variant
deltas are differences of posterior means (alt - ref) in (a, b, c):

- reporter context: satMutMPRA elements in pGL4.23, 3'UTR inserts in the EGFP reporter
  (sk.model.inputs);
- endogenous context: the alternate allele substituted into the input of the gene whose
  representative transcript (3'UTR set) or promoter window (promoter set) contains the variant.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from sk.data.genome import encode, parse_gtf
from sk.data.sources import SATMUT as SATMUT_SCORE_SETS
from sk.data.splits import load_cv_folds
from sk.data.variants import SATMUT
from sk.model import inputs as mi
from sk.paths import PROCESSED

COMPLEMENT = str.maketrans("ACGT", "TGCA")
VARIANTS = PROCESSED / "variants"


@torch.no_grad()
def predict(
    net, codes: np.ndarray, flags: np.ndarray, cov: np.ndarray, device: str, batch: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior means [N, 3] and covariances [N, 3, 3], float64 on CPU."""
    net.eval()
    means, covs = [], []
    x_cov = torch.as_tensor(mi.transform_covariates(np.asarray(cov)))
    for i in range(0, len(codes), batch):
        m, t = net(
            torch.as_tensor(np.asarray(codes[i : i + batch])).to(device),
            torch.as_tensor(np.asarray(flags[i : i + batch])).to(device),
            x_cov[i : i + batch].to(device),
        )
        m, t = m.cpu().double(), t.cpu().double()
        means.append(m.numpy())
        covs.append((t @ t.mT).numpy())
    if not means:
        return np.zeros((0, 3)), np.zeros((0, 3, 3))
    return np.concatenate(means), np.concatenate(covs)


@lru_cache(maxsize=1)
def _genes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "genes" / "genes.parquet")


@lru_cache(maxsize=1)
def _layouts() -> dict[str, mi.GeneLayout]:
    out = {}
    for gid, tx in parse_gtf().items():
        if not tx.exons:
            continue
        tss = tx.exons[0][0] if tx.strand == "+" else tx.exons[-1][1] - 1
        out[gid] = mi.GeneLayout(tx.chrom, tx.strand, tss, list(tx.exons), tx.length)
    return out


def gene_predictions(net, fold: str, device: str) -> pd.DataFrame:
    ids, enc = mi.gene_inputs()
    genes = _genes().set_index("gene_id").reindex(ids)
    rows = np.flatnonzero(genes["chrom"].isin(load_cv_folds()[fold]).to_numpy())
    mean, cov = predict(net, enc.codes[rows], enc.flags[rows], enc.covariates[rows], device)
    return _frame(ids[rows], mean, cov, "gene_id")


def _frame(keys, mean: np.ndarray, cov: np.ndarray, key: str) -> pd.DataFrame:
    df = pd.DataFrame({key: list(keys)})
    for i, r in enumerate("abc"):
        df[f"mean_{r}"] = mean[:, i]
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        df[f"cov_{'abc'[i]}{'abc'[j]}"] = cov[:, i, j]
    return df


def _deltas(keys, ref_mean: np.ndarray, alt_mean: np.ndarray, extra: dict) -> pd.DataFrame:
    d = alt_mean - ref_mean
    df = pd.DataFrame({"key": list(keys), **extra})
    for i, r in enumerate("abc"):
        df[f"delta_{r}"] = d[:, i]
        df[f"ref_{r}"] = ref_mean[:, i]
    return df


# ------------------------------------------------------------------------ promoter set


@lru_cache(maxsize=1)
def _elements() -> dict[str, dict]:
    """Element sequence (plus strand), coordinates and reporter orientation per element id."""
    out = {}
    for name, element_id, _ in SATMUT_SCORE_SETS.values():
        if element_id in out:
            continue
        meta = json.loads((SATMUT / f"{name}.meta.json").read_text())
        coords = meta["extraMetadata"]
        chrom = f"chr{str(coords['chr']).removeprefix('chr')}"
        seq = meta["targetGenes"][0]["targetSequence"]["sequence"].upper()
        start = coords["start"] - 1
        strand = mi.element_orientation(chrom, start, start + len(seq), _genes())
        out[element_id] = {"chrom": chrom, "start": start, "seq": seq, "strand": strand}
    return out


def _with_base(codes: np.ndarray, idx: int, base: str) -> np.ndarray:
    out = codes.copy()
    out[idx] = encode(base)[0]
    return out


def promoter_reporter(net, fold: str, device: str) -> pd.DataFrame:
    v = pd.read_parquet(VARIANTS / "promoter_enhancer.parquet")
    v = v[v["fold"] == fold].drop_duplicates("variant_id")
    frames = []
    for element_id, g in v.groupby("element_id"):
        el = _elements()[element_id]
        seq = mi.oriented(el["seq"], el["strand"])
        codes, flags, cov, first = mi.mpra_element_input(seq)
        alt_codes = []
        for row in g.itertuples():
            off = row.pos0 - el["start"]
            alt = row.alt
            if el["strand"] == "-":
                off, alt = len(seq) - 1 - off, alt.translate(COMPLEMENT)
            assert mi.decode_base(codes[first + off]) == (
                row.ref if el["strand"] == "+" else row.ref.translate(COMPLEMENT)
            )
            alt_codes.append(_with_base(codes, first + off, alt))
        ref_mean, _ = predict(net, codes[None], flags[None], cov[None], device)
        n = len(alt_codes)
        alt_mean, _ = predict(
            net,
            np.stack(alt_codes),
            np.repeat(flags[None], n, 0),
            np.repeat(cov[None], n, 0),
            device,
        )
        frames.append(
            _deltas(
                g["variant_id"],
                np.repeat(ref_mean, n, 0),
                alt_mean,
                {"element_id": element_id, "orientation": el["strand"]},
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def utr_reporter(net, fold: str, device: str) -> pd.DataFrame:
    v = pd.read_parquet(VARIANTS / "utr3.parquet")
    v = v[v["fold"] == fold].drop_duplicates(["variant_id", "insert_ref"])
    if not len(v):
        return pd.DataFrame()
    ref, alt, flags, cov = [], [], [], []
    for row in v.itertuples():
        rc, fl, cv, first = mi.utr_reporter_input(row.insert_ref)
        ref.append(rc)
        alt.append(_with_base(rc, first + row.insert_offset, row.insert_alt[row.insert_offset]))
        flags.append(fl)
        cov.append(cv)
    flags, cov = np.stack(flags), np.stack(cov)
    ref_mean, _ = predict(net, np.stack(ref), flags, cov, device)
    alt_mean, _ = predict(net, np.stack(alt), flags, cov, device)
    return _deltas(v["variant_id"], ref_mean, alt_mean, {"insert_ref": v["insert_ref"].to_numpy()})


# ------------------------------------------------------------------- endogenous context


def _promoter_gene(chrom: str, pos0: int) -> str | None:
    g = _genes()
    g = g[g["chrom"] == chrom]
    rel = np.where(g["strand"] == "+", pos0 - g["tss"], g["tss"] - pos0)
    hit = g[(rel >= -mi.PROMOTER_UP) & (rel < mi.PROMOTER_DOWN)]
    if not len(hit):
        return None
    return hit.iloc[(hit["tss"] - pos0).abs().argmin()]["gene_id"]


def endogenous(net, fold: str, device: str) -> pd.DataFrame:
    ids, enc = mi.gene_inputs()
    row_of = {g: i for i, g in enumerate(ids)}
    layouts = _layouts()
    jobs = []  # (variant_set, variant_id, gene_id, positions, alt base on gene strand)
    p = pd.read_parquet(VARIANTS / "promoter_enhancer.parquet")
    p = p[p["fold"] == fold].drop_duplicates("variant_id")
    for row in p.itertuples():
        gid = _promoter_gene(row.chrom, row.pos0)
        if gid is not None:
            jobs.append(("promoter_enhancer", row.variant_id, gid, row.pos0, row.ref, row.alt))
    u = pd.read_parquet(VARIANTS / "utr3.parquet")
    u = u[(u["fold"] == fold) & u["gene_id"].notna()].drop_duplicates("variant_id")
    for row in u.itertuples():
        jobs.append(("utr3", row.variant_id, row.gene_id, row.pos0, row.ref, row.alt))
    keep, alt_codes = [], []
    for job in jobs:
        _, _, gid, pos0, ref, alt = job
        if gid not in row_of or gid not in layouts:
            continue
        lay = layouts[gid]
        idx = mi.input_positions(lay, pos0)
        if not idx:
            continue
        codes = np.array(enc.codes[row_of[gid]])
        base_ref = ref if lay.strand == "+" else ref.translate(COMPLEMENT)
        base_alt = alt if lay.strand == "+" else alt.translate(COMPLEMENT)
        if any(mi.decode_base(codes[i]) != base_ref for i in idx):
            continue
        for i in idx:
            codes[i] = encode(base_alt)[0]
        keep.append((*job[:3], ",".join(map(str, idx))))
        alt_codes.append(codes)
    if not keep:
        return pd.DataFrame()
    gene_rows = np.array([row_of[k[2]] for k in keep])
    uniq, inv = np.unique(gene_rows, return_inverse=True)
    ref_mean, _ = predict(net, enc.codes[uniq], enc.flags[uniq], enc.covariates[uniq], device)
    alt_mean, _ = predict(
        net, np.stack(alt_codes), enc.flags[gene_rows], enc.covariates[gene_rows], device
    )
    return _deltas(
        [k[1] for k in keep],
        ref_mean[inv],
        alt_mean,
        {
            "variant_set": [k[0] for k in keep],
            "gene_id": [k[2] for k in keep],
            "input_index": [k[3] for k in keep],
        },
    )
