"""Idempotent downloads of every raw source, verified against recorded checksums.

A file's SHA256 is recorded in data/OBSERVED_CHECKSUMS.json on first download and every later
run verifies against it, so a silently changed upstream file fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from sk.paths import RAW, ROOT
from sk.utils.logging import get_logger

log = get_logger(__name__)

CHECKSUMS = ROOT / "data" / "OBSERVED_CHECKSUMS.json"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
SPRINGER = "https://static-content.springer.com/esm"
MAVEDB = "https://api.mavedb.org/api/v1/score-sets"
ENCODE_FILE = "https://www.encodeproject.org/files"
EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=gb&retmode=text"
)
JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/2024/CORE/"
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
)
TARGETSCAN_URL = "https://www.targetscan.org/vert_80/vert_80_data_download/miR_Family_Info.txt.zip"

# satMutMPRA score sets (Kircher et al. 2019) for elements assayed in human cell lines, as
# urn -> (score set name, element id, cell line). A score set is one measurement; an element
# measured at two time points, in two replicates or in two cell lines keeps one element id.
# The TERT GABPA siRNA knockdown and scrambled-control score sets (00000031-c-1, -d-1) are
# perturbation conditions and are not used.
SATMUT = {
    "00000014-a-1": ("BCL11A", "BCL11A", "HEL 92.1.7"),
    "00000016-a-1": ("FOXE1", "FOXE1", "HeLa"),
    "00000017-a-1": ("GP1BB", "GP1BB", "HEL 92.1.7"),
    "00000018-a-1": ("HBB", "HBB", "HEL 92.1.7"),
    "00000019-a-1": ("HBG1", "HBG1", "HEL 92.1.7"),
    "00000020-a-1": ("HNF4A", "HNF4A", "HEK293T"),
    "00000021-a-1": ("IRF4", "IRF4", "SK-MEL-28"),
    "00000022-a-1": ("IRF6", "IRF6", "HaCaT"),
    "00000023-a-1": ("LDLR_rep1", "LDLR", "HepG2"),
    "00000023-a-2": ("LDLR_rep2", "LDLR", "HepG2"),
    "00000024-a-1": ("MSMB", "MSMB", "HEK293T"),
    "00000025-a-1": ("MYC_rs6983267", "MYC_rs6983267", "HEK293T"),
    "00000026-a-1": ("MYC_rs11986220", "MYC_rs11986220", "LNCaP"),
    "00000027-a-1": ("PKLR_24h", "PKLR", "K562"),
    "00000027-b-1": ("PKLR_48h", "PKLR", "K562"),
    "00000029-a-1": ("SORT1_rep1", "SORT1", "HepG2"),
    "00000029-a-2": ("SORT1_rep2", "SORT1", "HepG2"),
    "00000031-a-1": ("TERT_HEK293T", "TERT", "HEK293T"),
    "00000031-b-1": ("TERT_SF7996", "TERT", "SF7996"),
}
# ENCODE K562 total RNA-seq (ENCSR000AEL) RSEM gene quantifications, GENCODE V29.
ENCODE_GENE_QUANT = {"ENCFF136UBW": "rep1", "ENCFF171FQU": "rep2"}


@dataclass(frozen=True)
class Source:
    url: str
    dest: Path
    # Member of a zip archive to extract next to it.
    extract: str | None = None


def all_sources() -> list[Source]:
    s = [
        Source(
            f"{GEO}/GSE75nnn/GSE75792/suppl/GSE75792_{name}",
            RAW / "gse75792" / f"GSE75792_{name}",
        )
        for name in (
            "README.txt",
            "transcript.annotation.gtf.gz",
            "binning_anno.txt.gz",
            "binned_count_table.txt.gz",
        )
    ]
    s += [
        Source(
            f"{GEO}/GSE126nnn/GSE126522/suppl/GSE126522_{name}",
            RAW / "gse126522" / f"GSE126522_{name}",
        )
        for name in ("time_course_slam_CPM.csv.gz", "RNAlevels.csv.gz")
    ]
    s += [
        Source(
            f"{GEO}/GSE207nnn/GSE207924/suppl/GSE207924_{name}",
            RAW / "gse207924" / f"GSE207924_{name}",
        )
        for name in ("Table_S1.xlsx", "processed_data_files.tar.gz")
    ]
    s += [
        Source(
            f"{GEO}/GSE264nnn/GSE264556/suppl/GSE264556_CG20230908_counts_{level}.tsv.gz",
            RAW / "gse264556" / f"GSE264556_counts_{level}.tsv.gz",
        )
        for level in ("exon", "gene")
    ]
    s += [
        Source(
            "https://cdn.elifesciences.org/articles/45396/elife-45396-fig1-data1-v2.xlsx",
            RAW / "wu2019_elife" / "elife-45396-fig1-data1-v2.xlsx",
        ),
        Source(
            f"{SPRINGER}/art%3A10.1038%2Fs41467-024-46795-7/MediaObjects/"
            "41467_2024_46795_MOESM4_ESM.xlsx",
            RAW / "maputr" / "41467_2024_46795_MOESM4_ESM.xlsx",
        ),
        Source(
            "https://cdn.elifesciences.org/articles/97682/elife-97682-supp1-v1.xlsx",
            RAW / "su2025" / "elife-97682-supp1-v1.xlsx",
        ),
        Source(
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
            "gencode.v44.basic.annotation.gtf.gz",
            RAW / "genome" / "gencode.v44.basic.annotation.gtf.gz",
        ),
        Source(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            RAW / "genome" / "hg38.fa.gz",
        ),
        Source(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
            RAW / "genome" / "hg19ToHg38.over.chain.gz",
        ),
    ]
    s += [
        Source(f"{MAVEDB}/urn:mavedb:{urn}/scores", RAW / "satmutmpra" / f"{name}.csv")
        for urn, (name, _, _) in SATMUT.items()
    ]
    s += [
        Source(f"{MAVEDB}/urn:mavedb:{urn}", RAW / "satmutmpra" / f"{name}.meta.json")
        for urn, (name, _, _) in SATMUT.items()
    ]
    s += [
        Source(
            f"{ENCODE_FILE}/{acc}/@@download/{acc}.tsv",
            RAW / "encode" / f"ENCSR000AEL_{rep}_{acc}.tsv",
        )
        for acc, rep in ENCODE_GENE_QUANT.items()
    ]
    # Reporter vectors (pGL4.23[luc2/minP], pEGFP-N1) and the motif baseline's annotations.
    s += [
        Source(f"{EFETCH}&id={acc}", RAW / "vectors" / f"{acc}.gb")
        for acc in ("DQ904455", "U55762")
    ]
    s += [
        Source(JASPAR_URL, RAW / "jaspar" / JASPAR_URL.rsplit("/", 1)[1]),
        Source(
            TARGETSCAN_URL,
            RAW / "targetscan" / "miR_Family_Info.txt.zip",
            extract="miR_Family_Info.txt",
        ),
    ]
    return s


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    part.rename(dest)


def fetch_all() -> None:
    recorded = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    for src in all_sources():
        key = str(src.dest.relative_to(RAW))
        if not src.dest.exists():
            _download(src.url, src.dest)
        digest = sha256_of(src.dest)
        if key in recorded and recorded[key]["sha256"] != digest:
            raise RuntimeError(f"checksum mismatch for {key}: upstream file changed")
        recorded[key] = {"url": src.url, "sha256": digest, "bytes": src.dest.stat().st_size}
        if src.extract and not (src.dest.parent / src.extract).exists():
            with zipfile.ZipFile(src.dest) as z:
                z.extract(src.extract, src.dest.parent)
    CHECKSUMS.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
    log.info("%d sources present and verified", len(recorded))


if __name__ == "__main__":
    fetch_all()
