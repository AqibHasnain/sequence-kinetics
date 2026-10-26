# Data sources

Raw files live under `data/raw/` and processed files under `data/processed/`; neither is
committed. `make data` downloads every file idempotently and `data/OBSERVED_CHECKSUMS.json`
records the SHA256 and size of each one, so a later run fails if an upstream file changes. The
SHA256 prefixes below are the first 12 hex characters of those records.

## K562 kinetics

| Source | Accession or URL | Version or date | SHA256 | License | Use |
|---|---|---|---|---|---|
| Subcellular TimeLapse-seq, K562 and 3T3 (Smalec, Ietswaart et al., Mol Cell 2024) | GEO GSE207924, `GSE207924_processed_data_files.tar.gz` and `GSE207924_Table_S1.xlsx` | deposited 2024-03-17, series released 2024-07-03 | `df88acc2d954`, `b81f15a17076` | GEO public | Fraction new per gene at 4sU pulses of 0, 15, 30, 60 and 120 min with 0.05 and 0.95 quantiles (`K562_total_WT_rep1/2.tsv`); whole-cell half-lives from Table S1 |
| Subcellular RNA-seq, K562 (same study) | GEO GSE264556, `CG20230908_counts_exon.tsv.gz` and `counts_gene.tsv.gz` | released 2024-07-03 | `ad35f247edca`, `be6d8df2bf37` | GEO public | Exonic and whole-gene counts for four unperturbed whole-cell replicates; the difference is the intronic count |
| TT-seq and RNA-seq, K562 (Schwalb et al., Science 2016) | GEO GSE75792, `binned_count_table.txt.gz`, `binning_anno.txt.gz` and transcription-unit GTF | released 2015-12-08 (bins), 2016-04-04 (GTF) | `97277fd78b9b`, `8160c40b778b`, `2c7a31a83d81` | GEO public | Strand-specific fragment counts in 200 bp hg38 bins for TT-seq (5 min 4sU pulse), 4sU-seq and total RNA, summed per gene over exonic and intronic bins; published synthesis and decay rates per transcription unit (GTF) |
| SLAM-seq mRNA decay, K562 (Wu et al., eLife 2019) | eLife 45396 Figure 1 source data 1 (`elife-45396-fig1-data1-v2.xlsx`), sheet `k562-SLAM-seq`; raw data GEO GSE126522 | published 2019-04-23 | `f5ad19c5fd62` | CC BY 4.0 | Published per-gene SLAM-seq decay rates, the second half-life assay for Gate G0 |
| SLAM-seq chase counts, K562 (same study) | GEO GSE126522, `time_course_slam_CPM.csv.gz`, `RNAlevels.csv.gz` | released 2019-05-01 | `03e24c7fab41`, `585c0b49667d` | GEO public | Chase CPM at 0, 2, 4 and 6 h, held for the Phase 1 noise model |
| ENCODE K562 total RNA-seq | ENCSR000AEL, gene quantifications `ENCFF136UBW` (rep 1) and `ENCFF171FQU` (rep 2), GENCODE V29 | released 2014-06-30 | `e5b75b2c844e`, `9badf422985d` | ENCODE terms of use | Additional steady-state exonic levels |

## Variants with known mechanism

| Source | Accession or URL | Version or date | SHA256 | License | Use |
|---|---|---|---|---|---|
| satMutMPRA saturation mutagenesis (Kircher et al., Nat Commun 2019) | MaveDB score sets, one per measurement, listed below | MaveDB API, retrieved 2026-10-25 | below | CC0 | Synthesis-side routing set: log2 reporter effect and p-value per single-nucleotide substitution, with the assayed element sequence and its hg38 coordinates |
| MapUTR rare 3' UTR variants (Fu et al., Nat Commun 2024) | Supplementary Data 1, `41467_2024_46795_MOESM4_ESM.xlsx` | published 2024-04-18 | `d2bdcafb68ae` | CC BY 4.0 | Decay-side routing set: lnFC, p-value and FDR per variant in HEK293 and HeLa, hg19 coordinates lifted to hg38 |
| Disease-database UTR variants (Su et al., eLife 2025, 97682) | Supplementary file 1, `elife-97682-supp1-v1.xlsx`, sheet `Sup_T2_metaTable` | published 2025-02-18 | `7339f268a965` | CC BY 4.0 | Decay-side routing set: reporter half-lives of reference and mutant UTRs in HEK293T and SH-SY5Y, hg38 coordinates, alleles on the transcript strand |

### satMutMPRA score sets

Each score set is `https://api.mavedb.org/api/v1/score-sets/<URN>/scores` with its metadata at
`.../score-sets/<URN>`. All are MaveDB version 1 of the score set, published 2019-02-19 or
2019-02-20, license CC0; the metadata also carries a pre-publication data-use statement that
lapsed with the 2019 paper. Coordinates are hg38, 1-based inclusive, and every element sequence
equals the hg38 plus strand at them. SHA256 prefixes are for the scores CSV; the metadata JSON
checksums are in `data/OBSERVED_CHECKSUMS.json`.

| URN | Score set | Element | Cell line | hg38 | SHA256 |
|---|---|---|---|---|---|
| `urn:mavedb:00000014-a-1` | BCL11A | BCL11A enhancer | HEL 92.1.7 | chr2:60494940-60495539 | `ded6f47b3abb` |
| `urn:mavedb:00000016-a-1` | FOXE1 | FOXE1 promoter | HeLa | chr9:97853255-97853854 | `655520e382ea` |
| `urn:mavedb:00000017-a-1` | GP1BB | GP1BB promoter | HEL 92.1.7 | chr22:19723266-19723650 | `407e14088f2b` |
| `urn:mavedb:00000018-a-1` | HBB | HBB promoter | HEL 92.1.7 | chr11:5227022-5227208 | `ec8b189f70c7` |
| `urn:mavedb:00000019-a-1` | HBG1 | HBG1 promoter | HEL 92.1.7 | chr11:5249805-5250078 | `8a9e1d4c1ff1` |
| `urn:mavedb:00000020-a-1` | HNF4A | HNF4A promoter | HEK293T | chr20:44355520-44355804 | `6dc2555a40aa` |
| `urn:mavedb:00000021-a-1` | IRF4 | IRF4 enhancer | SK-MEL-28 | chr6:396143-396593 | `a94694a90445` |
| `urn:mavedb:00000022-a-1` | IRF6 | IRF6 enhancer | HaCaT | chr1:209815790-209816390 | `b8e0665dea04` |
| `urn:mavedb:00000023-a-1` | LDLR_rep1 | LDLR promoter | HepG2 | chr19:11089231-11089548 | `5900bba4f317` |
| `urn:mavedb:00000023-a-2` | LDLR_rep2 | LDLR promoter | HepG2 | chr19:11089231-11089548 | `eb7fed22c8e3` |
| `urn:mavedb:00000024-a-1` | MSMB | MSMB promoter | HEK293T | chr10:46046244-46046834 | `e3cf661f6be8` |
| `urn:mavedb:00000025-a-1` | MYC_rs6983267 | MYC enhancer (rs6983267) | HEK293T | chr8:127400829-127401428 | `56e62b141751` |
| `urn:mavedb:00000026-a-1` | MYC_rs11986220 | MYC enhancer (rs11986220) | LNCaP, 100 nM DHT | chr8:127519270-127519732 | `e4ca0cef8ea3` |
| `urn:mavedb:00000027-a-1` | PKLR_24h | PKLR promoter, 24 h | K562 | chr1:155301395-155301864 | `5b340753929f` |
| `urn:mavedb:00000027-b-1` | PKLR_48h | PKLR promoter, 48 h | K562 | chr1:155301395-155301864 | `c2df8c0d1e60` |
| `urn:mavedb:00000029-a-1` | SORT1_rep1 | SORT1 enhancer | HepG2 | chr1:109274652-109275251 | `c22f993fcc4f` |
| `urn:mavedb:00000029-a-2` | SORT1_rep2 | SORT1 enhancer | HepG2 | chr1:109274652-109275251 | `6fe848943765` |
| `urn:mavedb:00000031-a-1` | TERT_HEK293T | TERT promoter | HEK293T | chr5:1294989-1295247 | `320f9e1d9446` |
| `urn:mavedb:00000031-b-1` | TERT_SF7996 | TERT promoter | SF7996 (glioblastoma) | chr5:1294989-1295247 | `8b0d9fb33657` |

Not used: F9 promoter (`00000015-a-1`, chrX), the SORT1 flipped-orientation score set
(`00000029-b-1`), the TERT GABPA siRNA knockdown and scrambled-control score sets
(`00000031-c-1`, `00000031-d-1`), and the elements assayed in mouse cells.

## Reference

| Source | Accession or URL | Version or date | SHA256 | License | Use |
|---|---|---|---|---|---|
| GENCODE basic annotation | `ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.basic.annotation.gtf.gz` | release 44, 2023-07-17 | `3e52f82c63f8` | CC BY (GENCODE) | Gene models, principal transcripts, promoter windows, covariates |
| hg38 reference | `hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz` | UCSC | `c1dd87068c25` | UCSC, free for all uses | Sequences and reference-allele checks |
| hg19 to hg38 chain | `hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz` | UCSC | `5c0598e500ce` | UCSC, free for all uses | Lifting MapUTR coordinates |

## Reporter vectors and baseline annotations (Phases 2 and 3)

The GenBank records are NCBI efetch output,
`eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=gb&retmode=text&id=<accession>`.
`make data` extracts `miR_Family_Info.txt` from the TargetScan archive.

| Source | Accession or URL | Version or date | SHA256 | License | Use |
|---|---|---|---|---|---|
| pGL4.23[luc2/minP] | GenBank DQ904455 (efetch, GenBank flat file) | 2008-10-07 | `68abd1899f4a` | public record | satMutMPRA reporter construct: minimal promoter, luc2 transcript |
| pEGFP-N1 | GenBank U55762 (efetch, GenBank flat file) | 2003-08-22 | `bf6a633fce38` | public record | Stand-in for the MapUTR EGFP reporter: CMV promoter, EGFP transcript, SV40 poly(A) |
| JASPAR 2024 CORE vertebrates non-redundant | `jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt` | 2024 release | `dd494278d356` | CC BY 4.0 | Motif baseline, promoter variants |
| TargetScan 8 miRNA families | `www.targetscan.org/vert_80/vert_80_data_download/miR_Family_Info.txt.zip` | release 8.0 | `d91827b9c952` (zip) | TargetScan terms | Motif baseline, 3'UTR seed matches |

The motif baseline also uses the ENCODE K562 RSEM files listed above for TF expression of genes
outside the split autosomes.

## Sources resolved but not downloaded

- Agarwal et al., Nature 639:411-420 (2025) lentiMPRA. The K562 data on the ENCODE portal
  (ENCSR460LZI pilot, ENCSR382BVV large-scale, ENCSR203UFY joint) are element activity
  quantifications, not allelic pairs; the paper's variant effects are MPRALegNet predictions.
  The plan's fallback, satMutMPRA, is used instead (`docs/DECISIONS.md`, 2026-10-25).
- Saluki weights: Zenodo record 10.5281/zenodo.6326409, `datasets.zip` (18.75 GB), models under
  `datasets/deeplearning/train_gru/f*_c*/train/model{0,1}_best.h5` (about 2 MB each). The
  archive is downloaded selectively in Phase 3, when the baseline is scored.
- GSE75220, 4sU-labeled transcripts in 65 YRI LCLs, with normalized and quantile-normalized
  expression tables, for the conditional Phase 4 replication.
