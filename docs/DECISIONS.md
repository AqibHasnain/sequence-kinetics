# Decision log

Append-only, newest last. Each entry records what was decided, when, and why, written at the time
of the decision. Any entry that follows a seen result says so.

## 2026-10-25 - Subcellular TimeLapse-seq accession resolved to GSE207924

The plan left the Smalec, Ietswaart et al. (Mol Cell 2024) accession unverified. The study is a
SuperSeries family; GSE207924 ("[RNA-seq]", 160 samples) holds the subcellular TimeLapse-seq in
K562 and 3T3, including `K562_total_WT_rep{1,2}` at 4sU pulses of 0, 15, 30, 60 and 120 min, and
Table S1 with per-gene half-lives for chromatin release, nuclear export, cytoplasmic decay and
whole cell. Verified by reading the GEO sample titles and the deposited tables. The two
replicates label their time columns with different prefixes (T and U), which the loader reads
from the header.

The archive `GSE207924_processed_data_files.tar.gz` is gzipped twice. `sk.data.k562` peels both
layers, since Python's tarfile unwraps one.

## 2026-10-25 - Steady-state counts come from GSE264556

GSE75792 reports counts per transcription unit, not per gene, and its deposited GTF carries no
gene identifier, so units must be mapped to genes by strand-matched overlap. One unit often
covers several genes, and the argmax assignment reaches only 9,446 genes, of which 2,533 clear
the Gate G0 read-depth bar. GSE264556 (same Churchman study as the TimeLapse-seq) deposits
per-gene exonic counts and whole-gene counts for four unperturbed K562 whole-cell replicates;
the difference is the intronic count. Those four replicates are the total RNA-seq samples for
Gate G0 and for the steady-state likelihood, and 7,207 genes clear the bar.

GSE75792 is kept for the TT-seq 5 min labeled libraries and its published synthesis and decay
rates. Inferred, not stated in the deposited README: in its file names L is 4sU-labeled RNA and
T total RNA, and "with" and "none" are with and without fragmentation, so L_with is TT-seq,
L_none 4sU-seq, T_with Total RNA-Frag-seq and T_none total RNA-seq. The GEO sample titles list
exactly these four library types.

## 2026-10-25 - ENCODE K562 total RNA-seq experiment: ENCSR000AEL

Stranded, rRNA-depleted total RNA > 200 nt, two isogenic replicates, GRCh38, GENCODE V29 gene
quantifications (ENCFF136UBW, ENCFF171FQU). The other released K562 total RNA-seq experiments
are subcellular fractions (CSHL, IRCM), drug-treated (Greenleaf) or unreplicated. RSEM
quantifications give exonic levels only, so this experiment supports the steady-state exonic
term and not the intronic one.

## 2026-10-25 - Agarwal 2025 lentiMPRA has no measured allelic table; satMutMPRA is used

The plan's primary synthesis-side source assumed the Agarwal et al. lentiMPRA included measured
allelic variant effects. Its data availability section lists ENCODE accessions (K562:
ENCSR460LZI, ENCSR382BVV, ENCSR203UFY), and every released processed file there is an element
activity quantification. The variant effects in the paper are MPRALegNet predictions, not
measurements, so they cannot serve as ground truth for routing. IGVF analysis set IGVFDS3278AFYF
is a HepG2 lentiMPRA set and also holds no variant table.

The plan's fallback is used: satMutMPRA saturation mutagenesis (Kircher et al., Nat Commun
2019), restricted to elements assayed in K562 or HepG2, taken from MaveDB (CC0) where each score
set carries the assayed element sequence and its hg38 coordinates. Elements used: PKLR promoter
in K562 (24 h and 48 h), LDLR promoter and SORT1 enhancer in HepG2 (two replicates each). The F9
promoter is on chrX and excluded. The cell-line mismatch for the HepG2 elements is the same
limitation the plan already records for the 3' UTR sets, and the two cell lines are reported
separately.

## 2026-10-25 - MapUTR and Su et al. effect tables

MapUTR effects are Supplementary Data 1 of Fu et al. (`41467_2024_46795_MOESM4_ESM.xlsx`):
17,214 rare variants in HEK293 (sheet 1) and 17,198 in HeLa (sheet 3), with lnFC, p-value and
FDR, in hg19 coordinates, lifted to hg38 here. BioProject 972741 holds only raw reads.

Su et al. effects are Supplementary file 1 of eLife 97682, sheet `Sup_T2_metaTable`: reporter
half-lives of reference and mutant UTRs in SH-SY5Y and HEK293T with per-variant p-values, in
hg38 coordinates. The effect is log2 of the mutant over the reference half-life and the FDR is
Benjamini-Hochberg over each cell line. Verified by checking alleles against hg38: the alleles
are reported on the transcript strand, which the fourth field of the variant id gives, so they
are complemented to genomic orientation before use. 89 of 37,836 variants have a reference
allele that does not match hg38 and are dropped.

Griesemer et al. (Cell 2021) was not needed and is not used.

## 2026-10-25 - Saluki weights location

Zenodo record 10.5281/zenodo.6326409 ("Saluki: Predicting mRNA half-life from mRNA sequence",
CC BY 4.0), one 18.75 GB `datasets.zip`. The released models are
`datasets/deeplearning/train_gru/f{fold}_c{cross}/train/model{0,1}_best.h5`, about 2 MB each,
verified by listing the archive's central directory over HTTP range requests. The Keras weights
are fetched selectively in Phase 3 rather than downloading the archive.

## 2026-10-25 - GSE126521 holds no endogenous K562 time points

GSE126521 is the ORFome (exogenous reporter) decay series: its 40 samples are 293T and K562
cells infected with the ORFome library and chased with actinomycin D. Endogenous K562 decay is
its sibling series GSE126522, SLAM-seq at 0, 2, 4 and 6 h in triplicate. Gate G0 uses the
published per-gene SLAM-seq decay rates from the paper's Figure 1 source data, as the plan
prefers published rate tables over re-processing; the deposited chase CPM tables are kept for
the Phase 1 noise model.

## 2026-10-25 - LCL total RNA series for the conditional Phase 4

Li et al. (Science 2016) do not name their RNA-seq series in the text. Figure 1A labels the two
RNA-seq sources G and P and cites Lappalainen et al. 2013 (GEUVADIS, ArrayExpress E-GEUV-1, 89
YRI LCLs) and Pickrell et al. 2010 (GSE19480, 69 YRI LCLs). This is inferred from the citations,
not stated as an accession. GEUVADIS is the larger overlap with the 65 4sU lines and would be
the total RNA source; the choice is confirmed against sample identifiers before any Phase 4 run,
which happens only if Gate G2 passes for the LCL set. Nothing from this source is downloaded in
Phase 0.

## 2026-10-25 - Routing uses 5-fold chromosome cross-validation

Decided from the Phase 0 counts, before any training. The synthesis-side set is entirely on
training chromosomes: PKLR is on chr1, SORT1 on chr1 and LDLR on chr19, so the fixed test split
(chr2, chr8, chr13) leaves zero scoreable promoter or enhancer variants against the 150 Gate G2
requires. The 3' UTR set has 570 significant variants on test chromosomes, enough on its own,
but both arms of H2 need the same scoring regime.

`configs/splits.yaml` therefore fixes five chromosome folds, balanced by gene count
(fold0 chr1, chr5, chr8, chr18; fold1 chr7, chr10, chr13, chr19, chr22; fold2 chr4, chr11,
chr12, chr15; fold3 chr2, chr6, chr9, chr14; fold4 chr3, chr16, chr17, chr20, chr21). Every
variant is scored by the fold model whose training chromosomes exclude the variant's
chromosome. The primary chr7 and chr2/chr8/chr13 split stays in place for gene-level metrics.

## 2026-10-25 - Gene model: GENCODE v44 basic, protein-coding, one transcript per gene

The representative transcript is the APPRIS principal transcript of the lowest rank, and among
ties, and for genes without an APPRIS tag, the longest basic transcript. 19,111 protein-coding
genes on autosomes result: 15,909 train, 930 validation, 2,272 test.

## 2026-10-25 - Gate G0 passes

Criterion 1: 7,207 genes have exonic and intronic counts above 10 in all four GSE264556
whole-cell replicates and fraction-new data at two or more TimeLapse-seq pulse times, against a
threshold of 5,000. Criterion 2: whole-cell half-lives from subcellular TimeLapse-seq and decay
rates from SLAM-seq agree at Spearman rho 0.611 over 8,738 shared genes, against a threshold of
0.5; restricted to the criterion-1 genes, rho is 0.615 over 6,230 genes. Numbers in
`results/phase0/gate_g0.json`.

## 2026-10-25 - Scope of the Phase 0 reporter-construct test

The plan's test "reporter constructs reproduce the published vector sequence around the
insertion site" is implemented as a check on the insert and its coordinates, not on the plasmid
backbone. For satMutMPRA the MaveDB score sets carry the assayed element sequence and its hg38
coordinates; the test asserts the element equals hg38 at those coordinates and that every
variant's offset into the element carries the reported reference allele, which is what fixes the
position of every variant in the set. For the 3' UTR sets the insert is the 164 nt genomic
window MapUTR synthesized around the variant, and the test asserts the alternate insert differs
from the reference at exactly the recorded position. The vector-side flanks (pGL4.11b and
pGL4.23 for satMutMPRA, the MapUTR master plasmid between EcoRI and BamHI downstream of EGFP)
are added in Phase 2, where the reporter mRNA input is assembled.

Promoter and transcript sequences are not materialized to disk. `sk.data.genome` builds them
from hg38 and the gene model on demand, which keeps the processed tables small and keeps one
definition of the promoter window and the mature transcript.

## 2026-10-25 - GSE75792 per-gene counts rebuilt from the 200 bp bins

Found at the start of Phase 1, before any fit. The deposited per-transcription-unit count
tables (`transcript_counts_exons/introns.txt.gz`) are keyed by numeric row names (1 to 86,674)
that match neither the row order of the deposited annotation nor each other: the exon and the
intron tables list their rows in different orders. Phase 0 read them by position, so
`schwalb_counts.parquet` assigned counts to the wrong genes. Verified: its total-RNA exonic
counts correlate with the GSE264556 whole-cell exonic counts at Spearman rho -0.02, and no tested
re-indexing (sorted keys, keys modulo the unit count) recovers a correlation. Gate G0 did not use
this table, so the gate result stands.

Per-gene counts are rebuilt from the deposited genome-wide binned table (`binned_count_table`,
`binning_anno`, hg38, 200 bp bins, one plus- and one minus-strand column per library). A bin
belongs to a protein-coding gene when that gene's span is the only annotated span on the strand
that covers it; it is exonic when it overlaps any GENCODE v44 basic exon of the gene, intronic
otherwise. The rebuilt total-RNA counts agree with the GSE264556 whole-cell counts at Spearman
rho 0.89 (exonic) and 0.89 (intronic) over 17,000 shared genes, and the TT-seq to total ratio
of exonic bins correlates with the TimeLapse-seq decay rate at rho 0.29. The table is
`ttseq_counts.parquet`; the per-unit tables are dropped from the download list.

## 2026-10-25 - Phase 1 likelihood: what each assay contributes

Rates are per hour. The fits use the 5,918 training-chromosome genes that meet Gate G0
criterion 1 and have annotated intronic sequence (one G0 training gene has none). Exonic and
intronic lengths are the union of the gene's GENCODE v44 basic exons and the rest of its span.

- **Whole-cell steady state (GSE264556):** negative binomial on exonic and intronic counts, one
  size factor per replicate and one dispersion per region. Exonic reads come from both pools
  (pre-mRNA carries exons), so the exonic mean is s L_e (u* + m*). Pre-mRNA and mature RNA are
  assumed to yield the same reads per nucleotide; without that assumption, or a spike-in, the
  location of b is not fixed.
- **TimeLapse-seq (GSE207924):** the plan's per-gene fallback (no per-read conversion data),
  with two changes. Observations are Gaussian in probability space, not logit space: the
  short-pulse fractions of slow genes sit near 0, where the deposited quantiles are far from
  Gaussian in logit space (median logit-scale SD 1.13 at 0 min against 0.15 at 120 min). The
  0 min fraction is not 0 (median 0.018) and is gene-specific (rho 0.69 between replicates), so
  each gene's mean 0 min fraction is a plug-in background bg and the 15 to 120 min observations
  are bg + (1 - bg) h(t). The deposited SD is scaled by a fitted kappa and an extra SD tau is
  added.
- **TimeLapse-seq reads are modeled as exonic reads.** This follows a seen result. With the
  fraction new modeled over the whole gene (introns included), the KIN fit over-predicts
  whole-cell exonic counts by a median factor of 1.9, the fitted exonic dispersion collapses to
  a negative binomial size of 1.0, and the median half-life is 13 h. Modeled over exonic reads,
  the exonic residuals are centered (median -0.01 in log), the exonic size is 43 (as under
  SS-EI, 43), the ELBO improves by 7.2 nats per gene, and the median half-life is 3.9 h, close
  to the 3.6 h median of the deposited whole-cell half-lives. The deposited files do not state
  which reads were counted (the methods text could not be retrieved), so this is inferred.
- **TT-seq (GSE75792, 5 min):** negative binomial on labeled counts in each gene's exonic and
  intronic bins, with mean k_j T (h + eps (1 - h)), where T is the gene's normalized
  fragmented-total-RNA count in the same bins. Conditioning on T cancels bin coverage and
  mappability, which the bins do not report. k_j (one per replicate) and eps are global.
- **SLAM-seq (GSE126522) is not in the KIN likelihood.** The deposited chase table holds T-to-C
  read CPM without matched totals or conversion efficiency. SLAM-seq half-lives remain the
  held-out assay of section 8.1.
- **KIN-1T** reuses the TT-seq scale k_j and contamination eps fitted under KIN, standing in for
  the spike-in normalization of the original study. With these free, a single 5 min pulse
  leaves the scale k nearly confounded with a common shift of b (f_u is close to linear in beta
  at 5 min).

## 2026-10-25 - Phase 1 fitting, gauge and simulation choices

- **Gauge:** after fitting, every posterior-mean a, the prior mean of a and the log size
  factors are shifted together so the median posterior-mean a over training genes is 0, which
  leaves the likelihood unchanged.
- **Initialization:** every gene starts at c = log(ln 2 / 4 h), with a - c and a - b from its
  whole-cell counts, under every design, so no design starts with gene-specific decay
  information from another assay.
- **Unidentified prior components under SS-E and SS-EI:** the empirical-Bayes objective does not
  depend on the prior along directions the design does not identify, so those components stay
  near their initial values and are not estimates. The SS fits are reported for the identified
  components only. Phase 2 must choose how the SS arms' priors are set along these directions.
- **SBC and H1 simulate from the KIN fit** (prior and noise), the only design that identifies
  every prior component, and every design's per-gene posterior is fitted under that same prior
  and noise model, as SBC requires. Covariates (lengths, deposited TimeLapse SDs, background,
  missingness) are resampled from real training genes; the TT-seq total-RNA count is the KIN
  fit's median depth per unit of level times the simulated level.

## 2026-10-25 - Observable range and H1 statistics, fixed before SBC

- **Observable range:** half-lives at which the Laplace posterior variance of c is at most 10% of
  its prior variance (contraction >= 0.9), from the Fisher information of the median training
  gene (a = 0, median covariates, b at its KIN-prior conditional mean given c), over a grid from
  3 min to 200 h. The longest contiguous run is the range.
- **H1, first part:** for simulated genes whose true half-life is inside the KIN observable
  range, the per-gene ratio of the 90% interval width of c under SS-EI to that under KIN; H1
  requires the median ratio >= 3.
- **H1, second part:** for all simulated genes, the per-gene ratio of the posterior SD of
  (a + b + c)/3 under SS-EI to its prior SD; H1 requires the median ratio > 0.8.

## 2026-10-25 - Inference fix 1: full-covariance per-gene posteriors

Follows a seen result. With diagonal Gaussian posteriors, as section 6.2 specifies, SBC fails
for every rate and combination under every design (`results/phase1/sbc_diag.json`): KIN 90%
coverage is 0.54, 0.62 and 0.70 for a, b and c, every chi-square p < 0.001 except KIN a - c,
and under SS-EI the posterior SD of (a + b + c)/3 is 0.07 of its prior SD. The cause is the
known mean-field failure on correlated posteriors: SS designs pin a - b and a - c and leave a
ridge along (1, 1, 1), and under KIN a is identified only through a - c and c, so the
posteriors are strongly correlated and a factorized Gaussian puts its variance across the
ridge instead of along it. The fix is a full-covariance Gaussian per gene (a 3 x 3 Cholesky
factor, six scale parameters instead of three). Pass criteria are unchanged. With it, every
Gate G1 quantity passes and the diagonal results are kept as the record of the failure. The
amortized heads of Phase 2 face the same correlations, so Phase 2 should output a
full-covariance posterior too (section 13 lists this as an extension; it is needed for
calibration).

## 2026-10-25 - Gate G1 passes; H1 holds

SBC with full-covariance posteriors, 1,000 genes simulated from the KIN fit, 99 posterior draws,
20 bins (`results/phase1/sbc_full.json`). Gate G1 quantities, chi-square p and 90% coverage:
KIN a 0.85 / 0.888, b 0.47 / 0.886, c 0.18 / 0.885; SS-EI a - b 0.36 / 0.896, a - c 0.25 /
0.892. KIN-1T passes for every quantity, and SS-EI passes for a, b and c individually too.
SS-E fails for b (p 0.008), a - b (p 4e-9, coverage 0.836) and a - c (p 1.5e-12, coverage
0.852), with excess ranks at both ends: under SS-E the exonic count fixes log(exp(a - b) +
exp(a - c)), a curved ridge a Gaussian cannot follow. SS-E is not part of the gate; Phase 2
reports SS-E uncertainty with that caveat.

Observable half-life range (contraction of c >= 0.9, median gene): KIN 0.19 h to 13.5 h. SS-E,
SS-EI and KIN-1T reach no such range: their largest contraction of c is 0.21, 0.37 and 0.79.
H1 part 1: median ratio of 90% interval widths of c, SS-EI over KIN, for the 931 simulated genes
inside the KIN range is 5.7 (quartiles 3.7, 9.1; 85% of genes >= 3). H1 part 2: median ratio of
posterior to prior SD of (a + b + c)/3 under SS-EI is 0.905 (> 0.8). The ratio is below 1
because the fitted prior correlates the rates, so a - b and a - c carry some information about
the mean log rate.

## 2026-10-25 - Promoter and enhancer set expanded to all human satMutMPRA elements

Section 8.3 clusters reporter variants by MPRA element, and three elements give three bootstrap
clusters, which cannot support an interval. MaveDB holds 15 human satMutMPRA elements assayed in
human cell lines, and all are used: BCL11A, GP1BB, HBB, HBG1 (HEL), FOXE1 (HeLa), HNF4A, MSMB,
MYC rs6983267 (HEK293T), MYC rs11986220 (LNCaP), IRF4 (SK-MEL-28), IRF6 (HaCaT), LDLR, SORT1
(HepG2), PKLR (K562), TERT (HEK293T and SF7996). F9 (chrX) and the elements assayed in mouse
cells (RET, UC88, ZRS, TCF7L2, ZFAND3) are excluded. The catalogue was checked against the MaveDB
API; chromosome placement is confirmed from MaveDB coordinates at load.

The bootstrap for this set resamples elements, then variants within each resampled element.
Tables report element and variant counts together, and routing accuracy is reported per element
and pooled. H2 thresholds and the equal weighting of the two variant sets are unchanged; an
element-weighted accuracy is reported as a secondary number.

Siraj et al. (bioRxiv 2024.05.05.592437, K562 among five lines, 37,284 allele-specific variants)
would give more K562 loci, but its per-variant tables are not deposited.

## 2026-10-25 - Promoter and enhancer elements from cell lines other than K562 are kept

H2 asks which rate a variant acts on, and a promoter or enhancer reporter variant acts on
synthesis by the design of the assay. The cell-line mismatch caps the H4 magnitude correlation,
the same limitation section 4.3 records for the 3'UTR sets. Routing accuracy and H4 are reported
per cell line, with PKLR (K562) as the matched-context subgroup. Restricting to K562 would leave
one element and no synthesis side of H2.

## 2026-10-25 - Cross-validation training grid: 60 runs, folds are never cut

SS-E, SS-EI and KIN train at three seeds x five folds (45 runs), KIN-1T at two seeds x five
folds (10), KIN-SHUF at one seed x five folds (5). SS-EI-EISA rescores SS-EI and trains nothing.
Each fold is the only model allowed to score its held-out chromosomes, so cutting a fold would
remove variants from the comparison; the Phase 2 step 2 reduction is applied to seeds. The H2
arms keep the same seed set because section 8.3 pairs by seed and variant. KIN-1T is not tested
against KIN with the H2 thresholds. If KIN-SHUF at one seed shows any routing gain over SS-EI,
its other two seeds are trained before H2 is reported.

## 2026-10-25 - Phase 2 posterior heads are full-covariance Gaussians

Follows the Phase 1 SBC result. Each head outputs the mean and a 3 x 3 Cholesky factor of
q(a, b, c | sequence), with the parameterization of the Phase 1 per-gene posteriors. The
posterior correlation is set by the design and is the same for every gene, so an amortized
covariance can represent it. Posterior means, and so routing, do not depend on the family when
the posterior is Gaussian; the amortization gap, held-out coverage and the non-contraction of
unidentified directions under the SS arms do. Section 6.2's diagonal heads are not trained.

## 2026-10-25 - SS-arm priors carry no information along unidentified directions

The ELBO-optimal posterior mean along a direction the likelihood does not constrain is the prior
conditional mean given the identified components, for any posterior covariance and for an
amortized network. An SS arm's predicted (delta a, delta b, delta c) is therefore a linear map
of its identified deltas, fixed by the prior covariance. Under the KIN-fitted prior SS-EI ties a,
b and c on a pure decay change; under the SS-fitted prior it routes a decay change to a; under a
prior independent of the identified components it routes every single-rate change to the right
rate with margin 0.25. The map for each candidate prior goes in docs/DESIGN_CHECK.md.

The SS arms use the independent prior: the mean and covariance of the identified components
(a - b and a - c under SS-EI, a - c under SS-E) are fitted, and the remaining component is
independent of them, with variance 1 in log units and mean set so the prior median c is
log(ln 2 / 4 h). This is the strongest steady-state baseline and carries no kinetic
information. SS-E ties a and c for every variant, so its pooled routing accuracy is 0.5 up to
noise and the SS-E clause of H2 reduces to KIN pooled accuracy >= 0.65. SS-EI routing under the
KIN population prior is reported as a sensitivity row by rescoring the identified deltas, and
is not part of H2.

## 2026-10-25 - KIN-1T keeps the KIN-fitted TT-seq scale and contamination

The two library scales and the contamination are global scalars with no gene-specific content
and stand in for the spike-in normalization of the original study. With them free, a 5 min
pulse confounds the scale with a common shift of b. Two checks accompany the reuse: the median
half-life implied by the scale agrees with the Schwalb-published decay rates within a factor of
2, and a KIN-1T refit with the scale at 0.5x and 2x reports the change in per-gene c and in
routing. If the first check fails, the scale is calibrated to the published median and the
change is logged. The TT-seq intronic dispersion under KIN-1T (NB size 410, against 3.7 under
KIN) is reported as a cross-assay discrepancy.

## 2026-10-25 - TimeLapse-seq fraction new covers exonic reads

The Mol Cell 2024 methods (PMC11315470) state that GRAND-SLAM computes fraction new from
alignments that fully overlap exon features, ignores reads that overlap introns, and that this
default was used for every analysis except the isoform analysis. The deposited per-gene tables
have no intronic entries. Open item: the lab's run script (churchmanlab/rna_flow) passes
`-introns` to gedi Slam, whose effect the GRAND-SLAM documentation does not describe. Reads
fully inside exons include pre-mRNA reads, which the exonic mixture over the two pools models.

## 2026-10-25 - TERT GABPA knockdown and control score sets are not used

MaveDB holds four TERT promoter score sets: HEK293T, SF7996, and SF7996 under GABPA siRNA
knockdown and scrambled siRNA. The HEK293T and SF7996 sets are used. The knockdown set measures a
perturbed trans environment, and the scrambled control is a third measurement of the same element
in SF7996 that the plan's element list does not name. All 12 added elements loaded; none is on
chrX, chrY or an unplaced contig, and each element sequence equals the hg38 plus strand at its
MaveDB coordinates, the same handling as PKLR, LDLR and SORT1.

## 2026-10-25 - MPRA elements on primary validation and test chromosomes are kept

BCL11A (chr2) and both MYC enhancers (chr8) lie on primary test chromosomes. Routing is scored
under cross-validation, where each element is scored only by the fold model that held its
chromosome out, so these elements stay in the set. The test that barred MPRA elements from
validation and test chromosomes is replaced by a check that every variant's recorded fold is the
fold holding its chromosome out.

## 2026-10-25 - Correction: SS-E routes every variant to a

The entry "SS-arm priors carry no information along unidentified directions" states that SS-E
ties a and c. In the Phase 1 likelihood about 10% of exonic signal is pre-mRNA, so SS-E
constrains (1, -0.10, -0.90) rather than a - c, and under the independent prior it routes every
variant to a: promoter set 1, 3'UTR set 0, pooled 0.5, with no ties (docs/DESIGN_CHECK.md). The
SS-E clause of H2 still reduces to KIN pooled accuracy >= 0.65.

## 2026-10-25 - H2 and H3 amended before any variant is scored

The routing simulation (docs/DESIGN_CHECK.md, results/design_sim/) shows that for variants with a
single known mechanism, SS-EI routes as well as KIN at zero estimation noise, because exonic and
intronic steady state already separate a pure synthesis change from a pure decay change. KIN
gains only on variants that change more than one rate, which neither MPRA set labels. The
original SS-EI clause of H2 can therefore pass only through the anisotropy of network errors,
and SS-EI-EISA is expected to match or beat KIN on single-mechanism sets. No variant had been
scored when this amendment was made.

- **H2 (amended).** (i) KIN pooled routing accuracy exceeds SS-E by at least 0.15, with the
  paired bootstrap 95% interval excluding zero (unchanged; equivalent to KIN >= 0.65). (ii) KIN
  and SS-EI pooled routing accuracy are equivalent: the paired bootstrap 95% interval of
  KIN - SS-EI lies inside [-0.05, 0.05]. An interval outside that band in either direction
  rejects equivalence and is reported with its sign; an interval wider than the band is
  inconclusive.
- **H3 first part (amended).** SS-EI-EISA routing accuracy on each variant set is at least KIN
  minus 0.05 (one-sided, paired bootstrap). The second part (splice-proximal variants) is
  unchanged.
- **Added secondary analysis.** For each trained arm, the error of predicted variant deltas is
  decomposed into the steady-state directions and the (1, 1, 1) direction, to explain the H2
  (ii) outcome.

## 2026-10-25 - Null variants and 3'UTR bootstrap clusters

A variant counts as null for the section 8.2 AUROC only if it is null (FDR > 0.5) in every
measurement that includes it; variants significant in one measurement and null in another are
excluded from the null set. The 3'UTR bootstrap resamples genes, then variants within gene, as
section 8.3 specifies for endogenous loci.

## 2026-10-25 - Phase 4 (LCL replication) is not run

Time-based decision, made at the start of Phase 2 and before any variant was scored. Phases 2, 3
and 5 had to finish in one overnight session of about 10 hours on one machine, and Phase 4 needs
a second data pipeline (GEUVADIS total RNA, 1000 Genomes genotypes, cis-QTL mapping) and a
second training grid. H3's second part (splice-proximal variants) depends on it and is reported
as untested.

## 2026-10-25 - Phase 2 implementation choices, fixed before training

- **Priors and noise are fixed at Phase 1 values.** Each arm's network is trained with the prior
  and assay noise parameters of its design's Phase 1 empirical-Bayes fit (KIN-SHUF uses KIN's;
  SS arms use the independent prior built from their fit), held fixed. The Phase 1 fits used the
  primary training chromosomes, which include chromosomes some folds hold out; these are 9 prior
  numbers and the global noise scalars, with no gene-specific content.
- **Training genes.** The 7,207 Gate G0 genes with annotated introns, minus the fold's held-out
  chromosomes. One training chromosome per fold is set aside for early stopping
  (configs/base.yaml: fold0 chr7, fold1 chr4, fold2 chr9, fold3 chr16, fold4 chr10).
- **Gene-level metrics use out-of-fold predictions.** No separate primary-split model is trained.
  Every gene is predicted by the fold model that held its chromosome out; section 8.1 metrics are
  reported on the primary test chromosomes (chr2, chr8, chr13) and on all genes. SLAM-seq is in no
  arm's likelihood, so every arm's c is compared with a held-out assay and no dedicated cross-assay
  run is needed. The TT-seq synthesis comparison uses a per-gene TT-seq proxy (labeled counts per
  kb of bins, size-factor normalized), not the published per-unit rates, whose units do not map
  one-to-one onto genes; KIN and KIN-1T train on TT-seq, so for them this is an in-assay check on
  held-out genes.
- **Transcripts longer than 12,288 nt keep their 3' end**, so every 3'UTR base stays in the input
  (308 of 19,111 genes are truncated).
- **Reporter constructs.** pGL4.23[luc2/minP] (GenBank DQ904455) for every satMutMPRA element,
  oriented on the strand of a protein-coding TSS within 1 kb of the element, else the plus
  strand; Kircher et al. cloned promoters into pGL4.11b, which lacks the minimal promoter, so
  promoter elements carry an extra 38 bp of minimal promoter here. pEGFP-N1 (GenBank U55762) stands
  in for the MapUTR master plasmid, whose sequence is not deposited, for both 3'UTR sets. Reporter
  covariates are the construct's own (no introns).
- **Sequence-overlap exclusion.** A variant is excluded from its fold's scoring when any 32-mer
  containing it (reference allele, either strand) occurs exactly in the input of a gene that fold
  trains on (sk.data.leakage, results/phase2/leakage.json). This implements section 5's exclusion
  of MPRA sequence overlapping training sequence at the variant level.
- **Training schedule.** AdamW, learning rate 1e-3, batch 32, five Monte Carlo samples, at most 40
  epochs with patience 10 on validation ELBO. Network and likelihood run on MPS (float32) and CPU
  (float64) respectively; CPU and MPS network outputs agree to 1.3e-6 on a test batch.

## 2026-10-25 - Phase 3 operational definitions, fixed before any variant is scored

The definitions are in the docstring of `src/sk/eval/phase3.py`, committed before any training
run finished. The ones the plan left open:

- A variant is significant if significant in any of its measurements (the Gate G2 convention).
  H4 uses the mean log2 effect over the measurements in which it is significant.
- H2 (ii): an interval inside [-0.05, 0.05] is equivalence; an interval at most 0.10 wide that
  extends beyond the band rejects equivalence with the sign of that side; a wider interval, or one
  extending beyond both edges, is inconclusive.
- H3 (i) uses the 5th percentile of the paired bootstrap of SS-EI-EISA - KIN per set.
- The error decomposition takes the true delta as the measured effect on the known rate, (e, 0, 0)
  or (0, 0, -e) in natural log, rescales each arm's predicted deltas by a least-squares factor
  (predicted reporter deltas have no fixed scale relative to measured effects), and splits the
  residual into the (1, 1, 1) component and the steady-state plane.
- 3'UTR variants outside annotated genes are clustered by 100 kb window for the bootstrap.

## 2026-10-25 - Non-finite training steps are skipped

KIN seed 0 fold1 stopped at epoch 14: a Monte Carlo draw from the posterior produced non-finite
TimeLapse-seq means and the likelihood raised. From commit 67331e2 a training step whose ELBO is
not finite is skipped and counted (`skipped_steps_total` in each run's history; more than 50 stops
the run), and a validation batch is redrawn up to three times. KIN seed 0 fold0 and fold2 ran
before the change and never hit such a step, so their training is identical under either code.
KIN seed 0 fold3, which had started under the old code a minute before the change, failed the
same way at epoch 14. Both failed runs were discarded and rerun under the guard. No variant result
had been examined.

## 2026-10-25 - Post hoc: the H2 (ii) rule misclassifies an interval wholly outside the band

Follows a seen result (seed 0 of SS-E, SS-EI and KIN). The operational rule fixed before
scoring labels any interval wider than 0.10 "inconclusive", including an interval that lies
entirely outside [-0.05, 0.05]. The amended H2 text says an interval outside the band rejects
equivalence. The seed-0 interval of KIN - SS-EI is about [-0.29, -0.15]: wider than 0.10 and
entirely below -0.05. The pre-registered rule stays the verdict. The evaluation also reports, as a
post hoc field, whether the interval lies entirely outside the band, and the report states both.
Neither reading supports equivalence.

## 2026-10-25 - KIN-SHUF shows no routing gain over SS-EI; its extra seeds are not trained

KIN-SHUF seed 0 pooled routing accuracy is 0.365 against 0.528 for SS-EI (paired difference
-0.163, 95% interval [-0.202, -0.115]), so the cross-validation grid entry's condition for
training its other two seeds is not met. KIN-SHUF is 0.066 above KIN ([0.000, 0.123]).

## 2026-10-25 - KIN-1T seed 1 is cut for time

At about 7.3 minutes per run, the remaining queue (seeds 1 and 2 of SS-E, SS-EI and KIN, then
KIN-1T seeds 0 and 1) would end after the 03:30 PDT cutoff for launching training. KIN-1T seed 1,
the last group in priority, is removed from the queue; KIN-1T keeps seed 0 on all five folds.
KIN-1T is not part of any H2 or H3 test. The KIN-1T scale checks promised in "KIN-1T keeps the
KIN-fitted TT-seq scale and contamination" (refits at 0.5x and 2x the scale) and the optional
Saluki zero-shot baseline are also cut for time.

## 2026-10-26 - Evaluation bug fixes after scoring began

Both follow seen results and change no metric definition. Subgroup intervals (per cell line, per
observable-range group) counted a resample that drew none of a subgroup's clusters as accuracy 0,
which pulled single-element subgroups' lower bounds to 0; such resamples are now skipped.
Single-seed arms (KIN-SHUF, KIN-1T) raised an error under the three-seed pairing of the H2 arms;
they are now paired on their own seed. The H2 to H4 verdicts use neither path.

## 2026-10-26 - Post hoc bug fix: 3'UTR reporter variants rescored with their own covariates

Found in review after every result had been seen. `score.utr_reporter` built each insert's
reporter input in a loop and then scored every insert of the fold with the flags and covariates
of the last insert, so each 3'UTR reporter delta was computed with another construct's GC and
length covariates. Promoter and endogenous scores were unaffected. The scorer now stacks each
insert's own flags and covariates, and `python -m sk.eval.rescore` rescored the 3'UTR reporter
rows of all 55 runs from their saved models (MPS, 7.4 minutes). Replaying the old scorer
reproduces the stored deltas exactly, so the whole change comes from the bug.

Over 204,479 rescored rows the change in a delta component has median 9.5e-6, 99th percentile
1.2e-3 and maximum 0.026 (`results/phase3/utr_reporter_rescore.json`), against median |delta|
of 4e-4 to 6e-3 per arm. Phase 3 was rerun. 3'UTR accuracy changed by at most 0.008 per arm
(KIN 0.592 -> 0.600, SS-E 0.529 -> 0.526); KIN pooled 0.518 -> 0.522; H2 (i) -0.067
[-0.162, +0.069] -> -0.062 [-0.157, +0.075]; H2 (ii) -0.0001 [-0.028, +0.040] -> +0.003
[-0.025, +0.044]; H3 (i) 3'UTR +0.377 -> +0.370; H4 3'UTR -0.016 (n 2,242) -> -0.009 (n
2,305). No verdict changed.

## 2026-10-26 - Post hoc correction: the KIN-SHUF control is paired on seed 0

The evaluation compared KIN-SHUF seed 0 with the three-seed means of SS-EI and KIN, and the
report quoted that unpaired result (-0.15 [-0.19, -0.11]). The control is now paired on seed 0,
as the 2026-10-25 entry on KIN-SHUF computed it. After the rescoring above, KIN-SHUF - SS-EI is
-0.163 [-0.203, -0.115] and KIN-SHUF - KIN is +0.059 [-0.007, +0.114]. The pre-registered rule
(cross-validation grid entry) trains KIN-SHUF seeds 1 and 2 if seed 0 shows a routing gain over
SS-EI. It shows none, so the decision not to train them stands.

## 2026-10-26 - Post hoc diagnostic: routing does not depend on the variant

Added after the routing results were seen. `python -m sk.eval.routing_diagnostic` compares,
per arm and set, the share routed to the known rate for significant variants, null variants and
random single-base edits of the same reporter constructs (seed means; tables.md). Every arm
routes the three groups to its known rate within 0.07 of each other, except KIN-1T on promoter
variants (0.37, 0.55, 0.50): SS-EI-EISA sends 97% of significant, null and random 3'UTR edits
to c, and 82%, 84% and 81% of promoter ones; KIN sends 60%, 61% and 60% of 3'UTR ones to c.
SS-EI's raw deltas of significant variants lie along (1, 1, -2), a pure decay change with
(1, 1, 1) removed (median |cos| 0.91 promoter, 0.98 3'UTR), so EISA routes nearly every edit to
c. SS-EI-EISA - KIN on each set (H3 (i)) therefore compares two variant-independent output
directions against the set's known rate. The H3 (i) verdict is unchanged: it fails, passing on
3'UTR variants and failing on promoter variants.

## 2026-10-26 - Post hoc: the gene-level decay gain is leak-free and mostly reachable from covariates

Added after the gene-level results were seen, from a review of the decay result
(`python -m sk.eval.gene_diagnostics`; seed 0, the 1,076 test-chromosome genes with a SLAM-seq
rate). Verified: every gene is predicted once, by the fold model that held its chromosome out,
for every arm; SLAM-seq enters no likelihood; KIN beats KIN-SHUF by +0.21 [0.16, 0.26] and SS-EI
by +0.25 [0.19, 0.30] (paired gene bootstrap); KIN's correlation is the same on genes sharing
no 32-mer with their fold's training genes (0.50, n 389); and KIN correlates less with SLAM-seq
on its own training genes than on held-out genes (0.46 and 0.41 against 0.54 and 0.49, folds 1
and 3). A covariate-only baseline, a ridge regression of Phase 1 KIN c on the ten gene
covariates fitted per fold on that fold's training genes, reaches 0.44 against KIN's 0.49 (KIN
- baseline +0.04 [0.01, 0.07]). KIN's partial rank correlation with SLAM-seq beyond the baseline
is 0.23 [0.17, 0.29], against 0.08 for KIN-SHUF. TimeLapse-seq against SLAM-seq on the same
genes is 0.64.

## 2026-10-26 - Post hoc: amortized posteriors are equally overconfident on training genes

Added after the amortization results were seen (`python -m sk.eval.gene_diagnostics`, fold1
seed 0). KIN's median posterior SDs (a, b, c) are 0.13, 0.17, 0.09 on training genes,
0.12, 0.17, 0.09 on early-stopping genes and 0.13, 0.17, 0.09 on the held-out NUTS genes,
against NUTS SDs of 0.14, 0.17, 0.12. Its means miss the per-gene data posterior by 0.7 to 1.0
log units (SD of the error per rate) on training and held-out genes alike: the median KL from
the Phase 1 per-gene posterior is 84 nats on training genes, against 66 from NUTS on held-out
genes, and exonic 90% predictive coverage is 0.28 on both. Widening the posterior by the
empirical covariance of its mean errors against NUTS brings the median KL to 5.6 nats and
coverage to 0.89 (SS-EI 4.0 and 0.89, KIN-1T 6.2 and 0.87). The widening is fitted on the same
200 genes it is scored on. Inferred: for genes the network cannot tell apart, the summed ELBO is
maximized by a posterior near the geometric mean of their data posteriors, which keeps
data-level precision.

## 2026-10-26 - Plan redacted for publication

PROJECT_PLAN.md as published omits the instructions about the execution environment and working
conventions (its original section 0), the reporting instructions (original section 15), one
sentence of motivation and one cross-reference to a private repository. Hypotheses, metrics,
thresholds, splits, arms, gates and every other section are unchanged. Section numbers in this
log refer to the plan's numbering, which the redaction keeps.
