# Sequence-to-kinetics: which measurements make regulatory variant effects identifiable as synthesis, processing or decay?

The project plan, written before any experiment was run. Instructions about the execution environment and working conventions were removed for publication; hypotheses, metrics, thresholds, splits, arms and gates are unchanged. Any change made after work starts goes in `docs/DECISIONS.md`, with the date and the reason, at the time it is made.

## 1. Summary

Steady-state RNA abundance is a ratio. For a gene with transcription rate alpha, pre-mRNA processing rate beta and mature mRNA decay rate gamma, the steady-state pre-mRNA level is alpha/beta and the mature level is alpha/gamma. Sequence-to-expression models such as Borzoi (Linder et al., Nat Genet 2025) are trained on steady-state coverage, so a variant that raises transcription and a variant that stabilizes the mRNA can produce the same predicted change in exonic coverage. Borzoi also predicts intronic coverage, and a common way to separate transcriptional from post-transcriptional effects is to compare exonic and intronic changes (exon-intron split analysis, EISA; Gaidatzis et al., Nat Biotechnol 2015). That comparison assumes the processing rate beta does not change.

This project makes the identifiability question precise and tests it on real variants. A sequence model predicts the three log rates of a linear kinetic ODE for each gene from DNA and mRNA sequence. The ODE has a closed-form solution, so the likelihood of steady-state and metabolic-labeling measurements is exact. The sequence network is an amortized inference network: it maps sequence to a posterior over rates, so it generalizes across genes and to variant sequences. The model is trained under four measurement designs, from exonic steady state only (what Borzoi-style data provide) up to steady state plus labeling time courses. Each trained model then scores variants whose mechanism is known from massively parallel reporter assays: promoter and enhancer variants that change transcription, and 3'UTR variants that change mRNA abundance through stability. The question is how well each design routes each variant to the right rate.


## 2. The kinetic model and its identifiability

### 2.1 Model

For gene g, with u the pre-mRNA (unspliced, intron-containing) and m the mature mRNA:

```
du/dt = alpha - beta * u
dm/dt = beta * u - gamma * m
```

Write a = log alpha, b = log beta, c = log gamma.

- Steady state: u* = alpha/beta, m* = alpha/gamma. Observed exonic and intronic counts are proportional to these, with a library-size factor and a gene-specific length and mappability offset.
- Labeling pulse of duration t from steady state (4sU, TT-seq, SLAM-seq, TimeLapse-seq): the labeled fractions are
  - `f_u(t) = 1 - exp(-beta t)`
  - `f_m(t) = 1 - (beta exp(-gamma t) - gamma exp(-beta t)) / (beta - gamma)`, with the beta = gamma limit `1 - (1 + gamma t) exp(-gamma t)`.
- Labeled fractions depend only on b and c. Steady-state levels fix a - b and a - c.

The ODE is linear and solved in closed form. No numerical ODE solver or neural ODE is needed, and the plan does not use one. An extension with a nuclear and a cytoplasmic mature pool (section 13) keeps the closed form.

### 2.2 What each design identifies

In log-rate space (a, b, c), with a global scale on alpha that no relative measurement fixes:

| Design | Observations | Identified combinations | Unidentified direction |
|---|---|---|---|
| SS-E | exonic steady state | a - c | two directions: (1,0,1) and (0,1,0) |
| SS-EI | exonic and intronic steady state | a - c, a - b | (1,1,1): all rates scaled together |
| SS-EI + EISA assumption | as SS-EI, with delta b = 0 imposed for variants | a, c given b fixed | none, if the assumption holds |
| KIN | SS-EI plus labeled fractions at one or more pulse times | a, b, c | none, for rates within the observable range of the pulse times |

The consequence for variants: under SS-EI, any change along (1,1,1) is invisible. A variant that raises gamma by d gives delta(a - c) = -d and delta(a - b) = 0, which is exactly what a variant that lowers both alpha and beta by d gives. A variant that alters processing (a splice-region variant, an intron-retention change) breaks the EISA assumption and is misrouted under SS-EI + EISA. Under KIN, labeling time courses resolve the (1,1,1) direction because they set an absolute time scale.

Phase 1 turns this table into numbers: the Fisher information of (a, b, c) under each design and realistic noise, the posterior contraction as a function of pulse times, and the range of half-lives for which a given set of pulse times identifies gamma (a single 5 to 10 minute TT-seq pulse identifies fast-decaying transcripts well and slow ones poorly; this is expected and must be quantified, not assumed).

## 3. Hypotheses

**H1 (identifiability on simulated data, a verification of section 2.2).** On genes simulated from the fitted prior with the real assays' noise models and pulse times, posterior 90% intervals for c are at least 3 times narrower under KIN than under SS-EI, for genes whose half-life lies inside the observable range defined in Phase 1. Under SS-EI, the posterior for (a + b + c)/3 does not contract relative to the prior (ratio of posterior to prior standard deviation > 0.8).

**H2 (routing of variants with known mechanism, the main result).** Define routing accuracy as the fraction of significant MPRA variants whose largest predicted absolute log-rate change is on the correct rate (a for promoter and enhancer variants, c for 3'UTR stability variants). KIN routing accuracy, pooled over both variant sets, exceeds SS-E by at least 0.15 and SS-EI by at least 0.05, with paired locus-bootstrap 95% intervals excluding zero.

**H3 (where EISA fails).** SS-EI + EISA routes 3'UTR stability variants about as well as KIN (difference within +/- 0.05), and routes worse than KIN on variants within 50 nt of a splice site or in introns of the modeled genes (variants where beta can change). The second part is tested on the splice-proximal subset of the natural-variant set (section 4.4) if Gate G2 passes for it; otherwise it is reported as untested.

**H4 (magnitude, secondary).** Within the correctly routed variants, the predicted change on the correct rate correlates with the measured MPRA effect: Spearman rho > 0 with 95% interval excluding zero, for each variant set and for KIN.

**Non-claims.** No claim of state-of-the-art expression prediction, no claim that the model replaces Borzoi, no claim about cell types other than those trained on, and no claim from reporter assays about endogenous loci beyond what section 6.3 states.

## 4. Data

Every source gets a row in `data/SOURCES.md`: URL, accession, version or date, checksum, license, and what it is used for. Accessions marked "unverified" must be resolved in Phase 0 before use; if one cannot be found or is unsuitable, use the listed fallback and log the choice.

### 4.1 Primary cell line: K562

| Source | Use | Status |
|---|---|---|
| GSE75792, TT-seq and RNA-seq in K562 (Schwalb et al., Science 2016), 12 samples | Synthesis (TT-seq, 5 min 4sU pulse) and total RNA, exonic and intronic counts | Verified on GEO (title "TT-Seq captures the human transient transcriptome", 12 samples) |
| GSE126521, SLAM-seq mRNA decay in 293T and K562 (Wu et al., eLife 2019), 28 samples | Decay time course (chase) for endogenous mRNAs in K562 | Verified on GEO (title "ORFome decay in 293t and k562"); confirm which samples are endogenous K562 time points |
| Subcellular TimeLapse-seq in K562, 4sU 0, 15, 30, 60, 120 min (Smalec, Ietswaart et al., Churchman lab; Mol Cell 2024) | Labeling time course with several pulse times; nuclear and cytoplasmic fractions for the section 13 extension | Accession unverified; check the paper's data availability statement |
| ENCODE K562 total RNA-seq (stranded, rRNA-depleted), two replicates | Additional steady-state exonic and intronic counts | Pick the experiment in Phase 0 and record its accession |
| GENCODE v44 basic annotation, hg38 | Gene models, TSS, introns, UTRs, CDS | Standard |
| hg38 FASTA | Sequences | Standard |

Per-gene quantities are obtained from processed files where they exist (count tables, published rate tables in supplementary data). Re-alignment is a fallback, with STAR (index fits in 128 GB) and, for SLAM-seq, conversion calling with GRAND-SLAM or SLAM-DUNK. Phase 0 decides and logs which. Re-implementing a published rate estimate is not the goal; the model needs counts and labeled fractions with their noise model.

### 4.2 Promoter and enhancer variants (synthesis-side ground truth)

Agarwal et al., "Massively parallel characterization of transcriptional regulatory elements", Nature 639:411-420 (2025), doi 10.1038/s41586-024-08430-9. lentiMPRA of more than 680,000 sequences in K562, HepG2 and WTC11, including allelic variant effects. Processed data are listed on the IGVF portal (`data.igvf.org`, analysis set IGVFDS3278AFYF was found, but whether it holds the K562 variant table is unverified). Use the K562 variant-effect table: log2 alt/ref activity, its standard error or FDR, and the 200 bp element sequences. Status: paper verified, data location unverified.

Fallback: the satMutMPRA saturation mutagenesis of promoters and enhancers (Kircher et al., Nat Commun 2019; GSE126550; `kircherlab.bihealth.org/satMutMPRA`), restricted to elements measured in K562 or HepG2.

### 4.3 3'UTR stability variants (decay-side ground truth)

- MapUTR (Fu et al., Nat Commun 2024, doi 10.1038/s41467-024-46795-7): about 17,000 rare 3'UTR variants tested for mRNA abundance in HEK293 and HeLa, about 22 to 27% significant. NCBI BioProject 972741 is linked to the study; processed effect tables are expected in the supplement. Status: paper verified, effect tables unverified.
- Su et al., eLife reviewed preprint 97682 ("Multiplexed assays of human disease-relevant mutations reveal UTR dinucleotide composition as a major determinant of RNA stability"): 6,555 disease-database UTR variants with stability measured in HEK293T and SH-SY5Y. Status: paper verified, tables unverified.
- Griesemer et al., Cell 2021 (genome-wide functional screen of 3'UTR variants): fallback or addition, unverified.

These are measured in HEK293, HeLa or SH-SY5Y, not K562. Stability determinants (AU-rich elements, microRNA sites, codon and dinucleotide composition) are largely shared across cell lines, and the model is trained on K562 decay, so the cell-line mismatch lowers the ceiling on H4 magnitude correlations more than it affects routing. The report states this. Variants in microRNA sites of microRNAs that are not expressed in K562 are a known source of misses; flag them using a K562 microRNA expression table if one is readily available, and report them separately.

### 4.4 Natural variants in one cell type (secondary, conditional on Gate G2)

GSE75220, "4sU metabolic labeled transcripts in 65 YRI LCLs" (Li et al., Science 2016, "RNA splicing is a primary link between genetic variation and disease"), 372 samples. Verified on GEO. Combined with total RNA-seq of the same lines (identify the series used by Li et al.; GSE19480 is the 2010 Pickrell et al. RNA-seq of 69 YRI LCLs) and 1000 Genomes phase 3 YRI genotypes, these give genetic effects on newly synthesized RNA and on total RNA in the same individuals. A variant whose effect on total RNA is not matched by an effect on 4sU-labeled RNA acts after transcription.

Use: compute cis-QTL effects for 4sU and total RNA with a standard linear model (genotype dosage, top expression PCs as covariates, cis window +/- 100 kb, permutation FDR), take lead variants that are significant for one and not the other, and test routing as in H2. Splice-proximal variants among them give the H3 test. This requires training a second model on LCL kinetic data (the 4sU and total RNA from this series), so it is Phase 4 and runs only if Phases 0 to 3 finish and G2 passes.

GSE34204 (Duan et al., mRNA half-lives in 7 HapMap LCLs) is too small for QTLs and is not used.

### 4.5 External baselines

- **Saluki** (Agarwal and Kelley, Genome Biol 2022): half-life from mRNA sequence. Use the released weights zero-shot to score 3'UTR variants (delta predicted half-life), as a decay-only reference. Locate the weights in the calico GitHub organization; status unverified.
- **EISA on the trained SS-EI model:** the SS-EI + EISA arm (section 7) plays the role of a Borzoi-style decomposition. Scoring real Borzoi or Flashzoi is optional (section 13) because 524 kb inference on MPS is slow.
- **Motif baseline:** for promoter variants, delta of the best JASPAR 2024 match weighted by TF expression in K562; for 3'UTR variants, change in AU-rich element count and in canonical microRNA seed matches. Routing for this baseline is by region (promoter -> a, 3'UTR -> c), so it routes perfectly by construction and is used only for H4 magnitudes.

## 5. Splits

- Genes are split by chromosome. Validation: chr7. Test: chr2, chr8, chr13. chrX, chrY, chrM and genes on unplaced contigs are excluded.
- Variants are scored only if their gene, or the element's nearest gene for enhancer-like MPRA elements, is on a test chromosome. MPRA elements whose sequence overlaps any training-chromosome sequence are excluded. `configs/splits.yaml` is the single source of truth and `tests/test_splits.py` asserts no leakage.
- If the routing sets on test chromosomes are too small for Gate G2, the fallback is 5-fold chromosome cross-validation (folds fixed in `configs/splits.yaml` before training), with every variant scored by the fold model that did not see its chromosome. Decide in Phase 0 from the counts and log it.

## 6. Model

### 6.1 Inputs

Each gene is represented by one input with two segments, so that every rate head sees the whole sequence and routing is learned, not hard-coded:
1. **Promoter segment:** 2,048 bp from TSS - 1,536 to TSS + 512 of the gene's GENCODE basic principal transcript (APPRIS principal, else the longest basic transcript; fixed rule recorded in configs).
2. **Transcript segment:** the mature mRNA (5'UTR, CDS, 3'UTR), truncated or padded to 12,288 nt, with Saluki-style extra tracks: a codon-frame track marking the first nucleotide of each codon and a splice-junction track marking exon-exon junctions.
3. **Gene-level covariates:** number of introns, total intron length, transcript length, 3'UTR length, CDS length, GC content of each region. These enter all heads.

A region-identity channel marks which segment each position belongs to.

### 6.2 Network

A convolutional tower shared across both segments (dilated residual blocks, as in Saluki and Basenji; size chosen so one epoch on about 12,000 genes runs in minutes on MPS), attention pooling within each segment, concatenation with covariates, and three heads that output the mean and log standard deviation of the posterior over a, b and c. The heads are amortized variational posteriors q(a, b, c | sequence) with diagonal Gaussian form; section 13 lists a full-covariance variant.

### 6.3 Variant scoring

For an endogenous variant, substitute the alternate allele into the gene's input and take the difference in posterior means (delta a, delta b, delta c). For MPRA variants, score in reporter context: the promoter segment holds the 200 bp lentiMPRA element followed by the lentiMPRA minimal promoter, and the transcript segment holds the reporter mRNA sequence (from the lentiMPRA and MapUTR vector sequences in their papers or Addgene; record exact sequences and sources). For 3'UTR MPRA variants, the 3'UTR fragment is inserted into the reporter transcript as in the assay. Reporter context is off-distribution for a model trained on endogenous genes; the report states this, and the endogenous-context score for 3'UTR variants that fall in modeled genes is reported alongside as a check.

Routing for a variant: the rate with the largest |delta|. All deltas are in log space, so they are comparable.

### 6.4 Likelihood

Per gene, per sample:
- Exonic and intronic counts: negative binomial, with mean = size factor x length and mappability offset x steady-state level, and a per-assay dispersion fitted jointly.
- Labeled reads: for TT-seq and 4sU pulldown, the labeled library is modeled as the labeled fraction times the steady-state level with its own size factor and a cross-contamination term. For SLAM-seq and TimeLapse-seq, labeled fractions come from T-to-C conversion counts with a binomial mixture over per-read uridine content, as in GRAND-SLAM; if per-read data are too costly to produce, use per-gene new-to-total ratio estimates with their reported uncertainty as Gaussian observations in logit space, and log the choice.
- The global scale of alpha is fixed by setting the median a over training genes to 0.

Each arm uses the subset of these terms its design allows (section 7).

### 6.5 Inference and training

Training maximizes the evidence lower bound summed over training genes: the expected log-likelihood under q minus the KL divergence from q to a hierarchical prior. The prior is Gaussian on (a, b, c) with mean and covariance learned as global parameters (empirical Bayes), so genes borrow strength from each other and the prior carries the population structure of rates. Five Monte Carlo samples per gene per step, reparameterized.

Two checks of the inference itself, both required:
1. **Simulation-based calibration (SBC; Talts et al. 2018):** simulate 1,000 genes from the fitted prior and the fitted noise model, at the real assays' pulse times and depths, refit the posterior for each (amortized, with the sequence input replaced by the simulated gene's identity, i.e. a per-gene free variational posterior), and check rank histograms for a, b, c under each design. Pass: chi-square uniformity p > 0.01 for each rate, and 90% interval coverage between 0.85 and 0.95.
2. **Amortization gap:** on 200 randomly chosen validation genes, compare the amortized posterior from the sequence network with a per-gene posterior from NUTS (NumPyro or a PyTorch NUTS implementation) given the same data and prior. Report the mean KL and the shift in posterior means in units of posterior standard deviation. This measures how much the sequence model's limited information costs relative to the data, and is a central figure.

## 7. Arms

### 7.1 Arms

| Arm | Likelihood terms used | What it represents |
|---|---|---|
| SS-E | exonic steady state | Borzoi-style exonic expression training |
| SS-EI | exonic and intronic steady state | Borzoi-style coverage including introns |
| SS-EI-EISA | as SS-EI; at scoring time, b is held at the reference value for variants (delta b = 0) and the a and c deltas are re-solved from the predicted deltas in a - b and a - c | EISA-style decomposition of a steady-state model |
| KIN | steady state plus all labeling data | The full kinetic design |
| KIN-1T | steady state plus only the single shortest-pulse labeling data (TT-seq 5 min) | The cheapest kinetic design; tests how much one time point buys |
| KIN-SHUF | as KIN, with labeling data permuted across genes within expression deciles | Control: a KIN gain must vanish when the labeling data carry no gene-specific information |

Three seeds per arm. Architecture, optimizer, schedule, batch size, early stopping (best validation ELBO, patience 10 epochs) and covariates are identical across arms and live in `configs/base.yaml`.

### 7.2 Why SS arms still have three heads

SS-E and SS-EI keep all three heads. The unidentified directions are set by the prior and by whatever the sequence network learns to correlate with them. This is deliberate: it is what happens when a model with a mechanistic readout is trained on data that cannot identify it. Their routing accuracy measures how far prior structure and sequence correlations get without identifying data. The report must not describe SS arm routing as "random"; it is determined by the learned prior and is measured.

### 7.3 Required pre-training check (write it in `docs/DESIGN_CHECK.md` before Phase 2)

For each arm, state and verify:
1. Which log-rate directions the arm's likelihood constrains (section 2.2), confirmed numerically by the rank of the Jacobian of the expected observations with respect to (a, b, c) at 100 random prior draws.
2. That no arm's routing is fixed by architecture: both segments reach all three heads (a unit test perturbs a 3'UTR base and checks that delta a can be non-zero in a randomly initialized network, and perturbs a promoter base and checks that delta c can be non-zero).
3. That the motif baseline's routing is by construction and is excluded from H2 for that reason.
4. That SS-EI-EISA differs from SS-EI only in scoring, so any difference between them is attributable to the delta b = 0 assumption.
5. That KIN-SHUF preserves the marginal distribution of labeled fractions within expression deciles.

If any check fails, fix the design, log it, and re-run the check before training.

## 8. Metrics

### 8.1 Gene-level fit (test chromosomes)

- Spearman correlation of posterior mean c with half-lives from an assay held out of training for that arm (for KIN and KIN-1T, hold out SLAM-seq during a dedicated cross-assay run, train on TT-seq and TimeLapse-seq, and report correlation with SLAM-seq half-lives; recorded as a separate run).
- Spearman correlation of posterior mean a with TT-seq synthesis estimates, and of predicted log steady-state exonic level with observed.
- Posterior 90% interval coverage of held-out measurements under each arm's posterior predictive.

### 8.2 Routing (variants, the H2 to H4 metrics)

- **Routing accuracy:** as defined in H2, per variant set and pooled with each set weighted equally.
- **Routing margin:** for each variant, |delta on the correct rate| minus the largest |delta| on the other rates, divided by the sum of the three |delta|. Report the median and the distribution.
- **Magnitude correlation:** Spearman rho between delta on the correct rate (sign-adjusted: delta a for promoter variants, -delta c for 3'UTR variants) and the measured MPRA effect.
- **Significant variants:** MPRA FDR < 0.05 and |log2 effect| > 0.2 (fixed before scoring). Non-significant variants (FDR > 0.5) are used to check that predicted |delta| is larger for significant than for null variants (AUROC), which guards against a model that routes well only because it predicts large deltas everywhere.

### 8.3 Uncertainty

95% intervals from 2,000 bootstrap resamples over loci: variants are clustered by gene for endogenous variants and by MPRA element for reporter variants, and clusters are resampled together. Between-arm comparisons are paired by seed and by variant. An arm's value is the mean over its seeds.

## 9. Phases and gates

**Phase 0: data and foundations.**
1. Resolve every unverified source in section 4. Download, checksum, record in `data/SOURCES.md`.
2. Build per-gene tables for K562: exonic and intronic counts per sample, labeled fractions per pulse time, gene covariates, sequences.
3. Build the routing variant sets with reporter-context sequences; record counts per set and per split.
4. Leakage tests pass.

**Gate G0 (data).** At least 5,000 genes have all of: exonic and intronic counts above 10 reads in every total RNA-seq sample, and labeled-fraction data in at least two pulse or chase times. Half-lives from two independent K562 assays (for example SLAM-seq and TT-seq derived) agree at Spearman rho >= 0.5 on shared genes. If not, stop and report, because the kinetic ground truth would be too noisy to support any routing claim.

**Phase 1: identifiability analysis and simulation (no sequence model).**
1. Fit the per-gene model with free per-gene posteriors (no sequence network) under each design, to estimate the hierarchical prior and noise parameters.
2. Compute Fisher information and posterior contraction under each design, at the real pulse times, as a function of half-life. Produce the "observable range" of half-lives for each design.
3. Run SBC (section 6.5). Test H1.

**Gate G1 (inference).** SBC passes for KIN for all three rates and for SS-EI for a - b and a - c. If SBC fails, fix the inference before any sequence model is trained.

**Phase 2: sequence model.**
1. Write and pass `docs/DESIGN_CHECK.md` (section 7.3).
2. Train KIN with one seed. Time one epoch and project the full grid (6 arms x 3 seeds). If the projection exceeds 36 hours, cut KIN-SHUF to two seeds first, then KIN-1T, and log the decision.
3. Train the grid.

**Gate G2 (routing sets).** At least 150 significant promoter or enhancer variants and at least 150 significant 3'UTR variants are scoreable on held-out chromosomes (or under the cross-validation fallback). For the Phase 4 LCL set, at least 100 lead variants with discordant 4sU and total-RNA effects.

**Phase 3: evaluation.** All metrics in section 8, the amortization gap, H2 to H4, and baselines.

**Phase 4 (conditional): LCL replication.** If time allows and G2 passes for the LCL set, train KIN and SS-EI on GSE75220 kinetic data, compute QTLs, and test routing on natural variants, including the splice-proximal subset for H3.

**Phase 5: report.** `docs/REPORT.md` and README, with figures:
1. Identifiability: posterior contraction of c against half-life under each design (the Phase 1 figure).
2. SBC rank histograms.
3. Routing accuracy per arm and variant set, with intervals (the central figure).
4. Routing margin distributions.
5. Amortization gap.
6. Gene-level calibration and cross-assay correlation.
Every claim made during the work and later withdrawn is listed with the reason.

## 10. Repository layout

```
sequence-kinetics/
  src/sk/
    data/       downloads, count tables, labeled fractions, sequences, variant sets, reporter constructs
    kinetics/   closed-form ODE solutions, likelihoods, Fisher information, simulation
    model/      sequence network, variational heads, hierarchical prior
    infer/      ELBO training, per-gene free posteriors, NUTS reference, SBC
    eval/       gene-level metrics, variant scoring, routing metrics, bootstrap, figures
    utils/      seeding, run manifests, logging
  configs/      base.yaml, splits.yaml, one file per arm
  docs/         DESIGN_CHECK.md, DECISIONS.md, COMPUTE_LOG.md, REPORT.md
  data/         SOURCES.md (raw and processed data are not committed)
  results/      committed metrics, tables, figures
  tests/
```

## 11. Tests

Each guards a specific way a result could be silently wrong:
- No test-chromosome gene or variant in training data, and no MPRA element sequence overlapping training sequence.
- Closed-form solutions match numerical integration of the ODE (scipy `solve_ivp`) to 1e-6 relative error across a grid of (beta, gamma, t), including beta = gamma.
- Labeled fractions are invariant to alpha, and steady-state ratios are invariant to scaling all three rates together (the identifiability claims in section 2.2).
- The Jacobian rank of each design's observation map matches the table in section 2.2.
- Reference allele matches hg38 at every endogenous variant, and alternate-allele inputs differ from reference at exactly one position in the expected segment.
- Reporter constructs reproduce the published vector sequence around the insertion site.
- Both segments reach all heads in an untrained network (section 7.3 item 2).
- Routing metrics on small hand-built examples, including ties.
- SBC rank computation on a conjugate Gaussian model where the posterior is known exactly.

## 12. Compute

All runs on the M4 Max. The model is small (about 12,000 genes, inputs of about 14 kb), so most of the time is data processing. Possible large costs: STAR alignment if processed counts are unavailable (budget about 1 hour per sample, run at `nice -n 15`), and NUTS on 200 genes (CPU). Every job gets a row in `docs/COMPUTE_LOG.md`: start time, git SHA, config hash, wall clock, purpose, outcome.

## 13. Extensions (only after Phase 5, each logged before starting)

- Nuclear and cytoplasmic mature pools from subcellular TimeLapse-seq: adds a nuclear export rate; the closed form still holds.
- Full-covariance Gaussian or normalizing-flow posterior heads; compare SBC and routing.
- Hierarchical model across cell lines (K562, HepG2, LCL) with partially pooled per-cell-line rate offsets.
- Flashzoi or Borzoi variant scores decomposed by EISA, as an external version of the SS-EI-EISA arm.

## 14. Risks

- **Reporter context is off-distribution.** Both MPRA sets are scored in reporter constructs. If routing on reporter constructs and on endogenous contexts disagree for the 3'UTR variants that can be scored both ways, report both and treat the reporter result with caution.
- **Cell-line mismatch for 3'UTR variants.** Section 4.3.
- **Kinetic ground-truth noise.** Single-pulse TT-seq constrains slow-decaying transcripts weakly. Gate G0 and the Phase 1 observable range make this explicit; routing results are also reported restricted to genes inside the observable range.
- **Most promoter MPRA variants have small effects.** The significance thresholds in 8.2 and the null-variant AUROC check address this.
- **A design that cannot fail.** Section 7.3 exists so that no arm's routing is decided by architecture; the motif baseline is excluded from H2 for that reason.
- **Data that cannot be found or processed in reasonable time.** Each source has a fallback; stop and report rather than substitute an untested source silently.

