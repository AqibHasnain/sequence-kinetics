# Compute log

Every run on one Apple M4 Max (16 cores, 128 GB unified memory, PyTorch MPS backend), at
`nice -n 15`. A result with no row here is not a result.

Phase 2 training and scoring code (`src/sk/train.py`, `src/sk/model/`, `src/sk/infer/amortized.py`,
`src/sk/eval/score.py`, `configs/base.yaml`) is identical from commit 67331e2 onward; KIN seed 0
fold0 and fold2 ran on 6ffa06c, which differs only in raising on a non-finite step instead of
skipping it, and neither run hit one. The 3'UTR reporter scores of every run were later
replaced by a post hoc rescoring with the fixed `score.utr_reporter` (last rows). "dirty" marks runs that started while regenerated result files and evaluation code (under
`results/` and `src/sk/eval/`, outside `score.py`) were uncommitted; the training job does not
read them.

| Start (UTC) | Git SHA | Config hash | Wall-clock min | Purpose | Outcome |
|---|---|---|---|---|---|
| 2026-10-25T20:30Z | 629360f | - | 25 | Phase 0: download every source (1.2 GB), build gene, K562 and variant tables, evaluate Gate G0 | Complete, all CPU; no alignment or GPU work was needed |
| 2026-10-25T22:10Z | a53a8b6 + uncommitted Phase 1 code (committed in 7dc3628 and the following commit) | 6f05a534ac2f | 5 | Phase 1 with diagonal posteriors: real-data fits of 4 designs on 5,918 genes, Fisher and Jacobian analysis, SBC on 1,000 simulated genes | Complete, CPU (8 threads); SBC fails, recorded in `results/phase1/*_diag.json` |
| 2026-10-25T22:14Z | a53a8b6 + uncommitted Phase 1 code (committed in 7dc3628 and the following commit) | 6f05a534ac2f | 5 | Phase 1 with full-covariance posteriors, same stages | Complete, CPU (8 threads); Gate G1 passes, H1 holds |
| 2026-10-26T02:36Z | d69bdca | - | 0.2 | Pre-training routing simulation (section 7.3): prior-conditional maps, noise sweep, Phase 1 Fisher operating point, 4,000 simulated genes | Complete, CPU; results/design_sim/, docs/DESIGN_CHECK.md |
| 2026-10-26T03:06Z | 32f33c2 + uncommitted (committed in 00f0918) | - | 6.1 | NUTS reference posteriors, 200 chr7 genes x 3 arms (KIN, SS-EI, KIN-1T), 4 chains x 1000 draws, CPU (4 cores) | Complete; max R-hat 1.0066, min bulk ESS 2638, 0 divergences |
| 2026-10-26T03:08Z | 32f33c2 + uncommitted (committed in 00f0918) | - | 0.7 | Motif baseline (JASPAR 2024, TargetScan 8) on both variant sets, CPU | Complete; results/phase3/motif_baseline/summary.json |
| 2026-10-26T03:10Z | b389317 | 77cae8bb54d7 | 0.2 x 60 | First launch of the Phase 2 grid | Every job failed at the first step (float64 cast of an MPS tensor); fixed in the next commit, no results |
| 2026-10-26T03:25Z | 6ffa06c | 77cae8bb54d7 | 7.5 | Timing run: KIN, seed 0, fold0 (5,311 training genes, MPS) | Complete; 20 epochs (best 9), 6.7 min training at 16 to 22 s per epoch, 0.7 min scoring. Projection: 60 runs x 7.5 min = 7.5 h, which ends after the 03:30 PDT launch cutoff; KIN-1T (the last group) is cut first |
| 2026-10-26T03:06Z | b389317 (uncommitted at the time) | - | 1.5 | Encode promoter and transcript inputs for 19,111 genes (CPU) | Complete; data/processed/inputs |
| 2026-10-26T03:26Z | 960e850 (uncommitted at the time) | - | 7.7 | Sequence-overlap leakage scan, 32-mers, five folds (CPU) | Complete; 298 3'UTR variants flagged, no MPRA element overlap |
| 2026-10-26T03:33Z | 6ffa06c (training code) | - | 4.8 | Phase 2: KIN, seed 0, fold1 | Failed at a non-finite likelihood step (old code); discarded and rerun |
| 2026-10-26T03:37Z | 7edc6b7 | 7576f2c496a0 | 7.5 | Phase 2: KIN, seed 0, fold2, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T03:45Z | 6ffa06c (training code) | - | 4.6 | Phase 2: KIN, seed 0, fold3 | Failed at a non-finite likelihood step (old code); discarded and rerun |
| 2026-10-26T03:50Z | f95c597 | d41e13d21509 | 7.0 | Phase 2: KIN, seed 0, fold1, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T03:57Z | f95c597 | 23efbcc9bfcf | 5.6 | Phase 2: KIN, seed 0, fold4, train and score held-out chromosomes (MPS) | Complete; 16 epochs, best 5 |
| 2026-10-26T04:02Z | f95c597 | 2f14c4e9b7a4 | 6.5 | Phase 2: SS-EI, seed 0, fold0, train and score held-out chromosomes (MPS) | Complete; 19 epochs, best 8 |
| 2026-10-26T04:09Z | f95c597 | f6803df6b867 | 6.2 | Phase 2: KIN, seed 0, fold3, train and score held-out chromosomes (MPS) | Complete; 18 epochs, best 7 |
| 2026-10-26T04:15Z | f95c597 | 8025b3c748e7 | 6.9 | Phase 2: SS-EI, seed 0, fold1, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T04:22Z | f95c597 | de6d7db51d1d | 7.2 | Phase 2: SS-EI, seed 0, fold2, train and score held-out chromosomes (MPS) | Complete; 21 epochs, best 10 |
| 2026-10-26T04:29Z | 7b5175f | c60e3dae98ba | 7.4 | Phase 2: SS-EI, seed 0, fold3, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T04:36Z | cb455c9 | 0f6a1db60b57 | 7.7 | Phase 2: SS-EI, seed 0, fold4, train and score held-out chromosomes (MPS) | Complete; 23 epochs, best 12 |
| 2026-10-26T04:44Z | cb455c9 | 18c103645d01 | 6.8 | Phase 2: SS-E, seed 0, fold0, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T04:51Z | cb455c9 | f5e0651e00b7 | 6.6 | Phase 2: SS-E, seed 0, fold1, train and score held-out chromosomes (MPS) | Complete; 19 epochs, best 8 |
| 2026-10-26T04:57Z | cb455c9 | 8e9fbb2386f0 | 8.8 | Phase 2: SS-E, seed 0, fold2, train and score held-out chromosomes (MPS) | Complete; 26 epochs, best 15 |
| 2026-10-26T05:06Z | cb455c9 | 9e0b37d5fdf0 | 7.4 | Phase 2: SS-E, seed 0, fold3, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T05:14Z | cb455c9 | fd3e0465e6ae | 6.1 | Phase 2: SS-E, seed 0, fold4, train and score held-out chromosomes (MPS) | Complete; 18 epochs, best 7 |
| 2026-10-26T05:20Z | cb455c9 | d147c3d380ca | 8.1 | Phase 2: KIN-SHUF, seed 0, fold0, train and score held-out chromosomes (MPS) | Complete; 24 epochs, best 13 |
| 2026-10-26T05:28Z | d090809 | 52f28d222d97 | 7.2 | Phase 2: KIN-SHUF, seed 0, fold1, train and score held-out chromosomes (MPS) | Complete; 21 epochs, best 10 |
| 2026-10-26T05:35Z | d090809 | 249556c98fe9 | 6.3 | Phase 2: KIN-SHUF, seed 0, fold2, train and score held-out chromosomes (MPS) | Complete; 18 epochs, best 7 |
| 2026-10-26T05:41Z | d090809 | c206b647cdaf | 7.5 | Phase 2: KIN-SHUF, seed 0, fold3, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T05:49Z | d090809 | 567b5db32a2e | 6.2 | Phase 2: KIN-SHUF, seed 0, fold4, train and score held-out chromosomes (MPS) | Complete; 18 epochs, best 7 |
| 2026-10-26T05:55Z | d090809 | 082b281e3dac | 7.2 | Phase 2: KIN, seed 1, fold0, train and score held-out chromosomes (MPS) | Complete; 21 epochs, best 10 |
| 2026-10-26T06:02Z | f9cbecc | 2c3d9feaf16f | 6.0 | Phase 2: KIN, seed 1, fold1, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T06:08Z | f9cbecc | b2e2abc05312 | 7.9 | Phase 2: KIN, seed 1, fold2, train and score held-out chromosomes (MPS) | Complete; 23 epochs, best 12 |
| 2026-10-26T06:16Z | f9cbecc | f83a0d406a99 | 6.0 | Phase 2: KIN, seed 1, fold3, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T06:22Z | f9cbecc | df4983eb3ade | 5.9 | Phase 2: KIN, seed 1, fold4, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T06:28Z | f9cbecc | 67a47625a506 | 5.3 | Phase 2: SS-EI, seed 1, fold0, train and score held-out chromosomes (MPS) | Complete; 15 epochs, best 4 |
| 2026-10-26T06:33Z | f9cbecc | bf01ddedfce9 | 7.0 | Phase 2: SS-EI, seed 1, fold1, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T06:40Z | f9cbecc | 5893f620ae5c | 7.6 | Phase 2: SS-EI, seed 1, fold2, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T06:48Z | f9cbecc | 0df4b29d6832 | 6.6 | Phase 2: SS-EI, seed 1, fold3, train and score held-out chromosomes (MPS) | Complete; 19 epochs, best 8 |
| 2026-10-26T06:54Z | f9cbecc | ee3e6a125170 | 7.5 | Phase 2: SS-EI, seed 1, fold4, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T07:02Z | f9cbecc | 32d281429676 | 5.3 | Phase 2: SS-E, seed 1, fold0, train and score held-out chromosomes (MPS) | Complete; 15 epochs, best 4 |
| 2026-10-26T07:07Z | f9cbecc | 8ae260feb91d | 7.3 | Phase 2: SS-E, seed 1, fold1, train and score held-out chromosomes (MPS) | Complete; 21 epochs, best 10 |
| 2026-10-26T07:15Z | f9cbecc | 134d2e2b9827 | 7.6 | Phase 2: SS-E, seed 1, fold2, train and score held-out chromosomes (MPS) | Complete; 22 epochs, best 11 |
| 2026-10-26T07:22Z | f9cbecc | 99e2154a9cd2 | 8.1 | Phase 2: SS-E, seed 1, fold3, train and score held-out chromosomes (MPS) | Complete; 24 epochs, best 13 |
| 2026-10-26T07:30Z | f9cbecc | 873ba789eb42 | 8.1 | Phase 2: SS-E, seed 1, fold4, train and score held-out chromosomes (MPS) | Complete; 24 epochs, best 13 |
| 2026-10-26T07:38Z | f9cbecc | 00477e7ee678 | 6.9 | Phase 2: KIN, seed 2, fold0, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T07:45Z | f9cbecc | b9b53e548b78 | 6.7 | Phase 2: KIN, seed 2, fold1, train and score held-out chromosomes (MPS) | Complete; 19 epochs, best 8 |
| 2026-10-26T07:52Z | f9cbecc | 1968027754b8 | 6.0 | Phase 2: KIN, seed 2, fold2, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T07:58Z | f9cbecc | c580e86016e3 | 5.9 | Phase 2: KIN, seed 2, fold3, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T08:04Z | f9cbecc | a00a6c44b1a2 | 5.3 | Phase 2: KIN, seed 2, fold4, train and score held-out chromosomes (MPS) | Complete; 15 epochs, best 4 |
| 2026-10-26T08:09Z | f9cbecc | 0eaf7e621c61 | 5.4 | Phase 2: SS-EI, seed 2, fold0, train and score held-out chromosomes (MPS) | Complete; 15 epochs, best 4 |
| 2026-10-26T08:15Z | f9cbecc | acdfecded4cc | 5.4 | Phase 2: SS-EI, seed 2, fold1, train and score held-out chromosomes (MPS) | Complete; 15 epochs, best 4 |
| 2026-10-26T08:20Z | f9cbecc | 7c78d9f4327b | 7.9 | Phase 2: SS-EI, seed 2, fold2, train and score held-out chromosomes (MPS) | Complete; 23 epochs, best 12 |
| 2026-10-26T08:28Z | f9cbecc | 85af30b7ff76 | 5.6 | Phase 2: SS-EI, seed 2, fold3, train and score held-out chromosomes (MPS) | Complete; 16 epochs, best 5 |
| 2026-10-26T08:33Z | f9cbecc | 65983b33a737 | 7.2 | Phase 2: SS-EI, seed 2, fold4, train and score held-out chromosomes (MPS) | Complete; 21 epochs, best 10 |
| 2026-10-26T08:41Z | f9cbecc | 1d5d0e9b734b | 6.9 | Phase 2: SS-E, seed 2, fold0, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T08:47Z | f9cbecc | 16901715487e | 8.2 | Phase 2: SS-E, seed 2, fold1, train and score held-out chromosomes (MPS) | Complete; 24 epochs, best 13 |
| 2026-10-26T08:56Z | f9cbecc | 71e1e8ef9c57 | 9.2 | Phase 2: SS-E, seed 2, fold2, train and score held-out chromosomes (MPS) | Complete; 27 epochs, best 16 |
| 2026-10-26T09:05Z | f9cbecc | 1b4bea86849a | 5.9 | Phase 2: SS-E, seed 2, fold3, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T09:11Z | f9cbecc | 29743591bd1c | 4.9 | Phase 2: SS-E, seed 2, fold4, train and score held-out chromosomes (MPS) | Complete; 14 epochs, best 3 |
| 2026-10-26T09:16Z | f9cbecc | f4e6b6fba220 | 8.1 | Phase 2: KIN-1T, seed 0, fold0, train and score held-out chromosomes (MPS) | Complete; 24 epochs, best 13 |
| 2026-10-26T09:24Z | 070bd3b (dirty) | 3f25f309cf66 | 6.0 | Phase 2: KIN-1T, seed 0, fold1, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T09:30Z | 070bd3b (dirty) | 9d32c9c4c1f8 | 6.9 | Phase 2: KIN-1T, seed 0, fold2, train and score held-out chromosomes (MPS) | Complete; 20 epochs, best 9 |
| 2026-10-26T09:37Z | 070bd3b (dirty) | c5196b116a96 | 6.5 | Phase 2: KIN-1T, seed 0, fold3, train and score held-out chromosomes (MPS) | Complete; 19 epochs, best 8 |
| 2026-10-26T09:43Z | 070bd3b (dirty) | 375d4989c54d | 5.9 | Phase 2: KIN-1T, seed 0, fold4, train and score held-out chromosomes (MPS) | Complete; 17 epochs, best 6 |
| 2026-10-26T09:50Z | 070bd3b + evaluation fixes (committed with the results) | - | 1 | Phase 3 evaluation (2,000 two-level bootstrap resamples), gene-level metrics, figures and tables (CPU) | Complete; results/phase3, results/figures |
| 2026-10-26T10:49Z | f294479 (dirty: `sk.eval.rescore` uncommitted) | - | 7.4 | Post hoc: rescore the 3'UTR reporter variants of all 55 runs from their saved models with the fixed `score.utr_reporter` (MPS) | Complete; results/phase3/utr_reporter_rescore.json |
| 2026-10-26T10:56Z | f294479 (dirty: KIN-SHUF pairing and routing diagnostic uncommitted) | - | 3 | Post hoc: Phase 3 evaluation, routing diagnostic (MPS), tables and figures | Complete; committed in f8d17c6 |
| 2026-10-26T11:33Z | b2fd190 (dirty: `sk.eval.gene_diagnostics` uncommitted) | - | 3 | Post hoc: gene-level decay and amortization diagnostics (MPS and CPU) | Complete; results/phase3/gene_diagnostics.json |
