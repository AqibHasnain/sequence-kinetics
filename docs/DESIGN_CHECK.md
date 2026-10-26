# Design check (section 7.3)

Status: all five items pass. Items 2 and 5 were checked on the Phase 2 network and KIN-SHUF
construction before the grid was trained. The routing simulation below finds that the SS-EI clause of
H2 cannot pass at zero estimation noise for pure single-rate effects, and that at Phase 1 assay
precision it fails; it can pass only through the noise structure the sequence network sets.

Script: `src/sk/eval/design_sim.py` (`python -m sk.eval.design_sim`, 9 s on CPU). Outputs:
`results/design_sim/design_sim.json`, `routing_vs_noise.png`, `kin_minus_ssei_anisotropic.png`.

## Item 1: directions each arm's likelihood constrains

Verified. Rank of the Jacobian of the expected observations with respect to (a, b, c) at 100 draws
from the KIN-fitted prior, median training-gene covariates (`results/phase1/identifiability_full.json`):

| Arm | Rank (count of 100) | Constrained directions | Null space |
|---|---|---|---|
| SS-E | 1 (100) | log(exp(a - b) + exp(a - c)) | two directions |
| SS-EI, SS-EI-EISA | 2 (100) | a - b, a - c | (1, 1, 1) |
| KIN | 3 (100) | a, b, c | none |
| KIN-1T | 3 (100) | a, b, c | none |

SS-E constrains a - c only approximately. The exonic mean is s L (u* + m*), so its gradient in
(a, b, c) is (1, -w, -(1 - w)), with w the pre-mRNA share of exonic signal. Over the simulated
genes w has median 0.10 (quartiles 0.05, 0.19). This changes SS-E routing (next section).

KIN-1T has full rank but weak c: its largest contraction of c on the Phase 1 grid is 0.79 and it
reaches no observable half-life range. KIN's range is 0.19 h to 13.5 h.

## Item 2: no arm's routing is fixed by architecture

Verified. Both segments pass through the shared tower, are pooled separately and concatenated
before one MLP that outputs all three means, so no head reads one segment only
(`src/sk/model/network.py`). `tests/test_model.py::test_both_segments_reach_every_rate_head`
changes one 3'UTR base (300 nt from the 3' end) of a random input to a randomly initialized
network and finds a non-zero delta a, and changes one promoter base (TSS - 188) and finds a
non-zero delta c.

## Item 3: the motif baseline routes by construction

Verified by definition. The motif baseline (section 4.5) routes promoter variants to a and 3'UTR
variants to c by region, so its routing accuracy is 1 on both sets by construction. It is
excluded from H2 and used only for the H4 magnitude comparison.

## Item 4: SS-EI-EISA differs from SS-EI only in scoring

Verified by construction. SS-EI-EISA trains nothing: it rescores the SS-EI models
(DECISIONS, cross-validation grid entry). Its deltas are a fixed linear map of SS-EI's predicted
deltas in a - b and a - c: delta b = 0, delta a = delta(a - b), delta c = delta(a - b) -
delta(a - c). The SS-EI prior along (1, 1, 1) does not enter, since it changes neither
identified delta. Any routing difference between the two arms is the delta b = 0 assumption
replacing SS-EI's prior conditional mean along (1, 1, 1). `design_sim.py` implements the arm as
this map applied to the SS-EI map (`EISA`).

## Item 5: KIN-SHUF preserves labeled-fraction marginals within expression deciles

Verified. `shuffle_labeling` (`src/sk/infer/amortized.py`) permutes whole labeling blocks
(TimeLapse-seq fractions, SDs, mask and background; TT-seq labeled counts, matched totals and bin
lengths) among genes of the same whole-cell expression decile, separately for gradient and
early-stopping genes, and leaves steady-state counts and lengths in place.
`tests/test_model.py::test_kin_shuf_preserves_labeling_marginals_within_expression_deciles`
checks that every source gene is in the same decile, that the multiset of labeled fractions in
each decile is unchanged, that blocks move as units, that steady state is untouched, and that
more than 80% of genes receive another gene's block.

## Prior-implied routing maps

Along a direction the likelihood does not constrain, the ELBO-optimal posterior mean is the prior
conditional mean given the constrained components, so an SS arm's predicted deltas are
M delta with M = Sigma P' (P Sigma P')^-1 P, P the constrained rows and Sigma the arm's prior
covariance. Predicted deltas for a unit change in one rate, with P exact (`analytic_maps` in the
JSON). All verified numerically:

| Arm and prior | delta a = 1 | delta b = 1 | delta c = 1 |
|---|---|---|---|
| SS-EI, independent prior (adopted) | (0.67, -0.33, -0.33): a, margin 0.25 | (-0.33, 0.67, -0.33): b, margin 0.25 | (-0.33, -0.33, 0.67): c, margin 0.25 |
| SS-EI, KIN-fitted prior | (0.99, -0.01, -0.01): a, margin 0.97 | (-0.49, 0.51, -0.49): b, margin 0.02 | (-0.503, -0.503, 0.497): a and b tie |
| SS-EI, SS-EI-fitted prior | (0.95, -0.05, -0.05): a | (-0.04, 0.96, -0.04): b | (-0.90, -0.90, 0.10): a and b tie |
| SS-E (a - c exact), independent prior | (0.5, 0, -0.5): a and c tie | (0, 0, 0) | (-0.5, 0, 0.5): a and c tie |

The adopted independent prior makes the SS-EI map the orthogonal projection that removes the
(1, 1, 1) component: every single-rate change routes correctly with margin 0.25. Under any SS-EI
prior a pure decay change gives delta a = delta b exactly, so it routes to c or ties a and b.

SS-E under the Phase 1 exonic likelihood: the constrained row is (1, -0.10, -0.90), and under the
independent SS-E prior the map sends every delta to a multiple of Sigma n = (1, -0.13, -0.87).
SS-E therefore routes every variant to a, at any noise level: promoter-set accuracy 1, 3'UTR-set
accuracy 0, pooled 0.5, no ties. The DECISIONS statement that SS-E ties a and c holds under the
a - c approximation of the exonic signal; with that approximation and ties counted as wrong,
SS-E pooled accuracy is 0. In both cases the SS-E side of H2 is fixed by the design.

## Routing simulation

**Setup.** 4,000 genes drawn from the KIN-fitted prior with median training-gene covariates;
one variant per gene per scenario, random sign, draws paired across arms. Correct rate: a for the
promoter set, c for the 3'UTR set. Routing is the largest |delta|; ties count as wrong and are
reported separately. Pooled accuracy weights the two sets equally.

Scenarios (unit direction x sign x effect size):

- promoter set: pure synthesis (`syn`); synthesis with delta b = rho x delta a (`syn_proc:rho`,
  rho from -0.75 to 0.9); `mix_promoter`, 70% pure and 30% with rho ~ U(-0.5, 0.5);
  `coupled_promoter:f`, a share f in {0.1, 0.2, 0.3} with rho ~ U(0.5, 1) and the rest pure.
- 3'UTR set: pure decay (`dec`); decay with delta a = kappa x delta c (`dec_syn:kappa`, kappa in
  {-0.5, -0.25, 0.25, 0.5}); `mix_utr3`, 70% pure and 30% with kappa ~ U(-0.5, 0.5).

The mixture shares are assumptions; no measurement fixes them.

Two noise regimes:

- **Ideal.** Each arm recovers its constrained components exactly up to isotropic Gaussian noise
  of SD r x |true delta| per constrained orthonormal direction, then applies its prior map. r
  is the unknown the sequence network sets, swept from 0 to 2. Because the map is linear and the
  noise scales with the effect, accuracy does not depend on effect size in this regime.
  KIN and KIN-1T coincide here (both full rank).
- **Phase 1 operating point.** Posterior-mean deltas are (Sigma^-1 + I)^-1 I delta plus the
  sampling noise of measuring each allele once, with I the Phase 1 Fisher information at the
  gene and effect sizes resampled from the significant MPRA variants (|log2| > 0.2, FDR < 0.05;
  natural-log quartiles 0.21, 0.32, 0.56 promoter and 0.20, 0.28, 0.46 3'UTR). Without the
  sampling noise, per-gene shrinkage alone never misroutes a KIN variant (accuracy 1.00 on every
  scenario); KIN-1T drops to 0.98 at rho = 0.9.

The Phase 1 noise is strongly anisotropic. As a ratio to the median significant effect (0.30 in
natural log), the allele-difference noise SD is 0.37 per steady-state-constrained direction for
both SS-EI and KIN, 0.99 along (1, 1, 1) for KIN and 3.2 for KIN-1T. The isotropic equivalent
for KIN, r = 0.65, is the dotted line in `routing_vs_noise.png`.

**Accuracy per scenario and arm** (ideal at r = 0, 0.5, 1, and the Phase 1 operating point;
Monte Carlo SE about 0.008). Ties are non-zero only where marked.

| Scenario (set) | Noise | SS-E | SS-EI | SS-EI-EISA | KIN | KIN-1T | SS-EI, KIN prior |
|---|---|---|---|---|---|---|---|
| syn (promoter) | r = 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | r = 0.5 | 1.00 | 0.60 | 0.74 | 0.78 | 0.78 | 0.89 |
| | r = 1 | 1.00 | 0.41 | 0.58 | 0.49 | 0.49 | 0.74 |
| | Phase 1 | 1.00 | 0.67 | 0.80 | 0.72 | 0.58 | 0.92 |
| syn_proc:0.6 (promoter) | r = 0 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| | r = 0.5 | 1.00 | 0.36 | 0.45 | 0.60 | 0.60 | 0.71 |
| | Phase 1 | 1.00 | 0.31 | 0.36 | 0.67 | 0.51 | 0.80 |
| mix_promoter | r = 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | r = 0.5 | 1.00 | 0.60 | 0.74 | 0.76 | 0.76 | 0.88 |
| | Phase 1 | 1.00 | 0.66 | 0.77 | 0.72 | 0.60 | 0.91 |
| dec (3'UTR) | r = 0 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 (all a-b ties) |
| | r = 0.5 | 0.00 | 0.61 | 0.76 | 0.79 | 0.79 | 0.00 |
| | r = 1 | 0.00 | 0.42 | 0.58 | 0.49 | 0.49 | 0.00 |
| | Phase 1 | 0.00 | 0.77 | 0.84 | 0.69 | 0.56 | 0.00 |
| dec_syn:0.5 (3'UTR) | r = 0 | 0.00 | 0.00 (all a-c ties) | 1.00 | 1.00 | 1.00 | 0.00 |
| | Phase 1 | 0.00 | 0.48 | 0.85 | 0.65 | 0.55 | 0.00 |
| mix_utr3 | r = 0 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 (70% ties) |
| | r = 0.5 | 0.00 | 0.58 | 0.72 | 0.75 | 0.75 | 0.00 |
| | Phase 1 | 0.00 | 0.77 | 0.84 | 0.70 | 0.57 | 0.00 |

At zero noise SS-EI routes synthesis with processing correctly for -1 < rho < 0.5, ties a and c at
rho = 0.5 and routes to c above it; it routes decay with a synthesis component correctly for
kappa < 0.5. KIN is correct for |rho| < 1 and |kappa| < 1. SS-EI-EISA fails at rho >= 0.5
(delta a = (1 - rho) d against delta c = -rho d) and is unaffected by kappa.

**H2 differences** (pooled, ideal sweep; the clause intervals are on the 0.1 grid of r):

| Pairing | KIN - SS-EI at r = 0 / 0.5 / 1 / 2 | SS-EI clause (>= 0.05) holds for r in | KIN - SS-E at r = 0 / 0.5 / 1 | SS-E clause (>= 0.15) holds for r <= | Phase 1: KIN - SS-EI | Phase 1: KIN - SS-E |
|---|---|---|---|---|---|---|
| syn + dec | 0.00 / 0.17 / 0.07 / 0.02 | 0.3 to 1.1 | 0.50 / 0.28 / -0.01 | 0.6 | -0.02 | +0.21 |
| mix_promoter + mix_utr3 | 0.00 / 0.16 / 0.07 / 0.02 | 0.2 to 1.3 | 0.50 / 0.26 / -0.02 | 0.6 | -0.00 | +0.21 |
| syn_proc:0.25 + dec | 0.00 / 0.19 / 0.08 / 0.03 | 0.2 to 1.4 | 0.50 / 0.26 / -0.02 | 0.6 | +0.03 | +0.23 |
| syn_proc:0.6 + dec | 0.50 / 0.20 / 0.08 / 0.02 | 0.0 to 1.2 | 0.50 / 0.19 / -0.04 | 0.5 | +0.14 | +0.18 |
| syn + dec_syn:0.5 | 0.50 / 0.21 / 0.08 / 0.02 | 0.0 to 1.3 | 0.50 / 0.22 / -0.04 | 0.5 | +0.11 | +0.19 |
| coupled_promoter:0.1 + mix_utr3 | 0.05 / 0.18 / 0.08 / 0.03 | 0.2 to 1.3 | 0.50 / 0.26 / -0.01 | 0.6 | +0.00 | +0.22 |
| coupled_promoter:0.2 + mix_utr3 | 0.10 / 0.18 / 0.08 / 0.02 | 0.0 to 1.4 | 0.50 / 0.24 / -0.02 | 0.6 | +0.02 | +0.20 |

The isotropic sweep gives KIN noise in three directions and SS-EI in two, which is the diagonal
r_ss = r_kin of `kin_minus_ssei_anisotropic.png`. Off the diagonal, for pure and mixed sets:
KIN - SS-EI is 0.00 whenever r_ss <= 0.1 and r_kin <= 0.25, negative whenever r_kin >= 0.5 and
r_ss <= 0.1, and >= 0.05 only when r_ss >= 0.25 and r_kin <= r_ss. Around the Phase 1 point
(r_ss 0.37, r_kin 0.99) the grid gives -0.14 at r_ss = 0.25 and 0.04 to 0.05 at r_ss = 0.5, and
the Phase 1 simulation gives -0.02.

## Verdict on H2

- **SS-E clause.** Fixed by design on the SS-E side: SS-E pooled accuracy is 0.5 at every noise
  level (0 if the exonic signal is approximated as a - c and ties count as wrong). The clause is
  KIN pooled accuracy >= 0.65, a test of KIN's absolute accuracy. It holds at zero noise
  (KIN = 1), for isotropic noise r <= 0.6, and at the Phase 1 operating point (+0.21). Data
  decide it.
- **SS-EI clause.** Fails by design at zero noise unless at least 10% of one set's variants
  carry a secondary change larger than half the primary one in the sign that flips SS-EI
  (rho > 0.5 or kappa > 0.5): the zero-noise difference is half the share of such variants per
  set. Under the adopted independent prior SS-EI routes every pure single-rate change correctly,
  so for pure and mixed sets the clause measures how the network's errors split between the
  steady-state directions and the labeling direction: it passes when r_ss >= 0.25 and
  r_kin <= r_ss, and fails at Phase 1 assay precision, where r_kin is 2.7 times r_ss. A pass or fail is decided by data, but by noise structure and the prevalence of
  coupled secondary effects; it does not measure the identifiability gain that H2 is meant to
  test. This is inferred from the simulation and holds for the adopted SS prior; under the
  KIN-fitted prior SS-EI routes no decay variant correctly and the clause would pass by design.
- **SS-EI-EISA** has the highest pooled accuracy of the H2 and H3 arms at the Phase 1 operating
  point on every pairing except those where every promoter variant has rho >= 0.5 (pure sets:
  0.82 against KIN 0.71), which bears on H3.

The SS-EI clause of H2 as written needed a decision before Phase 2 training. H2 and H3 were
amended on this basis before any variant was scored (DECISIONS, "H2 and H3 amended before any
variant is scored").
