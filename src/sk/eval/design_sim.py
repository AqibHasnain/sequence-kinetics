"""Pre-training routing simulation: can H2 be decided by data, or is it fixed by the design?

Each arm's predicted variant deltas are a linear map of the true deltas. Along directions its
likelihood constrains, an arm recovers the true delta plus estimation noise; along the others
the ELBO-optimal posterior mean is the prior conditional mean given the constrained components.
Two operating regimes are simulated on genes drawn from the KIN-fitted prior:

  ideal   the constrained components are known up to isotropic noise of SD r x |delta|, swept
          over r (the noise a sequence network will set). The map is
          Sigma P' (P Sigma P')^-1 P, with P spanning the row space of the arm's Jacobian.
  fisher  the Phase 1 per-gene Fisher information I at Phase 1 depth: posterior-mean deltas
          are (Sigma^-1 + I)^-1 I delta plus the sampling noise of measuring each allele once,
          with effect sizes drawn from the significant MPRA effects.

Routing is the rate with the largest |delta| (section 6.3); ties count as wrong and are
counted separately. Outputs go to results/design_sim/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sk.data import kinetic_inputs  # noqa: E402
from sk.eval.phase1 import RUN, _median_template, _noise_module, _ttseq_depth, _with_theta  # noqa: E402
from sk.kinetics import fisher  # noqa: E402
from sk.kinetics.observations import DESIGNS  # noqa: E402
from sk.paths import PROCESSED, RESULTS  # noqa: E402
from sk.utils.logging import get_logger  # noqa: E402
from sk.utils.manifest import write_json  # noqa: E402
from sk.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

OUT = RESULTS / "design_sim"
DTYPE = torch.float64
SEED = 0
N_GENES = 4000
NOISE_RATIOS = np.round(np.linspace(0.0, 2.0, 21), 2)
TABLE_RATIOS = (0.0, 0.25, 0.5, 1.0, 2.0)
TIE_RTOL = 1e-9
RATE = {"promoter": 0, "utr3": 2}
H2_SSEI, H2_SSE = 0.05, 0.15

# Arms scored, with the design whose likelihood they use and the prior that fills the
# unconstrained directions. SS-EI-EISA rescores SS-EI and has no map of its own.
ARMS = {
    "SS-E": ("SS-E", "independent"),
    "SS-EI": ("SS-EI", "independent"),
    "SS-EI-EISA": ("SS-EI", "independent"),
    "KIN": ("KIN", "KIN"),
    "KIN-1T": ("KIN-1T", "KIN-1T"),
    "SS-EI | KIN prior": ("SS-EI", "KIN"),
}
H2_ARMS = ("SS-E", "SS-EI", "KIN")

# Identified combinations named by the SS-arm prior decision; the independent prior fits the
# population covariance of these and gives the orthogonal complement variance 1.
SS_IDENTIFIED = {
    "SS-E": np.array([[1.0, 0.0, -1.0]]),
    "SS-EI": np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]),
}
# EISA: delta b = 0, delta a = delta(a - b), delta c = delta(a - b) - delta(a - c).
EISA = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, -1.0]]) @ SS_IDENTIFIED["SS-EI"]


# ------------------------------------------------------------------- linear algebra


def orthonormal_complement(rows: np.ndarray) -> np.ndarray:
    """Orthonormal basis [3, 3 - k] of the orthogonal complement of the row space of rows."""
    _, _, vt = np.linalg.svd(rows)
    return vt[rows.shape[0] :].T


def independent_prior_cov(identified: np.ndarray, fitted_cov: np.ndarray) -> np.ndarray:
    """Prior covariance in (a, b, c) under which the orthonormal complement of the identified
    rows is independent of the identified components, with unit variance per direction."""
    comp = orthonormal_complement(identified)
    a = np.vstack([identified, comp.T])
    k = identified.shape[0]
    block = np.eye(3)
    block[:k, :k] = identified @ fitted_cov @ identified.T
    a_inv = np.linalg.inv(a)
    return a_inv @ block @ a_inv.T


def conditional_map(prior_cov: np.ndarray, identified: np.ndarray) -> np.ndarray:
    """3 x 3 map from a true delta to the posterior-mean delta when the identified rows are
    known exactly and every other direction takes its prior conditional mean."""
    p = identified
    return prior_cov @ p.T @ np.linalg.solve(p @ prior_cov @ p.T, p)


def identified_rows(info: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Orthonormal rows spanning the range of a Fisher information matrix [3, 3]."""
    w, v = np.linalg.eigh(info)
    return v[:, w > rtol * w.max()].T


# --------------------------------------------------------------------------- routing


def route(delta: np.ndarray, correct: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-variant (correct, tie) for deltas [n, 3]. A tie between the two largest |delta|
    counts as incorrect whichever rates are involved."""
    mag = np.abs(delta)
    top2 = np.sort(mag, axis=1)[:, -2:]
    tie = top2[:, 1] - top2[:, 0] <= TIE_RTOL * np.maximum(top2[:, 1], 1e-300)
    ok = (mag.argmax(1) == correct) & ~tie
    return ok, tie


# ------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    name: str
    variant_set: str  # "promoter" (correct rate a) or "utr3" (correct rate c)
    description: str

    def directions(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Unit-scale true deltas [n, 3] before the effect size and sign are applied."""
        kind, _, p = self.name.partition(":")
        d = np.zeros((n, 3))
        if kind == "syn":
            d[:, 0] = 1.0
        elif kind == "syn_proc":
            d[:, 0], d[:, 1] = 1.0, float(p)
        elif kind == "dec":
            d[:, 2] = 1.0
        elif kind == "dec_syn":
            d[:, 2], d[:, 0] = 1.0, float(p)
        elif kind == "mix_promoter":
            # 70% pure synthesis, 30% with processing coupled at rho ~ U(-0.5, 0.5).
            d[:, 0] = 1.0
            coupled = rng.random(n) < 0.3
            d[coupled, 1] = rng.uniform(-0.5, 0.5, coupled.sum())
        elif kind == "mix_utr3":
            # 70% pure decay, 30% with a synthesis component kappa ~ U(-0.5, 0.5).
            d[:, 2] = 1.0
            mixed = rng.random(n) < 0.3
            d[mixed, 0] = rng.uniform(-0.5, 0.5, mixed.sum())
        elif kind == "coupled_promoter":
            # A share p of variants with processing coupled at rho ~ U(0.5, 1); the rest pure.
            d[:, 0] = 1.0
            coupled = rng.random(n) < float(p)
            d[coupled, 1] = rng.uniform(0.5, 1.0, coupled.sum())
        else:
            raise ValueError(self.name)
        return d


RHOS = (-0.75, -0.5, -0.25, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
KAPPAS = (-0.5, -0.25, 0.25, 0.5)
SCENARIOS = [
    Scenario("syn", "promoter", "pure synthesis"),
    *[Scenario(f"syn_proc:{r}", "promoter", f"synthesis, delta b = {r} x delta a") for r in RHOS],
    Scenario("mix_promoter", "promoter", "70% pure synthesis, 30% rho ~ U(-0.5, 0.5)"),
    *[
        Scenario(f"coupled_promoter:{f}", "promoter", f"{f:.0%} with rho ~ U(0.5, 1), rest pure")
        for f in (0.1, 0.2, 0.3)
    ],
    Scenario("dec", "utr3", "pure decay"),
    *[Scenario(f"dec_syn:{k}", "utr3", f"decay, delta a = {k} x delta c") for k in KAPPAS],
    Scenario("mix_utr3", "utr3", "70% pure decay, 30% kappa ~ U(-0.5, 0.5)"),
]
# (promoter scenario, 3'UTR scenario) pairs scored for H2 with equal set weighting.
PAIRINGS = [
    ("syn", "dec"),
    ("mix_promoter", "mix_utr3"),
    *[(f"syn_proc:{r}", "dec") for r in (0.25, 0.5, 0.6, 0.75)],
    *[("syn", f"dec_syn:{k}") for k in (0.25, 0.5)],
    *[(f"coupled_promoter:{f}", "mix_utr3") for f in (0.1, 0.2, 0.3)],
]


def mpra_effects_ln() -> dict[str, np.ndarray]:
    """|effect| of significant MPRA variants per set, in natural-log units."""
    out = {}
    for name, key in (("promoter_enhancer", "promoter"), ("utr3", "utr3")):
        t = pd.read_parquet(PROCESSED / "variants" / f"{name}.parquet")
        out[key] = np.abs(t.loc[t["significant"], "log2_effect"].to_numpy()) * math.log(2)
    return out


# ----------------------------------------------------------------------------- setup


def load_setup(n: int, rng: np.random.Generator) -> dict:
    """Gene draws from the KIN prior, each arm's prior, Fisher information and ideal map."""
    saved = torch.load(RUN / "fits_full.pt", weights_only=False)
    fits = saved["fits"]
    kin = fits["KIN"]
    data = kinetic_inputs.build(saved["gene_ids"], DTYPE)
    noise = _noise_module(kin["noise"])
    depth = _ttseq_depth(data, kin["q_mean"])
    torch.manual_seed(int(rng.integers(2**31)))
    theta = torch.distributions.MultivariateNormal(kin["prior_mean"], kin["prior_cov"]).sample((n,))
    template = _with_theta(_median_template(data, n), theta, depth)

    cov = {d: fits[d]["prior_cov"].numpy() for d in fits}
    priors = {
        "KIN": cov["KIN"],
        "KIN-1T": cov["KIN-1T"],
        "independent:SS-E": independent_prior_cov(SS_IDENTIFIED["SS-E"], cov["SS-E"]),
        "independent:SS-EI": independent_prior_cov(SS_IDENTIFIED["SS-EI"], cov["SS-EI"]),
    }
    info = {
        d: fisher.fisher_information(theta, template, noise, DESIGNS[d]).numpy()
        for d in ("SS-E", "SS-EI", "KIN", "KIN-1T")
    }
    arms = {}
    for arm, (design, prior) in ARMS.items():
        sigma = priors[f"independent:{design}" if prior == "independent" else prior]
        ideal = np.stack([conditional_map(sigma, identified_rows(i)) for i in info[design]])
        post = np.linalg.inv(np.linalg.inv(sigma)[None] + info[design])
        w, v = np.linalg.eigh(info[design])
        info_sqrt = v @ (np.sqrt(np.clip(w, 0, None))[..., None] * v.mT)
        arms[arm] = {
            "ideal": ideal,
            "fisher_gain": post @ info[design],
            # Sampling noise of the posterior-mean delta when each allele is measured once.
            "fisher_noise": math.sqrt(2) * post @ info_sqrt,
        }
        if arm == "SS-EI-EISA":
            for k in ("ideal", "fisher_gain", "fisher_noise"):
                arms[arm][k] = EISA @ arms[arm][k]
    return {"theta": theta.numpy(), "priors": priors, "info": info, "arms": arms}


# ----------------------------------------------------------------------- simulation


def simulate(setup: dict, effects: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Accuracy and tie share per scenario, arm and noise level, with paired draws across arms."""
    n = len(setup["theta"])
    out: dict[str, dict] = {}
    for sc in SCENARIOS:
        unit = sc.directions(rng, n)
        sign = rng.choice([-1.0, 1.0], n)[:, None]
        size = rng.choice(effects[sc.variant_set], n)[:, None]
        true = sign * unit
        eps = rng.standard_normal((n, 3))
        xi = rng.standard_normal((n, 3))
        norm = np.linalg.norm(true, axis=1, keepdims=True)
        correct = RATE[sc.variant_set]
        res: dict[str, dict] = {}
        for arm, m in setup["arms"].items():
            acc, tie = [], []
            for r in NOISE_RATIOS:
                pred = np.einsum("gij,gj->gi", m["ideal"], true + r * norm * eps)
                ok, t = route(pred, correct)
                acc.append(ok.mean())
                tie.append(t.mean())
            d = size * true
            pred = np.einsum("gij,gj->gi", m["fisher_gain"], d) + np.einsum(
                "gij,gj->gi", m["fisher_noise"], xi
            )
            ok_f, tie_f = route(pred, correct)
            # Noise-free Fisher map isolates per-gene shrinkage from sampling noise.
            ok_f0, tie_f0 = route(np.einsum("gij,gj->gi", m["fisher_gain"], d), correct)
            res[arm] = {
                "ideal_accuracy": np.array(acc),
                "ideal_tie": np.array(tie),
                "fisher_accuracy": float(ok_f.mean()),
                "fisher_tie": float(tie_f.mean()),
                "fisher_noiseless_accuracy": float(ok_f0.mean()),
                "fisher_noiseless_tie": float(tie_f0.mean()),
            }
        out[sc.name] = res
    return out


def pooled(sim: dict, prom: str, utr: str, arm: str, key: str):
    return 0.5 * (sim[prom][arm][key] + sim[utr][arm][key])


def h2_summary(sim: dict) -> dict:
    """Per pairing: pooled accuracy per arm, KIN - SS-EI and KIN - SS-E, and the noise ratios
    at which each clause holds."""
    out = {}
    for prom, utr in PAIRINGS:
        acc = {a: pooled(sim, prom, utr, a, "ideal_accuracy") for a in ARMS}
        fis = {a: pooled(sim, prom, utr, a, "fisher_accuracy") for a in ARMS}
        d_ei = acc["KIN"] - acc["SS-EI"]
        d_e = acc["KIN"] - acc["SS-E"]
        out[f"{prom} + {utr}"] = {
            "ideal": {
                "noise_ratio": NOISE_RATIOS.tolist(),
                "pooled_accuracy": {a: v.tolist() for a, v in acc.items()},
                "kin_minus_ssei": d_ei.tolist(),
                "kin_minus_sse": d_e.tolist(),
                "ssei_clause_holds_at": NOISE_RATIOS[d_ei >= H2_SSEI].tolist(),
                "sse_clause_holds_at": NOISE_RATIOS[d_e >= H2_SSE].tolist(),
            },
            "fisher": {
                "pooled_accuracy": fis,
                "kin_minus_ssei": fis["KIN"] - fis["SS-EI"],
                "kin_minus_sse": fis["KIN"] - fis["SS-E"],
            },
        }
    return out


ANISO_RATIOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
ANISO_PAIRINGS = (("syn", "dec"), ("mix_promoter", "mix_utr3"))


def anisotropic_grid(setup: dict, rng: np.random.Generator) -> dict:
    """KIN - SS-EI and KIN - SS-E with separate noise ratios in the plane steady state
    constrains (r_ss) and along (1, 1, 1), which only labeling constrains (r_kin).

    Phase 1 shows the two are far apart under KIN, so the isotropic sweep is one diagonal of
    this grid.
    """
    u = np.full(3, 1 / math.sqrt(3))
    along, plane = np.outer(u, u), np.eye(3) - np.outer(u, u)
    by_name = {s.name: s for s in SCENARIOS}
    n = len(setup["theta"])
    out = {}
    for prom, utr in ANISO_PAIRINGS:
        draws = {}
        for name in (prom, utr):
            sc = by_name[name]
            true = rng.choice([-1.0, 1.0], n)[:, None] * sc.directions(rng, n)
            norm = np.linalg.norm(true, axis=1, keepdims=True)
            e_ss = norm * (rng.standard_normal((n, 3)) @ plane)
            e_k = norm * (rng.standard_normal((n, 3)) @ along)
            draws[name] = (true, e_ss, e_k, RATE[sc.variant_set])
        d_ei = np.zeros((len(ANISO_RATIOS), len(ANISO_RATIOS)))
        d_e = np.zeros_like(d_ei)
        for i, r_ss in enumerate(ANISO_RATIOS):
            for j, r_k in enumerate(ANISO_RATIOS):
                acc = {}
                for arm in H2_ARMS:
                    m = setup["arms"][arm]["ideal"]
                    per_set = []
                    for true, e_ss, e_k, correct in draws.values():
                        pred = np.einsum("gij,gj->gi", m, true + r_ss * e_ss + r_k * e_k)
                        per_set.append(route(pred, correct)[0].mean())
                    acc[arm] = float(np.mean(per_set))
                d_ei[i, j] = acc["KIN"] - acc["SS-EI"]
                d_e[i, j] = acc["KIN"] - acc["SS-E"]
        out[f"{prom} + {utr}"] = {
            "rows_r_ss": list(ANISO_RATIOS),
            "cols_r_kin": list(ANISO_RATIOS),
            "kin_minus_ssei": d_ei.round(4).tolist(),
            "kin_minus_sse": d_e.round(4).tolist(),
        }
    return out


def kin_noise_split(setup: dict, effects: dict[str, np.ndarray]) -> dict:
    """SD of the Phase 1 allele-difference noise of KIN and SS-EI along (1, 1, 1) and in the
    orthogonal plane, as ratios to the median significant MPRA effect."""
    u = np.full(3, 1 / math.sqrt(3))
    plane = np.eye(3) - np.outer(u, u)
    med = float(np.median(np.concatenate(list(effects.values()))))
    out = {}
    for arm in ("SS-EI", "KIN", "KIN-1T"):
        cov = setup["arms"][arm]["fisher_noise"] @ setup["arms"][arm]["fisher_noise"].mT
        sd_along = np.sqrt(np.clip(np.einsum("i,gij,j->g", u, cov, u), 0, None))
        # Per-direction SD in the plane: half the plane trace is the mean variance there.
        sd_plane = np.sqrt(np.einsum("ij,gji->g", plane, cov) / 2)
        out[arm] = {
            "median_r_kin": float(np.median(sd_along) / med),
            "median_r_ss": float(np.median(sd_plane) / med),
        }
    return out


# --------------------------------------------------------------------- analytic maps


def analytic_maps(setup: dict) -> dict:
    """Prior-implied map of unit single-rate changes, with exact (a - b, a - c) rows for SS-EI
    and a - c for SS-E, as the decision text states them."""
    priors = {
        "SS-EI | independent": (setup["priors"]["independent:SS-EI"], SS_IDENTIFIED["SS-EI"]),
        "SS-EI | KIN prior": (setup["priors"]["KIN"], SS_IDENTIFIED["SS-EI"]),
        "SS-E | independent": (setup["priors"]["independent:SS-E"], SS_IDENTIFIED["SS-E"]),
    }
    saved = torch.load(RUN / "fits_full.pt", weights_only=False)["fits"]
    for d in ("SS-EI", "SS-E"):
        priors[f"SS-EI | {d}-fitted prior"] = (
            saved[d]["prior_cov"].numpy(),
            SS_IDENTIFIED["SS-EI"],
        )
    out = {}
    for name, (sigma, rows) in priors.items():
        m = conditional_map(sigma, rows)
        per = {}
        for j, rate in enumerate("abc"):
            delta = m[:, j]
            mag = np.abs(delta)
            order = np.argsort(mag)[::-1]
            per[f"delta {rate} = 1"] = {
                "predicted": delta.round(4).tolist(),
                "routes_to": "abc"[order[0]],
                "tie": bool(mag[order[0]] - mag[order[1]] <= 1e-9),
                "margin": (
                    float((mag[j] - np.delete(mag, j).max()) / mag.sum()) if mag.sum() else None
                ),
            }
        out[name] = {"map": m.round(4).tolist(), "unit_changes": per}
    return out


def ss_e_exonic_direction(setup: dict) -> dict:
    """Under the Phase 1 exonic likelihood SS-E constrains log(exp(a - b) + exp(a - c)), whose
    gradient (1, -w, -(1 - w)) carries the pre-mRNA share w of exonic signal."""
    rows = np.stack([identified_rows(i)[0] for i in setup["info"]["SS-E"]])
    rows = rows * np.sign(rows[:, :1])
    rows = rows / rows[:, :1]
    return {
        "median_row_normalized_a_1": np.median(rows, 0).round(4).tolist(),
        "pre_mrna_share_quartiles": np.quantile(-rows[:, 1], [0.25, 0.5, 0.75]).round(4).tolist(),
    }


# --------------------------------------------------------------------------- figures

COLORS = {
    "SS-E": "#2a78d6",
    "SS-EI": "#eb6834",
    "SS-EI-EISA": "#8a5cc2",
    "KIN": "#1baf7a",
    "KIN-1T": "#eda100",
    "SS-EI | KIN prior": "#9a9a9a",
}


def plot(h2: dict, aniso: dict, r_op: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (pair, title) in zip(
        axes,
        [
            ("syn + dec", "Pure synthesis + pure decay"),
            ("mix_promoter + mix_utr3", "Mixed sets (70% pure)"),
            ("coupled_promoter:0.2 + mix_utr3", "20% of promoter variants rho in (0.5, 1)"),
        ],
        strict=True,
    ):
        ideal = h2[pair]["ideal"]
        for arm, acc in ideal["pooled_accuracy"].items():
            if arm == "KIN-1T":
                continue
            ax.plot(NOISE_RATIOS, acc, color=COLORS[arm], lw=2, label=arm)
        ax.axvline(r_op, color="#555", lw=1, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("noise SD / |true delta| (per identified direction)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("pooled routing accuracy (ties wrong)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "routing_vs_noise.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ticks = [str(r) for r in ANISO_RATIOS]
    for ax, (pair, title) in zip(
        axes,
        [("syn + dec", "Pure synthesis + pure decay"), ("mix_promoter + mix_utr3", "Mixed sets")],
        strict=True,
    ):
        grid = np.array(aniso[pair]["kin_minus_ssei"])
        im = ax.imshow(grid, origin="lower", cmap="RdBu", vmin=-0.5, vmax=0.5)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                weight = "bold" if grid[i, j] >= H2_SSEI else "normal"
                ax.text(
                    j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, weight=weight
                )
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_yticks(range(len(ticks)), ticks)
        ax.set_xlabel("r_kin: noise SD along (1, 1, 1) / |true delta|")
        ax.set_title(f"KIN - SS-EI, {title} (bold: >= {H2_SSEI})", fontsize=10)
    axes[0].set_ylabel("r_ss: noise SD per steady-state direction / |true delta|")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "kin_minus_ssei_anisotropic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ main


def main() -> None:
    set_all_seeds(SEED)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    effects = mpra_effects_ln()
    setup = load_setup(N_GENES, rng)
    sim = simulate(setup, effects, rng)
    h2 = h2_summary(sim)

    aniso = anisotropic_grid(setup, rng)
    split = kin_noise_split(setup, effects)
    # Isotropic equivalent of KIN's Phase 1 noise: mean variance over the three directions.
    r_op = math.sqrt((split["KIN"]["median_r_kin"] ** 2 + 2 * split["KIN"]["median_r_ss"] ** 2) / 3)

    idx = [int(np.searchsorted(NOISE_RATIOS, r)) for r in TABLE_RATIOS]
    table = {
        sc: {
            arm: {
                "accuracy_at": {str(NOISE_RATIOS[i]): float(v["ideal_accuracy"][i]) for i in idx},
                "tie_at": {str(NOISE_RATIOS[i]): float(v["ideal_tie"][i]) for i in idx},
                "fisher_accuracy": v["fisher_accuracy"],
                "fisher_tie": v["fisher_tie"],
                "fisher_noiseless_accuracy": v["fisher_noiseless_accuracy"],
                "fisher_noiseless_tie": v["fisher_noiseless_tie"],
            }
            for arm, v in res.items()
        }
        for sc, res in sim.items()
    }
    out = {
        "n_genes": N_GENES,
        "seed": SEED,
        "genes": "drawn from the KIN-fitted prior (results/phase1/fit_full.json), median "
        "training-gene covariates",
        "scenarios": {s.name: (s.variant_set, s.description) for s in SCENARIOS},
        "effect_size_ln": {
            k: {"n": len(v), "quartiles": np.quantile(v, [0.25, 0.5, 0.75]).round(4).tolist()}
            for k, v in effects.items()
        },
        "phase1_operating_point": {
            "noise_split": split,
            "kin_isotropic_equivalent_ratio": r_op,
        },
        "analytic_maps": analytic_maps(setup),
        "ss_e_exonic_direction": ss_e_exonic_direction(setup),
        "per_scenario": table,
        "h2": h2,
        "anisotropic": aniso,
    }
    write_json(out, OUT / "design_sim.json")
    plot(h2, aniso, r_op)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
