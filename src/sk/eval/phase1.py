"""Phase 1: fit the per-gene model under each design, quantify identifiability, run SBC.

Stages, each writing under results/phase1/ (and fitted parameters under runs/phase1/):
  fit      per-gene free posteriors on training-chromosome genes under each design
  fisher   Fisher information, posterior contraction against half-life, Jacobian ranks
  sbc      simulation-based calibration from the fitted KIN prior and noise model, H1, Gate G1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sk.data import kinetic_inputs  # noqa: E402
from sk.infer import sbc  # noqa: E402
from sk.infer.vi import (  # noqa: E402
    FitConfig,
    GaussianPrior,
    GenePosteriors,
    elbo_per_gene,
    fit,
    initial_means,
)
from sk.kinetics import fisher  # noqa: E402
from sk.kinetics.observations import DESIGNS, GeneData, NoiseParams, simulate  # noqa: E402
from sk.paths import CONFIGS, RESULTS, RUNS  # noqa: E402
from sk.utils.logging import get_logger  # noqa: E402
from sk.utils.manifest import build_manifest, write_json  # noqa: E402
from sk.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

OUT = RESULTS / "phase1"
RUN = RUNS / "phase1"
DTYPE = torch.float64
QUANTITIES = {
    "a": (1.0, 0.0, 0.0),
    "b": (0.0, 1.0, 0.0),
    "c": (0.0, 0.0, 1.0),
    "a-b": (1.0, -1.0, 0.0),
    "a-c": (1.0, 0.0, -1.0),
    "(a+b+c)/3": (1 / 3, 1 / 3, 1 / 3),
}
SBC_QUANTITIES = ("a", "b", "c", "a-b", "a-c")
COLORS = {"SS-E": "#2a78d6", "SS-EI": "#eb6834", "KIN": "#1baf7a", "KIN-1T": "#eda100"}
STYLES = {"SS-E": ":", "SS-EI": "--", "KIN": "-", "KIN-1T": "-."}
TT_FIXED_IN_KIN1T = ("log_tt_scale", "logit_tt_contam")


def _vec(name: str) -> torch.Tensor:
    return torch.tensor(QUANTITIES[name], dtype=DTYPE)


def load_config() -> dict:
    return yaml.safe_load((CONFIGS / "phase1.yaml").read_text())


# ---------------------------------------------------------------------------- fit


def _fit_design(design: str, data: GeneData, cfg: dict, family: str, kin: dict | None) -> dict:
    noise = NoiseParams(dtype=DTYPE)
    trainable = None
    with torch.no_grad():
        totals = data.wc_exon.sum(0)
        noise.log_size_factor.copy_(torch.log(totals / totals.mean()))
        if design == "KIN-1T":
            noise.load_state_dict(kin["noise"])
            trainable = tuple(
                n for n, _ in noise.named_parameters() if n.split(".")[0] not in TT_FIXED_IN_KIN1T
            )
            trainable = tuple(sorted({n.split(".")[0] for n in trainable}))
    m0 = initial_means(data, noise.log_size_factor, cfg["c_init"])
    prior = GaussianPrior(m0.mean(0))
    q = GenePosteriors(m0, family)
    f = cfg["fit"]
    fc = FitConfig(
        steps=f["steps"],
        lr=f["lr"],
        lr_final=f["lr_final"],
        n_samples=f["n_samples"],
        noise_trainable=trainable,
    )
    t0 = time.time()
    history = fit(q, prior, noise, data, DESIGNS[design], fc)
    # Fix the global alpha scale: median posterior-mean a over training genes is 0. Shifting
    # every a, the prior mean of a and the log size factors together leaves the likelihood
    # unchanged.
    with torch.no_grad():
        shift = q.mean[:, 0].median()
        q.mean[:, 0] -= shift
        prior.mean[0] -= shift
        noise.log_size_factor += shift
        elbo = elbo_per_gene(q, prior, noise, data, DESIGNS[design], 200).mean().item()
    log.info("%s fitted in %.0f s, ELBO per gene %.3f", design, time.time() - t0, elbo)
    return {
        "design": design,
        "family": family,
        "q_mean": q.mean.detach().clone(),
        "q_scale_tril": q.scale_tril().detach().clone(),
        "prior_mean": prior.mean.detach().clone(),
        "prior_cov": prior.cov().detach().clone(),
        "noise": {k: v.detach().clone() for k, v in noise.state_dict().items()},
        "elbo_per_gene": elbo,
        "history": history,
        "seconds": time.time() - t0,
    }


def _noise_summary(state: dict) -> dict:
    exp = {k: torch.exp(v).tolist() for k, v in state.items() if k.startswith("log_")}
    out = {
        "size_factor": exp["log_size_factor"],
        "nb_size": {k.split(".")[1]: v[0] for k, v in exp.items() if k.startswith("log_nb_size")},
        "ttseq_scale": exp["log_tt_scale"],
        "ttseq_contamination": torch.sigmoid(state["logit_tt_contam"]).item(),
        "timelapse_sd_scale_kappa": exp["log_tl_sd_scale"][0],
        "timelapse_extra_sd_tau": exp["log_tl_extra_sd"][0],
    }
    return out


def run_fit(cfg: dict, family: str) -> dict:
    set_all_seeds(cfg["seed"])
    gene_ids = kinetic_inputs.modeled_genes("train")
    data = kinetic_inputs.build(gene_ids, DTYPE)
    fits: dict[str, dict] = {}
    for design in cfg["designs"]:
        fits[design] = _fit_design(design, data, cfg, family, fits.get("KIN"))
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"gene_ids": gene_ids, "fits": fits}, RUN / f"fits_{family}.pt")
    summary = {}
    for design, r in fits.items():
        sd = torch.sqrt(torch.diagonal(r["q_scale_tril"] @ r["q_scale_tril"].mT, dim1=-2, dim2=-1))
        summary[design] = {
            "prior_mean_abc": r["prior_mean"].tolist(),
            "prior_cov_abc": r["prior_cov"].tolist(),
            "prior_sd_abc": torch.sqrt(torch.diagonal(r["prior_cov"])).tolist(),
            "noise": _noise_summary(r["noise"]),
            "elbo_per_gene": r["elbo_per_gene"],
            "posterior_mean_median_abc": r["q_mean"].median(0).values.tolist(),
            "posterior_sd_median_abc": sd.median(0).values.tolist(),
            "seconds": r["seconds"],
        }
    write_json(
        {"n_genes": len(gene_ids), "family": family, "rate_units": "per hour", "designs": summary},
        OUT / f"fit_{family}.json",
    )
    return fits


# ------------------------------------------------------------------- templates


def _ttseq_depth(data: GeneData, theta: torch.Tensor) -> torch.Tensor:
    """Normalized total-RNA reads per unit of steady-state level in TT-seq bins, per region."""
    a, b, c = theta.unbind(-1)
    level = data.tt_len_u * torch.exp(a - b)[:, None] + data.tt_len_m * torch.exp(a - c)[:, None]
    ratio = torch.where(data.tt_mask, data.tt_total / level, torch.nan)
    return torch.nanmedian(ratio, dim=0).values


def _with_theta(data: GeneData, theta: torch.Tensor, depth: torch.Tensor) -> GeneData:
    """Covariates of data with TT-seq total-RNA depth recomputed for the genes' new theta."""
    a, b, c = theta.unbind(-1)
    level = data.tt_len_u * torch.exp(a - b)[:, None] + data.tt_len_m * torch.exp(a - c)[:, None]
    return replace(data, tt_total=depth * level)


def _median_template(data: GeneData, n: int) -> GeneData:
    """n copies of a median gene with every observation present."""

    def med(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = torch.where(mask, x, torch.nan)
        return torch.nanmedian(x, dim=0).values.expand(n, *x.shape[1:]).clone()

    return GeneData(
        wc_exon=med(data.wc_exon),
        wc_intron=med(data.wc_intron),
        len_exon=med(data.len_exon),
        len_intron=med(data.len_intron),
        tl_y=med(data.tl_y, data.tl_mask),
        tl_sd=med(data.tl_sd, data.tl_mask),
        tl_mask=torch.ones(n, *data.tl_mask.shape[1:], dtype=torch.bool),
        tl_bg=med(data.tl_bg),
        tt_y=med(data.tt_y, data.tt_mask[..., None].expand_as(data.tt_y)),
        tt_total=med(data.tt_total, data.tt_mask),
        tt_mask=torch.ones(n, 2, dtype=torch.bool),
        tt_len_u=med(data.tt_len_u, data.tt_mask),
        tt_len_m=med(data.tt_len_m, data.tt_mask),
    )


def _noise_module(state: dict) -> NoiseParams:
    noise = NoiseParams(dtype=DTYPE)
    noise.load_state_dict(state)
    for p in noise.parameters():
        p.requires_grad_(False)
    return noise


# ---------------------------------------------------------------------- fisher


def _contiguous_range(x: np.ndarray, ok: np.ndarray) -> list[float] | None:
    """Longest contiguous run of grid points where ok holds, as [min, max] of x."""
    best, start = None, None
    for i, flag in enumerate([*ok, False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return None if best is None else [float(x[best[0]]), float(x[best[1] - 1])]


def run_fisher(cfg: dict, family: str) -> dict:
    set_all_seeds(cfg["seed"])
    saved = torch.load(RUN / f"fits_{family}.pt", weights_only=False)
    kin = saved["fits"]["KIN"]
    data = kinetic_inputs.build(saved["gene_ids"], DTYPE)
    noise = _noise_module(kin["noise"])
    prior_mean, prior_cov = kin["prior_mean"], kin["prior_cov"]
    depth = _ttseq_depth(data, kin["q_mean"])

    fc = cfg["fisher"]
    hl = np.geomspace(fc["half_life_h_min"], fc["half_life_h_max"], fc["n_grid"])
    c = torch.log(math.log(2) / torch.tensor(hl, dtype=DTYPE))
    # b at its prior conditional mean given a = 0 (the median gene) and c.
    s = prior_cov
    idx_ac = [0, 2]
    gain = s[1, idx_ac] @ torch.linalg.inv(s[idx_ac][:, idx_ac])
    a = torch.zeros_like(c)
    b = prior_mean[1] + (torch.stack([a, c], -1) - prior_mean[idx_ac]) @ gain
    theta = torch.stack([a, b, c], -1)
    template = _with_theta(_median_template(data, len(hl)), theta, depth)

    curves: dict[str, dict] = {}
    ranges: dict[str, list[float] | None] = {}
    for design, terms in DESIGNS.items():
        info = fisher.fisher_information(theta, template, noise, terms)
        post = fisher.laplace_posterior_cov(info, prior_cov)
        curves[design] = {
            q: [fisher.contraction(p, prior_cov, _vec(q)).item() for p in post] for q in QUANTITIES
        }
        curves[design]["c_interval90_width"] = (2 * 1.6448536 * torch.sqrt(post[:, 2, 2])).tolist()
        ok = np.array(curves[design]["c"]) >= fc["contraction_threshold"]
        ranges[design] = _contiguous_range(hl, ok)

    # Jacobian rank of each design's observation map at prior draws, on real genes' covariates.
    jc = cfg["jacobian"]
    draws = torch.distributions.MultivariateNormal(prior_mean, prior_cov).sample((jc["n_draws"],))
    at_draws = _with_theta(_median_template(data, jc["n_draws"]), draws, depth)
    ranks = {}
    for design, terms in DESIGNS.items():
        jac, _ = fisher.jacobian(draws, at_draws, noise, terms)
        r = fisher.jacobian_rank(jac, jc["rtol"])
        values, counts = np.unique(r.numpy(), return_counts=True)
        ranks[design] = {int(k): int(v) for k, v in zip(values, counts, strict=True)}

    _plot_contraction(hl, curves, ranges["KIN"], fc["contraction_threshold"])
    out = {
        "template": "median training gene, a = 0, b at its KIN-prior conditional mean given c",
        "jacobian_template": "median training gene covariates at prior draws",
        "contraction_threshold_c": fc["contraction_threshold"],
        "half_life_h": hl.tolist(),
        "contraction": curves,
        "observable_half_life_h": ranges,
        "jacobian_rank_counts": ranks,
        "jacobian_draws": jc["n_draws"],
        "ttseq_depth_per_level": depth.tolist(),
    }
    write_json(out, OUT / f"identifiability_{family}.json")
    return out


def _plot_contraction(hl: np.ndarray, curves: dict, kin_range, threshold: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, q in zip(axes, ("a", "b", "c"), strict=True):
        for design, cur in curves.items():
            ax.plot(hl, cur[q], STYLES[design], color=COLORS[design], lw=2, label=design)
        ax.axhline(threshold, color="#52514e", lw=0.8, ls=(0, (2, 3)))
        if q == "c" and kin_range is not None:
            ax.axvspan(*kin_range, color="#1baf7a", alpha=0.08, lw=0)
        ax.set_xscale("log")
        ax.set_title(f"{q}", loc="left")
        ax.set_xlabel("mRNA half-life (h)")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("posterior contraction\n1 - var(post) / var(prior)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[2].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Posterior contraction of each log rate at the real pulse times and depths "
        "(median gene; shaded: KIN observable range of c)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "contraction_vs_half_life.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- sbc


def run_sbc(cfg: dict, family: str, fit_family: str) -> dict:
    """SBC of the per-gene posterior family `family`, simulating from the `fit_family` fit."""
    set_all_seeds(cfg["seed"] + 1)
    saved = torch.load(RUN / f"fits_{fit_family}.pt", weights_only=False)
    kin = saved["fits"]["KIN"]
    data = kinetic_inputs.build(saved["gene_ids"], DTYPE)
    noise = _noise_module(kin["noise"])
    prior = GaussianPrior(kin["prior_mean"], torch.linalg.cholesky(kin["prior_cov"]))
    for p in prior.parameters():
        p.requires_grad_(False)
    depth = _ttseq_depth(data, kin["q_mean"])
    ranges = json.loads((OUT / f"identifiability_{fit_family}.json").read_text())[
        "observable_half_life_h"
    ]

    sc = cfg["sbc"]
    n = sc["n_genes"]
    truth = prior.dist().sample((n,))
    covariates = data.subset(torch.randint(len(data), (n,)))
    sim = simulate(truth, _with_theta(covariates, truth, depth), noise)

    results: dict[str, dict] = {}
    posterior: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for design, terms in DESIGNS.items():
        t0 = time.time()
        q = GenePosteriors(prior.mean.detach().expand(n, 3), family)
        fit(
            q,
            prior,
            noise,
            sim,
            terms,
            FitConfig(
                steps=sc["steps"],
                lr=sc["lr"],
                lr_final=sc["lr_final"],
                n_samples=sc["n_samples"],
                fit_prior=False,
                noise_trainable=(),
            ),
        )
        with torch.no_grad():
            mean, tril = q.mean.detach(), q.scale_tril().detach()
            cov = tril @ tril.mT
            draws = q.dist().sample((sc["n_draws"],))  # [L, n, 3]
        posterior[design] = (mean, cov)
        per_q = {}
        for name in SBC_QUANTITIES:
            v = _vec(name)
            t = (truth @ v).numpy()
            r = sbc.ranks(t, (draws @ v).T.numpy())
            u = sbc.uniformity(r, sc["n_draws"], sc["n_bins"])
            cov_q = sbc.coverage(
                t, (mean @ v).numpy(), torch.sqrt(v @ cov @ v).numpy(), sc["coverage_level"]
            )
            lo, hi = sc["coverage_range"]
            per_q[name] = {
                "chi2_p": u["p"],
                "chi2": u["chi2"],
                "coverage90": cov_q,
                "pass": bool(u["p"] > sc["min_p"] and lo <= cov_q <= hi),
                "rank_counts": u["counts"],
            }
        results[design] = per_q
        log.info(
            "SBC %s %s in %.0f s: %s",
            family,
            design,
            time.time() - t0,
            {k: (round(v["chi2_p"], 3), round(v["coverage90"], 3)) for k, v in per_q.items()},
        )

    h1 = _h1(cfg, truth, posterior, prior, ranges["KIN"])
    gate = {
        "KIN": {q: results["KIN"][q]["pass"] for q in ("a", "b", "c")},
        "SS-EI": {q: results["SS-EI"][q]["pass"] for q in ("a-b", "a-c")},
    }
    gate["pass"] = all(all(v.values()) for v in gate.values())
    _plot_ranks(results, family, sc)
    out = {
        "family": family,
        "simulated_from": f"KIN fit with {fit_family} posteriors",
        "n_genes": n,
        "n_draws": sc["n_draws"],
        "n_bins": sc["n_bins"],
        "criteria": {"chi2_p_gt": sc["min_p"], "coverage90_in": sc["coverage_range"]},
        "results": results,
        "h1": h1,
        "gate_g1": gate,
    }
    write_json(out, OUT / f"sbc_{family}.json")
    return out


def _h1(cfg: dict, truth, posterior, prior: GaussianPrior, kin_range) -> dict:
    hc = cfg["h1"]
    z = 1.6448536269514722
    half_life = math.log(2) / torch.exp(truth[:, 2])
    inside = (half_life >= kin_range[0]) & (half_life <= kin_range[1])
    width = {d: 2 * z * torch.sqrt(posterior[d][1][:, 2, 2]) for d in ("KIN", "SS-EI")}
    ratio = (width["SS-EI"] / width["KIN"])[inside]
    v = _vec("(a+b+c)/3")
    prior_sd = torch.sqrt(v @ prior.cov().detach() @ v)
    post_sd = torch.sqrt(torch.einsum("i,gij,j->g", v, posterior["SS-EI"][1], v))
    sd_ratio = post_sd / prior_sd
    part1 = ratio.median().item()
    part2 = sd_ratio.median().item()
    return {
        "kin_observable_half_life_h": kin_range,
        "n_genes_inside_range": int(inside.sum()),
        "c_width90_ratio_ssei_over_kin": {
            "median": part1,
            "quartiles": torch.quantile(ratio, torch.tensor([0.25, 0.75], dtype=DTYPE)).tolist(),
            "share_at_least_threshold": (ratio >= hc["width_ratio_min"]).double().mean().item(),
            "threshold": hc["width_ratio_min"],
            "pass": part1 >= hc["width_ratio_min"],
        },
        "ssei_mean_log_rate_sd_ratio_post_over_prior": {
            "median": part2,
            "quartiles": torch.quantile(sd_ratio, torch.tensor([0.25, 0.75], dtype=DTYPE)).tolist(),
            "prior_sd": prior_sd.item(),
            "threshold": hc["sd_ratio_min"],
            "pass": part2 > hc["sd_ratio_min"],
        },
        "pass": part1 >= hc["width_ratio_min"] and part2 > hc["sd_ratio_min"],
    }


def _plot_ranks(results: dict, family: str, sc: dict) -> None:
    from scipy.stats import binom

    designs = list(results)
    fig, axes = plt.subplots(
        len(designs), len(SBC_QUANTITIES), figsize=(13, 2.2 * len(designs)), sharex=True
    )
    n, bins = sc["n_genes"], sc["n_bins"]
    lo, hi = binom.ppf([0.005, 0.995], n, 1 / bins)
    for i, design in enumerate(designs):
        for j, name in enumerate(SBC_QUANTITIES):
            ax = axes[i, j]
            r = results[design][name]
            ax.axhspan(lo, hi, color="#52514e", alpha=0.12, lw=0)
            ax.bar(range(bins), r["rank_counts"], width=0.85, color=COLORS[design])
            ax.set_ylim(0, max(2.2 * n / bins, max(r["rank_counts"]) * 1.05))
            mark = "pass" if r["pass"] else "FAIL"
            ax.set_title(
                f"{design}  {name}\np={r['chi2_p']:.2g}  cov90={r['coverage90']:.2f}  {mark}",
                fontsize=8,
                loc="left",
            )
            ax.set_yticks([])
            ax.set_xticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
    fig.suptitle(
        f"SBC rank histograms, {family}-covariance posteriors, {n} simulated genes "
        f"({bins} bins; band: 99% binomial range under uniformity)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"sbc_ranks_{family}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fit", "fisher", "sbc", "all"))
    ap.add_argument("--family", default="full", choices=("diag", "full"))
    ap.add_argument(
        "--fit-family", default=None, help="posterior family of the fit SBC simulates from"
    )
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["threads"])
    manifest = build_manifest(cfg, "cpu")
    manifest["stage"], manifest["family"] = args.stage, args.family
    write_json(manifest, RUN / f"manifest_{args.stage}_{args.family}_{manifest['run_id']}.json")
    if args.stage in ("fit", "all"):
        run_fit(cfg, args.family)
    if args.stage in ("fisher", "all"):
        run_fisher(cfg, args.family)
    if args.stage in ("sbc", "all"):
        res = run_sbc(cfg, args.family, args.fit_family or args.family)
        log.info("H1 %s, Gate G1 %s", res["h1"], res["gate_g1"])


if __name__ == "__main__":
    main()
