"""Phase 5 figures, from committed results only.

fig1 identifiability, fig2 SBC and fig0 (the design simulation) are copied from
results/phase1 and results/design_sim. This module draws:

  fig3_routing_accuracy.png   accuracy per arm and variant set with 95% intervals
  fig4_routing_margin.png     routing margin distributions per arm and set
  fig5_amortization_gap.png   amortized against NUTS posterior means and the KL gap
  fig6_gene_level.png         gene-level correlations and posterior predictive coverage
"""

from __future__ import annotations

import json
import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sk.paths import RESULTS, RUNS  # noqa: E402

OUT = RESULTS / "figures"
P3 = RESULTS / "phase3"
ARM_ORDER = ("SS-E", "SS-EI", "SS-EI-EISA", "KIN", "KIN-1T", "KIN-SHUF", "SS-EI | KIN prior")
COLORS = {
    "SS-E": "#2a78d6",
    "SS-EI": "#eb6834",
    "KIN": "#1baf7a",
    "KIN-1T": "#eda100",
    "SS-EI-EISA": "#e87ba4",
    "KIN-SHUF": "#008300",
    "SS-EI | KIN prior": "#b0afaa",
}
SETS = {"promoter_enhancer": "Promoter and enhancer (correct: a)", "utr3": "3'UTR (correct: c)"}
INK, MUTED = "#2b2a28", "#6f6e69"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", alpha=0.25, lw=0.5)


def fig3(res: dict) -> None:
    arms = [a for a in ARM_ORDER if a in res["pooled"]]
    panels = [*SETS, "pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, panel in zip(axes, panels, strict=True):
        for i, a in enumerate(arms):
            r = res["pooled"][a] if panel == "pooled" else res["sets"][panel]["arms"][a]
            lo, hi = r["accuracy_ci"]
            hatch = "///" if a == "SS-EI | KIN prior" else None
            ax.bar(i, r["accuracy"], 0.7, color=COLORS[a], hatch=hatch, edgecolor="white", lw=0)
            ax.errorbar(
                i,
                r["accuracy"],
                [[r["accuracy"] - lo], [hi - r["accuracy"]]],
                color=INK,
                lw=1,
                capsize=3,
            )
            ax.text(i, hi + 0.02, f"{r['accuracy']:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(arms)), arms, rotation=35, ha="right", fontsize=8)
        title = "Pooled (sets weighted equally)" if panel == "pooled" else SETS[panel]
        if panel != "pooled":
            title += f"\n{res['sets'][panel]['n_significant']} significant variants"
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.axhline(1 / 3, color=MUTED, lw=0.8, ls=(0, (2, 3)))
        _style(ax)
    axes[0].set_ylabel("routing accuracy (share of seeds x variants)")
    axes[0].set_ylim(0, 1.1)
    fig.suptitle(
        f"Routing accuracy, {res['context']} context (95% two-level bootstrap intervals; "
        "dotted: one in three)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"fig3_routing_accuracy_{res['context']}.png", dpi=150)
    plt.close(fig)


def fig4(margins: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, title) in zip(axes, SETS.items(), strict=True):
        arms = [a for a in ARM_ORDER if a in margins[name]]
        data = [margins[name][a] for a in arms]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, a in zip(parts["bodies"], arms, strict=True):
            body.set_facecolor(COLORS[a])
            body.set_alpha(0.7)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color(INK)
            parts[k].set_linewidth(0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(1, len(arms) + 1), arms, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        _style(ax)
    axes[0].set_ylabel("routing margin\n(|d correct| - max |d other|) / sum |d|")
    fig.suptitle(
        "Routing margin of significant variants, seed-mean reporter deltas (> 0: routed correctly)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_routing_margin.png", dpi=150)
    plt.close(fig)


def fig5(gaps: dict) -> None:
    arms = [a for a in ("KIN", "SS-EI", "KIN-1T") if a in gaps]
    fig, axes = plt.subplots(1, len(arms) + 1, figsize=(4 * (len(arms) + 1), 3.8))
    for ax, a in zip(axes, arms, strict=False):
        g = gaps[a]
        ax.scatter(g["nuts_c"], g["amort_c"], s=10, color=COLORS[a], alpha=0.8, lw=0)
        lim = [
            min(g["nuts_c"].min(), g["amort_c"].min()),
            max(g["nuts_c"].max(), g["amort_c"].max()),
        ]
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.set_xlabel("NUTS posterior mean c (gene's own data)")
        ax.set_ylabel("amortized posterior mean c (sequence)")
        ax.set_title(a, loc="left", fontsize=9, color=INK)
        _style(ax)
    ax = axes[-1]
    for i, a in enumerate(arms):
        g = gaps[a]
        ax.boxplot(
            [np.log10(g["kl_amort"]), np.log10(g["kl_prior"])],
            positions=[3 * i, 3 * i + 1],
            widths=0.7,
            patch_artist=True,
            boxprops={"facecolor": COLORS[a], "alpha": 0.7},
            medianprops={"color": INK},
            flierprops={"markersize": 2},
        )
    ax.set_xticks([3 * i + 0.5 for i in range(len(arms))], arms)
    ax.set_ylabel("log10 KL(NUTS || q), nats")
    ax.set_title("left: q = amortized, right: q = prior", loc="left", fontsize=9, color=INK)
    _style(ax)
    fig.suptitle("Amortization gap on 200 held-out chr7 genes (fold1 models, seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_amortization_gap.png", dpi=150)
    plt.close(fig)


def fig6(gm: dict) -> None:
    metrics = {
        "c_vs_slamseq_decay": "c vs SLAM-seq decay\n(held-out assay)",
        "c_vs_timelapse_decay": "c vs TimeLapse-seq decay",
        "a_vs_ttseq_synthesis_proxy": "a vs TT-seq\nsynthesis proxy",
        "predicted_vs_observed_exonic_level": "predicted vs observed\nexonic level",
    }
    arms = [a for a in ARM_ORDER if a in gm["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    width = 0.8 / len(arms)
    for i, a in enumerate(arms):
        seeds = gm["arms"][a]
        vals = np.array(
            [[s["test_chromosomes"][m]["rho"] for m in metrics] for s in seeds.values()]
        )
        x = np.arange(len(metrics)) + (i - (len(arms) - 1) / 2) * width
        ax.bar(x, vals.mean(0), width * 0.92, color=COLORS[a], label=a)
        if len(vals) > 1:
            ax.errorbar(
                x,
                vals.mean(0),
                [vals.mean(0) - vals.min(0), vals.max(0) - vals.mean(0)],
                fmt="none",
                color=INK,
                lw=0.8,
            )
    ax.set_xticks(range(len(metrics)), list(metrics.values()), fontsize=8)
    ax.set_ylabel("Spearman rho, test chromosomes")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title(
        "Out-of-fold gene-level correlations (bars: seed mean; whiskers: seed range)",
        loc="left",
        fontsize=9,
        color=INK,
    )
    _style(ax)
    ax = axes[1]
    terms = ("wc_exon", "wc_intron", "timelapse")
    for i, a in enumerate(arms):
        cov = gm["arms"][a].get("seed0", {}).get("coverage90")
        if not cov:
            continue
        x = np.arange(len(terms)) + (i - (len(arms) - 1) / 2) * width
        ax.bar(x, [cov[t] for t in terms], width * 0.92, color=COLORS[a])
    ax.axhline(0.9, color=INK, lw=0.8, ls=(0, (2, 3)))
    ax.set_xticks(
        range(len(terms)), ["whole-cell exonic", "whole-cell intronic", "TimeLapse"], fontsize=8
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("90% predictive interval coverage")
    ax.set_title(
        "Held-out observations, seed 0 (dotted: nominal)", loc="left", fontsize=9, color=INK
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_gene_level.png", dpi=150)
    plt.close(fig)


def main() -> None:
    from sk.eval import phase3
    from sk.eval.gene_metrics import amortization_gap

    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (RESULTS / "phase1" / "contraction_vs_half_life.png", "fig1_identifiability.png"),
        (RESULTS / "phase1" / "sbc_ranks_full.png", "fig2_sbc_ranks.png"),
        (RESULTS / "design_sim" / "routing_vs_noise.png", "fig0_design_simulation.png"),
    ):
        shutil.copyfile(src, OUT / dst)
    res = phase3.evaluate("reporter")
    fig3(res)
    margins = {
        n: {
            a: phase3.margin(r["_mean_delta"], phase3.CORRECT[n])
            for a, r in res["sets"][n]["arms"].items()
        }
        for n in phase3.CORRECT
    }
    fig4(margins)
    endo = json.loads((P3 / "routing_endogenous.json").read_text())
    if endo.get("pooled"):
        fig3(phase3.evaluate("endogenous"))
    gaps = {}
    for a in ("KIN", "SS-EI", "KIN-1T"):
        g = amortization_gap(a, 0)
        if g is None:
            continue
        nuts = np.load(P3 / "nuts" / f"nuts_{a}.npz", allow_pickle=True)
        post = pd.read_parquet(RUNS / "phase2" / a / "seed0" / "fold1" / "genes.parquet")
        post = post.set_index("gene_id").loc[[str(x) for x in nuts["gene_ids"]]]
        gaps[a] = {
            "nuts_c": nuts["mean"][:, 2],
            "amort_c": post["mean_c"].to_numpy(),
            "kl_amort": g["_kl_amort"],
            "kl_prior": g["_kl_prior"],
        }
    if gaps:
        fig5(gaps)
    fig6(json.loads((P3 / "gene_metrics.json").read_text()))


if __name__ == "__main__":
    main()
