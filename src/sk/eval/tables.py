"""Markdown tables of the Phase 3 results, written to results/phase3/tables.md."""

from __future__ import annotations

import json

from sk.paths import RESULTS

P3 = RESULTS / "phase3"
ARMS = ("SS-E", "SS-EI", "SS-EI-EISA", "KIN", "KIN-1T", "KIN-SHUF", "SS-EI | KIN prior")
SETS = {"promoter_enhancer": "promoter and enhancer", "utr3": "3'UTR"}


def _ci(v: dict, key: str = "accuracy") -> str:
    lo, hi = v[f"{key}_ci"]
    return f"{v[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def routing_table(res: dict) -> list[str]:
    arms = [a for a in ARMS if a in res["pooled"]]
    lines = [
        "| Arm | Seeds | Promoter and enhancer | 3'UTR | Pooled |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        s = res["sets"]
        lines.append(
            f"| {a.replace(' | ', ', ')} | {len(res['seeds'][a])} | "
            f"{_ci(s['promoter_enhancer']['arms'][a])} | "
            f"{_ci(s['utr3']['arms'][a])} | {_ci(res['pooled'][a])} |"
        )
    n = res["sets"]
    lines.append("")
    lines.append(
        f"Significant variants: {n['promoter_enhancer']['n_significant']} promoter and enhancer "
        f"({n['promoter_enhancer']['n_clusters']} elements), {n['utr3']['n_significant']} 3'UTR "
        f"({n['utr3']['n_clusters']} gene clusters)."
    )
    return lines


def detail_table(res: dict) -> list[str]:
    lines = [
        "| Arm | Set | Routed to a / b / c | Median margin | Null AUROC | LS scale | "
        "RMS error (1,1,1) / plane | RMS true (1,1,1) / plane | Predicted share on (1,1,1) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in [a for a in ARMS if a in res["pooled"]]:
        for name, label in SETS.items():
            r = res["sets"][name]["arms"][a]
            e = r["error_decomposition"]
            share = " / ".join(f"{x:.2f}" for x in r["routed_share_abc"])
            lines.append(
                f"| {a.replace(' | ', ', ')} | {label} | {share} | {r['margin_median']:.2f} | "
                f"{r['null_auroc_max_abs_delta']:.3f} | {e['ls_scale']:.3g} | "
                f"{e['rms_error_111']:.3f} / {e['rms_error_steady_state_plane']:.3f} | "
                f"{e['rms_true_111']:.3f} / {e['rms_true_plane']:.3f} | "
                f"{e['share_of_predicted_sq_norm_along_111']:.2f} |"
            )
    return lines


def subgroup_table(res: dict) -> list[str]:
    arms = [a for a in ("SS-EI", "SS-EI-EISA", "KIN", "KIN-SHUF") if a in res["pooled"]]
    lines = [
        "| Subgroup | n (clusters) | " + " | ".join(arms) + " |",
        "|---|---|" + "---|" * len(arms),
    ]
    for label, g in res["subgroups"].items():
        cells = [_ci(g[a]) for a in arms]
        lines.append(
            f"| {label} | {g['n_significant']} ({g['n_clusters']}) | " + " | ".join(cells) + " |"
        )
    return lines


def element_table(res: dict) -> list[str]:
    arms = [a for a in ("SS-E", "SS-EI", "KIN") if a in res["pooled"]]
    pe = res["sets"]["promoter_enhancer"]["arms"]
    elements = sorted(pe[arms[0]]["per_cluster"])
    lines = ["| Element | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for e in elements:
        n = pe[arms[0]]["per_cluster"][e]["n"]
        cells = [f"{pe[a]['per_cluster'][e]['accuracy']:.2f}" for a in arms]
        lines.append(f"| {e} | {n} | " + " | ".join(cells) + " |")
    return lines


def diagnostic_table(diag: dict) -> list[str]:
    correct = {"promoter_enhancer": 0, "utr3": 2}
    lines = [
        "Share routed to the set's known rate (a for promoter and enhancer, c for 3'UTR), "
        "mean over seeds; random edits are single-base changes at random positions of the "
        "same reporter constructs (`python -m sk.eval.routing_diagnostic`).",
        "",
        "| Arm | Set | Significant | Null | Random single-base edits |",
        "|---|---|---|---|---|",
    ]
    for a, sets in diag["arms"].items():
        for name, label in SETS.items():
            e, i = sets[name], correct[name]
            cells = [
                f"{e[g]['routed_share_abc'][i]:.2f} (n = {e[g]['n']})"
                for g in ("significant", "null", "random_edit")
            ]
            lines.append(f"| {a.replace(' | ', ', ')} | {label} | " + " | ".join(cells) + " |")
    cos = diag["ss_ei_median_abs_cos_decay_direction"]
    lines += [
        "",
        "Median |cos| between SS-EI's raw deltas of significant variants and (1, 1, -2): "
        f"{cos['promoter_enhancer']:.2f} (promoter and enhancer), {cos['utr3']:.2f} (3'UTR).",
    ]
    return lines


def _f(v: dict) -> str:
    return f"{v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"


def gene_diagnostic_tables(diag: dict) -> list[str]:
    d = diag["decay"]
    partial = d["partial_rho_given_covariate_baseline"]
    lines = [
        f"Seed 0, {d['n_genes']} test-chromosome genes with a SLAM-seq rate. The covariate-only "
        "baseline is a per-fold ridge regression of Phase 1 KIN c on the ten gene covariates, "
        "fitted on the fold's training genes.",
        "",
        "| Predictor | Spearman with SLAM-seq | Partial given covariate baseline |",
        "|---|---|---|",
    ]
    for a, r in d["rho"].items():
        cell = _f(partial[a]) if a in partial else "-"
        lines.append(f"| {a} | {r:.3f} | {cell} |")
    lines += ["", "| Paired difference (gene bootstrap) | Estimate [95% interval] |", "|---|---|"]
    lines += [f"| {k} | {_f(v)} |" for k, v in d["paired_difference"].items()]
    lines += ["", "| 32-mer share with fold training genes | n | KIN | KIN-SHUF | SS-EI |"]
    lines += ["|---|---|---|---|---|"]
    for label, v in d["by_kmer_share"].items():
        lines.append(
            f"| {label} | {v['n']} | {v['KIN']:.3f} | {v['KIN-SHUF']:.3f} | {v['SS-EI']:.3f} |"
        )
    a = diag["amortization"]
    lines += [
        "",
        f"Amortized posteriors, {a['fold']} seed {a['seed']} models; held-out genes are the 200 "
        "chr7 NUTS genes.",
        "",
        "| Arm | Median SD a, b, c: training / early-stopping / held-out / NUTS | "
        "KL(NUTS \\|\\| amortized) | KL(NUTS \\|\\| widened) | Exonic 90% coverage: "
        "amortized / widened / NUTS |",
        "|---|---|---|---|---|",
    ]
    for arm, e in a["arms"].items():
        sd = " / ".join(
            ", ".join(f"{x:.2f}" for x in e["posterior_sd_median_abc"][k])
            for k in ("training_genes", "early_stopping_genes", "held_out_genes", "held_out_nuts")
        )
        h = e["held_out"]
        lines.append(
            f"| {arm} | {sd} | {h['kl_nuts_to_amortized_median']:.1f} | "
            f"{h['kl_nuts_to_widened_median']:.1f} | {h['exon_coverage90_amortized']:.2f} / "
            f"{h['exon_coverage90_widened']:.2f} / {h['exon_coverage90_nuts']:.2f} |"
        )
    return lines


def main() -> None:
    rep = json.loads((P3 / "routing_reporter.json").read_text())
    endo = json.loads((P3 / "routing_endogenous.json").read_text())
    out = [
        "# Phase 3 tables",
        "",
        "Generated by `python -m sk.eval.tables` from results/phase3.",
        "",
    ]
    out += ["## Routing accuracy, reporter context (primary)", "", *routing_table(rep), ""]
    out += ["## Routing accuracy, endogenous context", "", *routing_table(endo), ""]
    out += ["## Routing detail, reporter context", "", *detail_table(rep), ""]
    out += ["## Routing detail, endogenous context", "", *detail_table(endo), ""]
    out += ["## Subgroups, reporter context", "", *subgroup_table(rep), ""]
    out += ["## Per element, reporter context (seed-mean accuracy)", "", *element_table(rep), ""]
    out += [
        "## Hypotheses, reporter context",
        "",
        "```json",
        json.dumps(rep["hypotheses"], indent=1),
        "```",
        "",
    ]
    out += [
        "## Hypotheses, endogenous context",
        "",
        "```json",
        json.dumps(endo["hypotheses"], indent=1),
        "```",
        "",
    ]
    diag = P3 / "routing_diagnostic.json"
    if diag.exists():
        out += [
            "## Routing of significant, null and random edits, reporter context (post hoc)",
            "",
            *diagnostic_table(json.loads(diag.read_text())),
            "",
        ]
    gene_diag = P3 / "gene_diagnostics.json"
    if gene_diag.exists():
        out += [
            "## Gene-level decay and amortized posteriors (post hoc)",
            "",
            *gene_diagnostic_tables(json.loads(gene_diag.read_text())),
            "",
        ]
    if "motif_baseline_h4" in rep:
        out += [
            "## Motif baseline (H4 comparison)",
            "",
            "```json",
            json.dumps(rep["motif_baseline_h4"], indent=1),
            "```",
            "",
        ]
    (P3 / "tables.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
