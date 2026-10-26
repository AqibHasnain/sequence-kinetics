"""Phase 3: routing metrics (section 8.2), bootstrap (8.3) and the amended H2 to H4 verdicts.

Definitions, fixed before any variant was scored:

- Unit: a unique substitution (variant_id; for the 3'UTR set, variant_id and reporter insert). It
  is significant if significant in any measurement (as in Gate G2) and null if null in every
  measurement that includes it (DECISIONS, null variants). Variants flagged for sequence overlap
  with their fold's training genes (sk.data.leakage) are excluded.
- Routing: correct when the largest |delta| among (a, b, c) is on a (promoter set) or c (3'UTR
  set); exact ties count as wrong. An arm's accuracy is the mean over its seeds; a variant's
  per-arm score is the share of seeds that route it correctly.
- Pooled accuracy weights the two sets equally. Bootstrap: 2,000 two-level resamples per set
  (elements then variants for the promoter set, genes then variants for the 3'UTR set), with the
  same resample used for every arm, so arm differences are paired by variant and seed.
- H2 (i): KIN - SS-E pooled >= 0.15 with the 95% interval above 0. H2 (ii): the 95% interval of
  KIN - SS-EI pooled inside [-0.05, 0.05] is equivalence; an interval no wider than the band that
  extends beyond it rejects equivalence, with the sign of the side it extends to; a wider
  interval is inconclusive. H3 (i): per set, the 5th percentile of SS-EI-EISA - KIN >= -0.05.
- H4: KIN seed-mean deltas; among significant variants it routes correctly, Spearman rho between
  the sign-adjusted correct-rate delta (delta a, or -delta c) and the mean log2 effect over the
  measurements in which the variant is significant; two-level bootstrap interval.
- Margin: (|delta correct| - max |delta other|) / sum |delta|, on seed-mean deltas.
- Null AUROC: max |delta| (seed mean) of significant against null variants.
- Error decomposition: true delta is the measured effect in natural log on the known rate,
  (e, 0, 0) for promoter variants and (0, 0, -e) for 3'UTR variants (abundance up is decay down).
  Predicted seed-mean deltas are scaled by the least-squares factor per arm and set, and the
  residual is split into its (1, 1, 1) component and its steady-state-plane component.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from sk.data import leakage
from sk.model.priors import SS_IDENTIFIED, arm_prior
from sk.paths import PROCESSED, RESULTS, RUNS
from sk.utils.logging import get_logger

log = get_logger(__name__)

OUT = RESULTS / "phase3"
FOLDS = tuple(f"fold{i}" for i in range(5))
TRAINED = ("SS-E", "SS-EI", "KIN", "KIN-1T", "KIN-SHUF")
CORRECT = {"promoter_enhancer": 0, "utr3": 2}
N_BOOT = 2000
SEED = 0
# EISA rescoring of SS-EI: delta b = 0, delta a = delta(a - b),
# delta c = delta(a - b) - delta(a - c).
EISA = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, -1.0]]) @ SS_IDENTIFIED["SS-EI"]
ONES = np.ones(3) / math.sqrt(3)


def completed_seeds(arm: str) -> list[int]:
    out = []
    for d in sorted((RUNS / "phase2" / arm).glob("seed*")):
        if all((d / f / "manifest.json").exists() for f in FOLDS):
            out.append(int(d.name.removeprefix("seed")))
    return out


def _load(arm: str, seed: int, context: str) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        df = pd.read_parquet(
            RUNS / "phase2" / arm / f"seed{seed}" / fold / f"variants_{context}.parquet"
        )
        frames.append(df.assign(fold=fold))
    df = pd.concat(frames, ignore_index=True)
    if context == "reporter":
        df["insert_ref"] = df["insert_ref"].fillna("") if "insert_ref" in df else ""
    flagged = leakage.flagged()
    keep = [(f, k) not in flagged for f, k in zip(df["fold"], df["key"], strict=True)]
    return df[keep]


def kin_prior_map() -> np.ndarray:
    """SS-EI identified deltas mapped through the KIN-fitted prior (sensitivity row)."""
    _, sigma = arm_prior("KIN")
    sigma = sigma.numpy()
    p = SS_IDENTIFIED["SS-EI"]
    return sigma @ p.T @ np.linalg.solve(p @ sigma @ p.T, p)


@lru_cache(maxsize=1)
def variant_table() -> dict[str, pd.DataFrame]:
    """Per set: one row per unique variant with labels, cluster and measured effect."""
    out = {}
    p = pd.read_parquet(PROCESSED / "variants" / "promoter_enhancer.parquet")
    p["unit"] = p["variant_id"]
    p["cluster"] = p["element_id"]
    u = pd.read_parquet(PROCESSED / "variants" / "utr3.parquet")
    u["unit"] = u["variant_id"] + "|" + u["insert_ref"]
    u["cluster"] = u["gene_id"].fillna(u["chrom"] + ":" + (u["pos0"] // 100_000).astype(str))
    for name, v in (("promoter_enhancer", p), ("utr3", u)):
        v = v[v["fold"] != "excluded"]
        sig_effect = v[v["significant"]].groupby("unit")["log2_effect"].mean()
        g = v.groupby("unit").agg(
            variant_id=("variant_id", "first"),
            cluster=("cluster", "first"),
            fold=("fold", "first"),
            gene_id=("gene_id", "first"),
            significant=("significant", "max"),
            null=("null_variant", "min"),
        )
        g["effect_log2"] = sig_effect.reindex(g.index)
        g.loc[g["significant"], "null"] = False
        out[name] = g
    return out


def arm_deltas(arm: str, seed: int, context: str = "reporter") -> dict[str, pd.DataFrame]:
    """Per set: deltas [unit -> a, b, c] for a trained or derived arm."""
    base = {"SS-EI-EISA": "SS-EI", "SS-EI | KIN prior": "SS-EI"}.get(arm, arm)
    df = _load(base, seed, context)
    out = {}
    for name in CORRECT:
        d = df[df["variant_set"] == name].copy()
        if context == "reporter" and name == "utr3":
            d["unit"] = d["key"] + "|" + d["insert_ref"]
        elif context == "reporter":
            d["unit"] = d["key"]
        else:
            d["unit"] = d["key"]
        if context == "endogenous" and name == "utr3":
            # Endogenous scores have one row per variant; the 3'UTR units also carry the insert.
            units = variant_table()["utr3"].reset_index()[["unit", "variant_id"]]
            d = d.drop(columns="unit").merge(units, left_on="key", right_on="variant_id")
        x = d[["delta_a", "delta_b", "delta_c"]].to_numpy()
        if arm == "SS-EI-EISA":
            x = x @ EISA.T
        elif arm == "SS-EI | KIN prior":
            x = x @ kin_prior_map().T
        out[name] = pd.DataFrame(x, index=d["unit"].to_numpy(), columns=list("abc"))
        out[name] = out[name][~out[name].index.duplicated()]
    return out


def route_correct(delta: np.ndarray, correct: int) -> np.ndarray:
    mag = np.abs(delta)
    top2 = np.sort(mag, axis=1)[:, -2:]
    tie = top2[:, 1] <= top2[:, 0]
    return (mag.argmax(1) == correct) & ~tie


def margin(delta: np.ndarray, correct: int) -> np.ndarray:
    mag = np.abs(delta)
    other = np.delete(mag, correct, axis=1).max(1)
    total = mag.sum(1)
    return np.where(total > 0, (mag[:, correct] - other) / np.where(total > 0, total, 1), 0.0)


def bootstrap_weights(clusters: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """[n_boot, n] counts of each unit under a two-level (cluster, then unit) resample."""
    codes, uniq = pd.factorize(clusters)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n, k = len(clusters), len(sizes)
    w = np.zeros((n_boot, n))
    for i in range(n_boot):
        picked = rng.integers(0, k, k)
        s = sizes[picked]
        draws = starts[np.repeat(picked, s)] + (rng.random(s.sum()) * np.repeat(s, s)).astype(int)
        w[i] = np.bincount(order[draws], minlength=n)
    return w


def interval(x: np.ndarray, level: float = 0.95) -> list[float]:
    lo = (1 - level) / 2
    return [float(np.quantile(x, lo)), float(np.quantile(x, 1 - lo))]


def spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        return float(spearmanr(x, y).statistic) if len(x) > 2 else float("nan")
    keep = w > 0
    xr, yr = rankdata(np.repeat(x[keep], w[keep].astype(int))), None
    yr = rankdata(np.repeat(y[keep], w[keep].astype(int)))
    return float(np.corrcoef(xr, yr)[0, 1])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def equivalence(ci: list[float], band: float = 0.05) -> str:
    lo, hi = ci
    if -band <= lo and hi <= band:
        return "equivalent"
    if hi - lo > 2 * band:
        return "inconclusive"
    if hi > band and lo < -band:
        return "inconclusive"
    return "not equivalent (KIN higher)" if hi > band else "not equivalent (KIN lower)"


# ------------------------------------------------------------------------ main analysis


def evaluate(context: str = "reporter", arms: tuple[str, ...] | None = None) -> dict:
    variants = variant_table()
    available = {a: completed_seeds(a) for a in TRAINED}
    derived = {"SS-EI-EISA": available["SS-EI"], "SS-EI | KIN prior": available["SS-EI"]}
    seeds = {**available, **derived}
    arms = arms or tuple(a for a in (*TRAINED, *derived) if seeds[a])
    # Seeds shared by the H2 and H3 arms, for paired comparisons.
    shared = sorted(set.intersection(*(set(seeds[a]) for a in ("SS-E", "SS-EI", "KIN"))))
    rng = np.random.default_rng(SEED)
    out: dict = {"context": context, "seeds": {a: seeds[a] for a in arms}, "shared_seeds": shared}
    per_set: dict = {}
    for name, correct in CORRECT.items():
        v = variants[name]
        deltas = {a: {s: arm_deltas(a, s, context)[name] for s in seeds[a]} for a in arms}
        units = v.index
        for a in arms:
            for s in seeds[a]:
                units = units.intersection(deltas[a][s].index)
        v = v.loc[units]
        sig = v[v["significant"]]
        null = v[v["null"]]
        w = bootstrap_weights(sig["cluster"].to_numpy(), N_BOOT, rng)
        res: dict = {
            "n_significant": len(sig),
            "n_clusters": int(sig["cluster"].nunique()),
            "n_null": len(null),
            "arms": {},
        }
        score_paired, score_by_seed = {}, {}
        for a in arms:
            x = {s: deltas[a][s].loc[sig.index].to_numpy() for s in seeds[a]}
            score_by_seed[a] = {s: route_correct(x[s], correct).astype(float) for s in x}
            ok = np.mean(list(score_by_seed[a].values()), 0)
            # Arms trained on fewer seeds (KIN-SHUF, KIN-1T) are paired on their own seeds.
            paired_seeds = shared if shared and set(shared) <= set(x) else list(x)
            score_paired[a] = np.mean([score_by_seed[a][s] for s in paired_seeds], 0)
            mean_delta = np.mean([x[s] for s in seeds[a]], 0)
            boot = w @ ok / w.sum(1)
            routed = np.bincount(np.abs(mean_delta).argmax(1), minlength=3) / len(mean_delta)
            xn = np.mean([deltas[a][s].loc[null.index].to_numpy() for s in seeds[a]], 0)
            per_seed = {int(s): float(route_correct(x[s], correct).mean()) for s in seeds[a]}
            res["arms"][a] = {
                "accuracy": float(ok.mean()),
                "accuracy_ci": interval(boot),
                "accuracy_per_seed": per_seed,
                "routed_share_abc": routed.tolist(),
                "margin_median": float(np.median(margin(mean_delta, correct))),
                "margin_quartiles": np.quantile(margin(mean_delta, correct), [0.25, 0.75]).tolist(),
                "null_auroc_max_abs_delta": auroc(
                    np.abs(mean_delta).max(1), np.abs(xn).max(1) if len(xn) else np.zeros(0)
                ),
                "median_abs_delta_abc": np.median(np.abs(mean_delta), 0).tolist(),
                "error_decomposition": error_decomposition(
                    mean_delta, sig["effect_log2"].to_numpy(), correct
                ),
                "_boot": boot,
                "_score": ok,
                "_mean_delta": mean_delta,
            }
            if "KIN" in a or a in ("SS-EI", "SS-E"):
                res["arms"][a]["per_cluster"] = per_cluster(sig, ok, mean_delta, correct)
        res["_w"] = w
        res["_paired"] = score_paired
        res["_by_seed"] = score_by_seed
        res["_sig"] = sig
        per_set[name] = res
    out["sets"] = per_set
    out["pooled"] = pooled(per_set, arms)
    out["hypotheses"] = hypotheses(per_set, out["pooled"], arms)
    out["subgroups"] = subgroups(per_set, arms)
    return out


@lru_cache(maxsize=1)
def _subgroup_units() -> dict[str, dict[str, set]]:
    """Units per subgroup: significant in a cell line's measurements, or, for 3'UTR variants,
    in a gene whose TimeLapse-seq half-life lies inside the KIN observable range."""
    out: dict[str, dict[str, set]] = {}
    p = pd.read_parquet(PROCESSED / "variants" / "promoter_enhancer.parquet")
    p["unit"] = p["variant_id"]
    u = pd.read_parquet(PROCESSED / "variants" / "utr3.parquet")
    u["unit"] = u["variant_id"] + "|" + u["insert_ref"]
    for name, v in (("promoter_enhancer", p), ("utr3", u)):
        sig = v[v["significant"]]
        out[name] = {f"cell line {c}": set(g["unit"]) for c, g in sig.groupby("cell_line")}
    ident = json.loads((RESULTS / "phase1" / "identifiability_full.json").read_text())
    lo, hi = ident["observable_half_life_h"]["KIN"]
    hl = pd.read_parquet(PROCESSED / "k562" / "half_lives.parquet")["half_life_timelapse_h"]
    inside = set(hl[(hl >= lo) & (hl <= hi)].index)
    out["utr3"]["gene half-life inside KIN observable range"] = set(
        u.loc[u["gene_id"].isin(inside), "unit"]
    )
    out["utr3"]["gene half-life outside range or unmeasured"] = set(
        u.loc[~u["gene_id"].isin(inside), "unit"]
    )
    return out


def subgroups(per_set: dict, arms) -> dict:
    out: dict = {}
    for name, groups in _subgroup_units().items():
        r = per_set[name]
        index = r["_sig"].index
        for label, units in groups.items():
            mask = index.isin(list(units))
            if mask.sum() < 10:
                continue
            w = r["_w"][:, mask]
            entry = {
                "n_significant": int(mask.sum()),
                "n_clusters": int(r["_sig"]["cluster"][mask].nunique()),
            }
            drawn = w.sum(1) > 0
            for a in arms:
                ok = r["arms"][a]["_score"][mask]
                # Resamples that draw none of the subgroup's clusters carry no estimate.
                boot = (w[drawn] @ ok) / w[drawn].sum(1)
                entry[a] = {"accuracy": float(ok.mean()), "accuracy_ci": interval(boot)}
            out[f"{name}: {label}"] = entry
    return out


def per_cluster(sig: pd.DataFrame, ok: np.ndarray, delta: np.ndarray, correct: int) -> dict:
    df = pd.DataFrame({"cluster": sig["cluster"].to_numpy(), "ok": ok})
    if sig["cluster"].nunique() > 50:
        return {"n_clusters": int(sig["cluster"].nunique())}
    return {
        c: {"n": int(len(g)), "accuracy": float(g["ok"].mean())} for c, g in df.groupby("cluster")
    }


def error_decomposition(delta: np.ndarray, effect_log2: np.ndarray, correct: int) -> dict:
    e = effect_log2 * math.log(2)
    true = np.zeros_like(delta)
    true[:, correct] = e if correct == 0 else -e
    denom = (delta * delta).sum()
    scale = float((delta * true).sum() / denom) if denom > 0 else 0.0
    resid = scale * delta - true
    along = resid @ ONES
    plane = resid - np.outer(along, ONES)
    pred_along = delta @ ONES
    return {
        "ls_scale": scale,
        "rms_error_111": float(np.sqrt(np.mean(along**2))),
        "rms_error_steady_state_plane": float(np.sqrt(np.mean((plane**2).sum(1)))),
        "rms_true_111": float(np.sqrt(np.mean((true @ ONES) ** 2))),
        "rms_true_plane": float(
            np.sqrt(np.mean(((true - np.outer(true @ ONES, ONES)) ** 2).sum(1)))
        ),
        "share_of_true_sq_norm_explained": float(
            1 - (resid**2).sum() / max((true**2).sum(), 1e-300)
        ),
        "share_of_predicted_sq_norm_along_111": float((pred_along**2).sum() / max(denom, 1e-300)),
    }


def pooled(per_set: dict, arms) -> dict:
    out = {}
    for a in arms:
        acc = 0.5 * sum(per_set[n]["arms"][a]["accuracy"] for n in CORRECT)
        boot = 0.5 * sum(per_set[n]["arms"][a]["_boot"] for n in CORRECT)
        elem = per_set["promoter_enhancer"]["arms"][a].get("per_cluster", {})
        elem_acc = (
            float(np.mean([c["accuracy"] for c in elem.values()]))
            if elem and "n_clusters" not in elem
            else None
        )
        out[a] = {
            "accuracy": acc,
            "accuracy_ci": interval(boot),
            "promoter_element_weighted_accuracy": elem_acc,
        }
    return out


def _paired(r: dict, a: str, b: str, seeds=None) -> np.ndarray:
    """Per-variant score of a - b, on the shared seeds or on the given seeds of both arms."""
    if seeds is None:
        return r["_paired"][a] - r["_paired"][b]
    by = r["_by_seed"]
    return np.mean([by[a][s] - by[b][s] for s in seeds], 0)


def paired_difference(per_set: dict, a: str, b: str, sets=tuple(CORRECT), seeds=None) -> np.ndarray:
    """Bootstrap distribution of accuracy(a) - accuracy(b), averaged over sets."""
    diff = 0.0
    for n in sets:
        r = per_set[n]
        w = r["_w"]
        diff = diff + (w @ _paired(r, a, b, seeds) / w.sum(1)) / len(sets)
    return diff


def point_difference(per_set: dict, a: str, b: str, sets=tuple(CORRECT), seeds=None) -> float:
    return float(np.mean([_paired(per_set[n], a, b, seeds).mean() for n in sets]))


def hypotheses(per_set: dict, pooled_acc: dict, arms) -> dict:
    out: dict = {}
    if not {"KIN", "SS-E", "SS-EI"} <= set(arms):
        return {"status": "H2 arms incomplete"}
    d_e = paired_difference(per_set, "KIN", "SS-E")
    est_e = point_difference(per_set, "KIN", "SS-E")
    ci_e = interval(d_e)
    out["H2_i"] = {
        "KIN_minus_SSE": est_e,
        "ci95": ci_e,
        "KIN_pooled": pooled_acc["KIN"]["accuracy"],
        "pass": bool(est_e >= 0.15 and ci_e[0] > 0),
    }
    d_ei = paired_difference(per_set, "KIN", "SS-EI")
    ci_ei = interval(d_ei)
    out["H2_ii"] = {
        "KIN_minus_SSEI": point_difference(per_set, "KIN", "SS-EI"),
        "ci95": ci_ei,
        "verdict": equivalence(ci_ei),
        # Added after the seed-0 result was seen; reported beside the verdict, not in it.
        "post_hoc_interval_entirely_outside_band": bool(ci_ei[0] > 0.05 or ci_ei[1] < -0.05),
    }
    if "SS-EI-EISA" in arms:
        h3 = {}
        for n in CORRECT:
            d = paired_difference(per_set, "SS-EI-EISA", "KIN", (n,))
            lower = float(np.quantile(d, 0.05))
            h3[n] = {
                "EISA_minus_KIN": point_difference(per_set, "SS-EI-EISA", "KIN", (n,)),
                "one_sided_5th_percentile": lower,
                "ci95": interval(d),
                "pass": bool(lower >= -0.05),
            }
        h3["pass"] = all(h3[n]["pass"] for n in CORRECT)
        out["H3_i"] = h3
    out["H3_ii"] = "untested: no splice-proximal variant set (Phase 4 not run)"
    if "KIN-SHUF" in arms:
        out["KIN_SHUF_control"] = kin_shuf_control(per_set)
    out["H4"] = h4(per_set)
    return out


def kin_shuf_control(per_set: dict) -> dict:
    """KIN-SHUF against SS-EI and KIN, pooled, paired on the KIN-SHUF seeds (seed 0). A gain
    over SS-EI triggers training KIN-SHUF's other seeds (DECISIONS, cross-validation grid)."""
    seeds = sorted(per_set["utr3"]["_by_seed"]["KIN-SHUF"])
    res: dict = {"seeds": seeds}
    for other in ("SS-EI", "KIN"):
        res[f"KIN_SHUF_minus_{other}"] = {
            "estimate": point_difference(per_set, "KIN-SHUF", other, seeds=seeds),
            "ci95": interval(paired_difference(per_set, "KIN-SHUF", other, seeds=seeds)),
        }
    return res


def h4(per_set: dict) -> dict:
    out = {}
    for n, correct in CORRECT.items():
        r = per_set[n]
        sig = r["_sig"]
        delta = r["arms"]["KIN"]["_mean_delta"]
        ok = route_correct(delta, correct)
        x = delta[:, correct] * (1 if correct == 0 else -1)
        y = sig["effect_log2"].to_numpy()
        keep = ok & np.isfinite(y)
        rho = spearman(x[keep], y[keep])
        boots = [spearman(x[keep], y[keep], wi[keep]) for wi in r["_w"][:N_BOOT]]
        ci = interval(np.array(boots))
        out[n] = {"n": int(keep.sum()), "rho": rho, "ci95": ci, "pass": bool(rho > 0 and ci[0] > 0)}
        # All significant variants, ignoring routing, as context.
        out[n]["rho_all_significant"] = spearman(x[np.isfinite(y)], y[np.isfinite(y)])
    out["pass"] = all(out[n]["pass"] for n in CORRECT)
    return out


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def motif_h4(res: dict) -> dict:
    """Spearman of the motif baseline score with the measured effect, on all significant units
    and on the units KIN routes correctly (the H4 set), with the same bootstrap resamples."""
    m = pd.read_parquet(RUNS / "phase3" / "motif_baseline.parquet")
    out = {}
    for name, correct in CORRECT.items():
        src = pd.read_parquet(PROCESSED / "variants" / f"{name}.parquet")
        unit = (
            src["variant_id"]
            if name == "promoter_enhancer"
            else (src["variant_id"] + "|" + src["insert_ref"])
        )
        mm = m[m["variant_set"] == name]
        score = pd.Series(mm["baseline_score"].to_numpy(), index=unit.to_numpy()[mm["row"]])
        score = score.groupby(level=0).mean()
        r = res["sets"][name]
        sig = r["_sig"]
        x = score.reindex(sig.index).to_numpy()
        y = sig["effect_log2"].to_numpy()
        ok = route_correct(r["arms"]["KIN"]["_mean_delta"], correct)
        entry = {}
        for label, keep in (
            ("all_significant", np.isfinite(x) & np.isfinite(y)),
            ("kin_correctly_routed", np.isfinite(x) & np.isfinite(y) & ok),
        ):
            boots = [spearman(x[keep], y[keep], wi[keep]) for wi in r["_w"]]
            entry[label] = {
                "n": int(keep.sum()),
                "rho": spearman(x[keep], y[keep]),
                "ci95": interval(np.array(boots)),
            }
        out[name] = entry
    return out


def agreement(rep: dict, endo: dict, arm: str = "KIN") -> dict:
    """Reporter against endogenous routing for 3'UTR units scored both ways (seed-mean deltas)."""
    out = {}
    for name in CORRECT:
        a, b = rep["sets"][name], endo["sets"][name]
        da = pd.DataFrame(a["arms"][arm]["_mean_delta"], index=a["_sig"].index)
        db = pd.DataFrame(b["arms"][arm]["_mean_delta"], index=b["_sig"].index)
        both = da.index.intersection(db.index)
        if len(both) < 10:
            continue
        ra = np.abs(da.loc[both].to_numpy()).argmax(1)
        rb = np.abs(db.loc[both].to_numpy()).argmax(1)
        out[name] = {
            "n": len(both),
            "same_routed_rate": float((ra == rb).mean()),
            "spearman_delta_abc": [
                spearman(da.loc[both, i].to_numpy(), db.loc[both, i].to_numpy()) for i in range(3)
            ],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for context in ("reporter", "endogenous"):
        res = evaluate(context)
        results[context] = res
        if context == "reporter":
            res["motif_baseline_h4"] = motif_h4(res)
        (OUT / f"routing_{context}.json").write_text(json.dumps(strip(res), indent=1) + "\n")
        log.info("%s: %s", context, json.dumps(strip(res["hypotheses"]))[:2000])
    agree = {
        a: agreement(results["reporter"], results["endogenous"], a)
        for a in results["reporter"]["sets"]["utr3"]["arms"]
    }
    (OUT / "reporter_vs_endogenous.json").write_text(json.dumps(agree, indent=1) + "\n")


if __name__ == "__main__":
    main()
