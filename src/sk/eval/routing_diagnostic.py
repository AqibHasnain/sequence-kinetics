"""Post hoc: does each arm's routing depend on the variant?

    python -m sk.eval.routing_diagnostic

Added after the routing results were seen (DECISIONS, 2026-10-26). For each arm and variant
set, the share of variants routed to each rate is compared across significant variants, null
variants and random single-base edits of the same reporter constructs. An arm that routes all
three groups alike has an output direction set by the construct and the network, and its
routing accuracy measures that direction against the set's known mechanism.

Significant and null shares are the mean over seeds of per-seed shares from the stored reporter
deltas (leakage-excluded units, as in sk.eval.phase3). Random edits are the same for every
arm and seed and are scored by each fold model on that fold's constructs: N_UTR_EDITS 3'UTR
inserts per fold, one random base each, and N_PROMOTER_EDITS random bases per satMutMPRA
element; their shares are also seed means. Writes
results/phase3/routing_diagnostic.json.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from sk.eval import phase3 as p3
from sk.eval import score
from sk.model import inputs as mi
from sk.paths import RESULTS
from sk.train import load_trained
from sk.utils.logging import get_logger

log = get_logger(__name__)

ARMS = ("SS-E", "SS-EI", "SS-EI-EISA", "KIN", "KIN-1T", "KIN-SHUF", "SS-EI | KIN prior")
DERIVED = {"SS-EI-EISA": "SS-EI", "SS-EI | KIN prior": "SS-EI"}
N_UTR_EDITS = 100
N_PROMOTER_EDITS = 30
SEED = 0
# Raw SS-EI deltas of a pure decay change lie along (1, 1, -2) once (1, 1, 1) is removed.
DECAY_DIRECTION = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)


def _share(delta: np.ndarray) -> list[float]:
    return (np.bincount(np.abs(delta).argmax(1), minlength=3) / len(delta)).tolist()


def _derive(arm: str, delta: np.ndarray) -> np.ndarray:
    if arm == "SS-EI-EISA":
        return delta @ p3.EISA.T
    if arm == "SS-EI | KIN prior":
        return delta @ p3.kin_prior_map().T
    return delta


def _random_edits(codes: np.ndarray, first: int, length: int, n: int, rng) -> np.ndarray:
    """n copies of codes, each with one base of [first, first + length) changed."""
    pos = first + rng.choice(length, n, replace=length < n)
    alt = np.repeat(codes[None], n, 0)
    alt[np.arange(n), pos] = (alt[np.arange(n), pos] + rng.integers(1, 4, n)) % 4
    return alt


def _edit_deltas(net, codes, flags, cov, alt, device) -> np.ndarray:
    n = len(alt)
    ref, _ = score.predict(net, codes[None], flags[None], cov[None], device)
    out, _ = score.predict(
        net, alt, np.repeat(flags[None], n, 0), np.repeat(cov[None], n, 0), device
    )
    return out - ref


@lru_cache(maxsize=1)
def random_edits() -> dict[str, list]:
    """Per fold: (set, codes, flags, covariates, edited codes) for each reporter construct."""
    rng = np.random.default_rng(SEED)
    utr = pd.read_parquet(score.VARIANTS / "utr3.parquet")
    pe = pd.read_parquet(score.VARIANTS / "promoter_enhancer.parquet")
    out: dict[str, list] = {}
    for fold in p3.FOLDS:
        jobs = []
        inserts = utr.loc[utr["fold"] == fold, "insert_ref"].drop_duplicates().to_numpy()
        for insert in rng.choice(inserts, min(N_UTR_EDITS, len(inserts)), replace=False):
            codes, flags, cov, first = mi.utr_reporter_input(insert)
            alt = _random_edits(codes, first, len(insert), 1, rng)
            jobs.append(("utr3", codes, flags, cov, alt))
        for element_id in sorted(pe.loc[pe["fold"] == fold, "element_id"].unique()):
            el = score._elements()[element_id]
            seq = mi.oriented(el["seq"], el["strand"])
            codes, flags, cov, first = mi.mpra_element_input(seq)
            alt = _random_edits(codes, first, len(seq), N_PROMOTER_EDITS, rng)
            jobs.append(("promoter_enhancer", codes, flags, cov, alt))
        out[fold] = jobs
    return out


def random_edit_deltas(arm: str, seed: int, device: str) -> dict[str, np.ndarray]:
    """Raw (a, b, c) deltas of the random edits per set, from one seed's fold models."""
    out: dict[str, list] = {name: [] for name in p3.CORRECT}
    for fold, jobs in random_edits().items():
        net = load_trained(arm, seed, fold, device)
        for name, codes, flags, cov, alt in jobs:
            out[name].append(_edit_deltas(net, codes, flags, cov, alt, device))
    return {k: np.concatenate(v) for k, v in out.items()}


def stored_shares(arm: str, name: str) -> dict:
    v = p3.variant_table()[name]
    res = {}
    for group in ("significant", "null"):
        units = v.index[v[group]]
        per_seed = [
            p3.arm_deltas(arm, s)[name].reindex(units).dropna().to_numpy()
            for s in p3.completed_seeds(DERIVED.get(arm, arm))
        ]
        res[group] = {
            "n": len(per_seed[0]),
            "routed_share_abc": np.mean([_share(x) for x in per_seed], 0).tolist(),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    edits = {
        a: [random_edit_deltas(a, s, args.device) for s in p3.completed_seeds(a)]
        for a in p3.TRAINED
    }
    out: dict = {"post_hoc": True, "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for name in p3.CORRECT:
            entry = stored_shares(arm, name)
            xs = [_derive(arm, e[name]) for e in edits[DERIVED.get(arm, arm)]]
            entry["random_edit"] = {
                "n": len(xs[0]),
                "routed_share_abc": np.mean([_share(x) for x in xs], 0).tolist(),
            }
            out["arms"][arm][name] = entry
    out["ss_ei_median_abs_cos_decay_direction"] = {}
    for name in p3.CORRECT:
        sig = p3.variant_table()[name]
        units = sig.index[sig["significant"]]
        x = np.concatenate(
            [
                p3.arm_deltas("SS-EI", s)[name].reindex(units).dropna().to_numpy()
                for s in p3.completed_seeds("SS-EI")
            ]
        )
        x = x[np.linalg.norm(x, axis=1) > 0]
        cos = np.abs(x @ DECAY_DIRECTION) / np.linalg.norm(x, axis=1)
        out["ss_ei_median_abs_cos_decay_direction"][name] = float(np.median(cos))
    path = RESULTS / "phase3" / "routing_diagnostic.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
