"""Rescore the 3'UTR reporter variants of every completed Phase 2 run in place.

    python -m sk.eval.rescore

The runs were scored while `score.utr_reporter` gave every insert the flags and covariates of
the last insert in its fold (DECISIONS, 2026-10-26). This replaces the utr3 rows of each run's
variants_reporter.parquet with rows from the fixed scorer and the run's saved model, and writes
the size of the change to results/phase3/utr_reporter_rescore.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from sk.eval import score
from sk.eval.phase3 import FOLDS, TRAINED, completed_seeds
from sk.paths import RESULTS
from sk.train import load_trained, run_dir
from sk.utils.logging import get_logger
from sk.utils.manifest import _git

log = get_logger(__name__)

DELTAS = ["delta_a", "delta_b", "delta_c"]


def rescore_run(arm: str, seed: int, fold: str, device: str) -> np.ndarray:
    """Rescore one run's utr3 rows; returns |new - old| of the deltas [n, 3]."""
    path = run_dir(arm, seed, fold) / "variants_reporter.parquet"
    old = pd.read_parquet(path)
    old_utr = old[old["variant_set"] == "utr3"].reset_index(drop=True)
    new_utr = score.utr_reporter(load_trained(arm, seed, fold, device), fold, device)
    new_utr = new_utr.assign(variant_set="utr3")
    if not (
        new_utr["key"].tolist() == old_utr["key"].tolist()
        and new_utr["insert_ref"].tolist() == old_utr["insert_ref"].tolist()
    ):
        raise RuntimeError(f"{arm} seed{seed} {fold}: rescored rows do not match stored rows")
    out = pd.concat([old[old["variant_set"] != "utr3"], new_utr], ignore_index=True)
    out.to_parquet(path, index=False)
    return np.abs(new_utr[DELTAS].to_numpy() - old_utr[DELTAS].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    changes, per_run = [], {}
    for arm in TRAINED:
        for seed in completed_seeds(arm):
            for fold in FOLDS:
                diff = rescore_run(arm, seed, fold, args.device)
                changes.append(diff)
                per_run[f"{arm}/seed{seed}/{fold}"] = float(diff.max())
                log.info("%s seed%d %s: max |change| %.3g", arm, seed, fold, diff.max())
    diff = np.concatenate(changes)
    summary = {
        "git_sha": _git("rev-parse", "HEAD"),
        "device": args.device,
        "n_runs": len(per_run),
        "n_rows": len(diff),
        "abs_delta_change_max": float(diff.max()),
        "abs_delta_change_median": float(np.median(diff)),
        "abs_delta_change_p99": float(np.quantile(diff, 0.99)),
        "max_per_run": per_run,
        "seconds": time.time() - t0,
    }
    (RESULTS / "phase3" / "utr_reporter_rescore.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    log.info("%s", json.dumps({k: v for k, v in summary.items() if k != "max_per_run"}))


if __name__ == "__main__":
    main()
