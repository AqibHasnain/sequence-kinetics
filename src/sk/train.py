"""Train one arm, seed and cross-validation fold, then score its held-out chromosomes.

    python -m sk.train --arm KIN --seed 0 --fold fold0

Writes runs/phase2/<arm>/seed<seed>/<fold>/: model.pt, history.json, genes.parquet (posteriors
of every protein-coding gene on the held-out chromosomes), variants_reporter.parquet,
variants_endogenous.parquet and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.distributions import MultivariateNormal

from sk.data import kinetic_inputs
from sk.data.splits import load_cv_folds
from sk.eval import score
from sk.infer.amortized import data_subset, gaussian_elbo, shuffle_labeling
from sk.model import inputs as mi
from sk.model.network import NetConfig, SequenceNet
from sk.model.priors import ARMS, arm_noise, arm_prior, terms
from sk.paths import CONFIGS, PROCESSED, RUNS
from sk.utils.logging import get_logger
from sk.utils.manifest import build_manifest, write_json
from sk.utils.seeding import set_all_seeds

log = get_logger(__name__)


def load_config() -> dict:
    return yaml.safe_load((CONFIGS / "base.yaml").read_text())


def run_dir(arm: str, seed: int, fold: str):
    return RUNS / "phase2" / arm / f"seed{seed}" / fold


def fold_genes(fold: str, cfg: dict) -> tuple[list[str], list[str]]:
    """Modeled genes for gradient steps and for early stopping, both off the held-out fold."""
    genes = pd.read_parquet(PROCESSED / "genes" / "genes.parquet").set_index("gene_id")
    ids = [g for s in ("train", "validation", "test") for g in kinetic_inputs.modeled_genes(s)]
    chrom = genes.loc[ids, "chrom"]
    held, val = load_cv_folds()[fold], cfg["validation_chrom"][fold]
    if val in held:
        raise ValueError("validation chromosome is held out by its own fold")
    train = sorted(g for g in ids if chrom[g] not in held and chrom[g] != val)
    valid = sorted(g for g in ids if chrom[g] == val)
    return train, valid


def build_net(cfg: dict, prior_mean: torch.Tensor) -> SequenceNet:
    n = cfg["network"]
    net_cfg = NetConfig(
        n_covariates=len(mi.COVARIATES),
        stem_channels=n["stem_channels"],
        channels=tuple(n["channels"]),
        kernel=n["kernel"],
        hidden=n["hidden"],
        dropout=n["dropout"],
    )
    return SequenceNet(net_cfg, prior_mean)


def load_trained(arm: str, seed: int, fold: str, device: str) -> SequenceNet:
    net = build_net(load_config(), arm_prior(arm)[0].float())
    net.load_state_dict(torch.load(run_dir(arm, seed, fold) / "model.pt", map_location="cpu"))
    return net.to(device).eval()


class GeneSet:
    """Inputs and observations of a gene list, held in memory."""

    def __init__(self, gene_ids: list[str]):
        ids, enc = mi.gene_inputs()
        rows = ids.get_indexer(gene_ids)
        if (rows < 0).any():
            raise ValueError("gene without an encoded input")
        self.gene_ids = gene_ids
        self.codes = torch.as_tensor(np.asarray(enc.codes[rows]))
        self.flags = torch.as_tensor(np.asarray(enc.flags[rows]))
        self.cov = torch.as_tensor(mi.transform_covariates(enc.covariates[rows]))
        self.data = kinetic_inputs.build(gene_ids)

    def __len__(self) -> int:
        return len(self.gene_ids)


def _forward(net, gs: GeneSet, idx: np.ndarray, device: str):
    t = torch.as_tensor(idx)
    return net(gs.codes[t].to(device), gs.flags[t].to(device), gs.cov[t].to(device))


@torch.no_grad()
def evaluate(net, gs: GeneSet, prior, noise, arm_terms, device, n_samples, batch) -> float:
    net.eval()
    total = 0.0
    for i in range(0, len(gs), batch):
        idx = np.arange(i, min(i + batch, len(gs)))
        mean, tril = _forward(net, gs, idx, device)
        value = float("-inf")
        for _ in range(3):
            try:
                elbo = gaussian_elbo(
                    mean, tril, prior, noise, data_subset(gs.data, idx), arm_terms, n_samples
                )
            except ValueError:
                continue
            if torch.isfinite(elbo).all():
                value = elbo.sum().item()
                break
        total += value
    return total / len(gs)


def train(arm: str, seed: int, fold: str, device: str, max_epochs: int | None = None) -> dict:
    cfg = load_config()
    tc = cfg["train"]
    if max_epochs is not None:
        tc["max_epochs"] = max_epochs
    out = run_dir(arm, seed, fold)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest({"arm": arm, "seed": seed, "fold": fold, **cfg}, device)
    set_all_seeds(seed)
    torch.set_num_threads(4)

    train_ids, val_ids = fold_genes(fold, cfg)
    tr, va = GeneSet(train_ids), GeneSet(val_ids)
    if arm == "KIN-SHUF":
        tr.data, _ = shuffle_labeling(tr.data, seed)
        va.data, _ = shuffle_labeling(va.data, seed + 1)
    mean0, cov0 = arm_prior(arm)
    prior = MultivariateNormal(mean0, cov0)
    noise = arm_noise(arm)
    arm_terms = terms(arm)

    net = build_net(cfg, mean0).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_state, bad, history, skipped = -np.inf, None, 0, [], 0
    t_start = time.time()
    for epoch in range(tc["max_epochs"]):
        t0 = time.time()
        net.train()
        order = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(order), tc["batch_size"]):
            idx = order[i : i + tc["batch_size"]]
            if len(idx) < 2:
                continue
            mean, tril = _forward(net, tr, idx, device)
            # A Monte Carlo draw far in the tail can overflow the labeled-fraction terms. Such a
            # step is skipped and counted; runs that never hit one are unaffected.
            try:
                elbo = gaussian_elbo(
                    mean, tril, prior, noise, data_subset(tr.data, idx), arm_terms, tc["n_samples"]
                )
                loss = -elbo.mean()
            except ValueError:
                loss = torch.tensor(float("nan"))
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 50:
                    raise RuntimeError("more than 50 non-finite training steps")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc["grad_clip"])
            opt.step()
            losses.append(loss.item())
        val = evaluate(net, va, prior, noise, arm_terms, device, tc["val_samples"], 64)
        history.append(
            {
                "epoch": epoch,
                "train_elbo": -float(np.mean(losses)),
                "val_elbo": val,
                "seconds": time.time() - t0,
                "skipped_steps_total": skipped,
            }
        )
        log.info(
            "%s seed%d %s epoch %d train %.3f val %.3f (%.0f s)",
            arm,
            seed,
            fold,
            epoch,
            -np.mean(losses),
            val,
            time.time() - t0,
        )
        if val > best:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    net.load_state_dict(best_state)
    torch.save(best_state, out / "model.pt")
    train_seconds = time.time() - t_start

    t0 = time.time()
    genes = score.gene_predictions(net, fold, device)
    genes.to_parquet(out / "genes.parquet", index=False)
    reporter = pd.concat(
        [
            score.promoter_reporter(net, fold, device).assign(variant_set="promoter_enhancer"),
            score.utr_reporter(net, fold, device).assign(variant_set="utr3"),
        ],
        ignore_index=True,
    )
    reporter.to_parquet(out / "variants_reporter.parquet", index=False)
    score.endogenous(net, fold, device).to_parquet(out / "variants_endogenous.parquet", index=False)
    summary = {
        **manifest,
        "n_train_genes": len(tr),
        "n_val_genes": len(va),
        "best_val_elbo": best,
        "epochs_run": len(history),
        "best_epoch": int(np.argmax([h["val_elbo"] for h in history])),
        "train_seconds": train_seconds,
        "score_seconds": time.time() - t0,
        "n_reporter_variants": len(reporter),
    }
    write_json(history, out / "history.json")
    write_json(summary, out / "manifest.json")
    log.info("%s", json.dumps({k: summary[k] for k in ("best_val_elbo", "epochs_run")}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    train(args.arm, args.seed, args.fold, args.device, args.max_epochs)


if __name__ == "__main__":
    main()
