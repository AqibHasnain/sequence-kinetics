"""Chromosome hold-out, read from configs/splits.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from sk.paths import CONFIGS

SPLIT_NAMES = ("train", "validation", "test")


@lru_cache(maxsize=4)
def load_splits(path: Path = CONFIGS / "splits.yaml") -> dict[str, frozenset[str]]:
    raw = yaml.safe_load(path.read_text())
    splits = {name: frozenset(raw[name]) for name in (*SPLIT_NAMES, "excluded")}
    named = [c for name in SPLIT_NAMES for c in splits[name]]
    if len(named) != len(set(named)):
        raise ValueError("a chromosome appears in more than one split")
    if set(named) & splits["excluded"]:
        raise ValueError("an excluded chromosome is assigned to a split")
    return splits


def normalize_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def split_of(chrom: str) -> str | None:
    """Split name for a chromosome, or None when it is excluded or not an autosome in the config."""
    chrom = normalize_chrom(chrom)
    for name, members in load_splits().items():
        if chrom in members:
            return None if name == "excluded" else name
    return None


def split_array(chroms: np.ndarray) -> np.ndarray:
    return np.array([split_of(c) or "none" for c in chroms], dtype=object)


@lru_cache(maxsize=4)
def load_cv_folds(path: Path = CONFIGS / "splits.yaml") -> dict[str, frozenset[str]]:
    """Held-out chromosomes of each cross-validation fold."""
    raw = yaml.safe_load(path.read_text())["cv_folds"]
    folds = {name: frozenset(chroms) for name, chroms in raw.items()}
    members = [c for chroms in folds.values() for c in chroms]
    if len(members) != len(set(members)):
        raise ValueError("a chromosome appears in more than one fold")
    assigned = {c for name in SPLIT_NAMES for c in load_splits()[name]}
    if set(members) != assigned:
        raise ValueError("the folds do not partition the assigned chromosomes")
    return folds


def fold_of(chrom: str) -> str | None:
    """Fold whose held-out chromosomes contain this chromosome."""
    chrom = normalize_chrom(chrom)
    for name, members in load_cv_folds().items():
        if chrom in members:
            return name
    return None
