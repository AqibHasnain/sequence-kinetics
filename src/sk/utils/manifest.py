"""Run manifests: git state, config hash and package versions for every run."""

from __future__ import annotations

import hashlib
import json
import platform
import secrets
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sk.utils.logging import json_default

_PACKAGES = ("torch", "numpy", "pandas")


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def build_manifest(config: dict, device: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    versions = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "run_id": f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}",
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": config,
        "config_hash": config_hash(config),
        "package_versions": versions,
        "device": device,
        "started_at": now.isoformat(),
    }


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    return path
