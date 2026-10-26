"""Versioned model artifacts on disk.

    artifacts/models/<version>/model.joblib   fitted model object
    artifacts/models/<version>/metadata.json  features, definitions, settings, training range,
                                              data hash, code commit, reference metrics
    artifacts/models/active.json              {"version": ...} of the model used for predictions

Versions look like "logistic-20261026T153000Z-1a2b3c4d": model name, UTC training
time, and a hash of the training configuration plus data.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd

from . import __version__

DEFAULT_DIR = Path("artifacts/models")


def file_sha256(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def git_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parents[1]).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
                               cwd=Path(__file__).resolve().parents[1]).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def make_version(name: str, trained_at: pd.Timestamp, fingerprint: dict) -> str:
    h = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{name}-{trained_at.strftime('%Y%m%dT%H%M%SZ')}-{h}"


def save_model(model, metadata: dict, root=DEFAULT_DIR, activate: bool = True) -> Path:
    root = Path(root)
    version = metadata["version"]
    d = root / version
    if d.exists():
        raise FileExistsError(f"model version {version} already exists")
    d.mkdir(parents=True)
    joblib.dump(model, d / "model.joblib")
    meta = {**metadata, "package_version": __version__, "code_commit": git_commit()}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if activate:
        set_active(version, root)
    return d


def set_active(version: str, root=DEFAULT_DIR) -> None:
    root = Path(root)
    if not (root / version / "model.joblib").exists():
        raise FileNotFoundError(f"no model artifact for {version}")
    (root / "active.json").write_text(json.dumps({"version": version}, indent=2))


def active_version(root=DEFAULT_DIR) -> str | None:
    f = Path(root) / "active.json"
    return json.loads(f.read_text())["version"] if f.exists() else None


def load_model(version: str | None = None, root=DEFAULT_DIR):
    root = Path(root)
    version = version or active_version(root)
    if version is None:
        raise FileNotFoundError(f"no active model in {root}; run `python -m nbamodel train` first")
    d = root / version
    return joblib.load(d / "model.joblib"), json.loads((d / "metadata.json").read_text())


def list_models(root=DEFAULT_DIR) -> list:
    root = Path(root)
    if not root.exists():
        return []
    metas = [json.loads(p.read_text()) for p in root.glob("*/metadata.json")]
    return sorted(metas, key=lambda m: m.get("trained_at", ""))
