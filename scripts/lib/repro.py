"""Reproducibility helpers: configuration, seeding, table I/O and per-stage provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config" / "run.yaml") -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_sha"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return cfg


def p(rel: str | Path) -> Path:
    q = Path(rel)
    return q if q.is_absolute() else ROOT / q


def stage_dir(stage: str) -> Path:
    d = ROOT / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def md5(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def write_provenance(stage: str, cfg: dict, inputs: list, outputs: list,
                     extra: dict | None = None, tag: str = "") -> Path:
    def h(x):
        x = Path(x)
        if x.is_file() and x.stat().st_size < 2e9:
            return md5(x)
        return "dir/large"
    rec = {
        "stage": stage, "time_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.get("seed"), "config_sha": cfg.get("_sha"), "git_commit": _git_commit(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "inputs": {str(i): h(i) for i in inputs},
        "outputs": {str(o): h(o) for o in outputs},
    }
    for mod in ("torch", "numpy", "pandas", "sklearn", "scanpy", "lightgbm"):
        try:
            rec[mod] = __import__(mod).__version__
        except Exception:
            pass
    if extra:
        rec.update(extra)
    out = stage_dir(stage) / (f"PROVENANCE_{tag}.json" if tag else "PROVENANCE.json")
    out.write_text(json.dumps(rec, indent=2, default=str))
    return out
