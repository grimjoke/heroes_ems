"""One run -> log.parquet, events.parquet (safety), meta.json (resolved config, seed, git
hash, package versions, calibration), metrics.json (flat metrics; what sweeps aggregate)."""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from heroes_sim.config import SimConfig

PACKAGES = ("numpy", "scipy", "mujoco", "myosuite", "pydantic", "pandas", "pyarrow")


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run(
    out_dir: str | Path,
    log: pd.DataFrame,
    cfg: SimConfig,
    seed: int,
    extra: dict[str, Any] | None = None,
    events: pd.DataFrame | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(log, preserve_index=False), out / "log.parquet")
    if events is not None:
        pq.write_table(pa.Table.from_pandas(events, preserve_index=False), out / "events.parquet")
    meta = {
        "seed": seed,
        "git": git_hash(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "config": cfg.model_dump(mode="json"),
        **(extra or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out
