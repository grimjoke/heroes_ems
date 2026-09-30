"""Parameter sweeps: grid spec -> numbered, self-contained run files; results -> one table.

A sweep directory looks like:

    sweeps/<name>/
      sweep.yaml          the spec it was made from
      index.csv           index, seed, one column per grid parameter
      0000.yaml ...       run files: {sweep, index, seed, params, config: <resolved SimConfig>}
      results/0000/ ...   one run's outputs (log.parquet, events.parquet, meta.json, metrics.json)
      slurm/              SLURM stdout/stderr per array task
      summary.csv         aggregate(): index.csv joined with every metrics.json

Run files carry the fully resolved config, so a sweep is reproducible after base.yaml
or the patient/scenario files change. No Hydra; plain YAML and CSV.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from heroes_sim.config import load_run_config


class SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    scenario: Path
    patient: Path | None = None  # default: the scenario's patient_file
    base: Path = Path("configs/base.yaml")
    overrides: dict[str, Any] = Field(default_factory=dict)  # fixed for every point
    grid: dict[str, list[Any]] = Field(min_length=1)  # dotted key -> values; full product
    seeds: list[int] = Field(min_length=1)


def load_spec(path: str | Path) -> SweepSpec:
    with open(path) as f:
        return SweepSpec.model_validate(yaml.safe_load(f))


def points(spec: SweepSpec) -> list[tuple[int, dict[str, Any]]]:
    """(seed, params) for every grid point x seed; seeds vary fastest."""
    keys = list(spec.grid)
    return [
        (seed, dict(zip(keys, combo, strict=True)))
        for combo in itertools.product(*(spec.grid[k] for k in keys))
        for seed in spec.seeds
    ]


def make_sweep(spec: SweepSpec, root: str | Path = "sweeps", force: bool = False) -> Path:
    """Validate every point now (a bad value fails here, not on the cluster) and write it."""
    out = Path(root) / spec.name
    if out.exists() and any(out.glob("*.yaml")) and not force:
        raise FileExistsError(f"{out} already has run files; pass force=True to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "slurm").mkdir(exist_ok=True)
    rows = []
    for i, (seed, params) in enumerate(points(spec)):
        cfg = load_run_config(spec.scenario, spec.base, {**spec.overrides, **params}, spec.patient)
        run = {
            "sweep": spec.name,
            "index": i,
            "seed": seed,
            "params": params,
            "config": cfg.model_dump(mode="json"),
        }
        (out / f"{i:04d}.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
        rows.append({"index": i, "seed": seed, **params})
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    (out / "sweep.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    return out


def run_files(sweep_dir: str | Path) -> list[Path]:
    return sorted(Path(sweep_dir).glob("[0-9][0-9][0-9][0-9].yaml"))


def aggregate(sweep_dir: str | Path) -> pd.DataFrame:
    """index.csv joined with each finished run's metrics.json; `done` marks finished runs."""
    d = Path(sweep_dir)
    index = pd.read_csv(d / "index.csv")
    rows = []
    for i in index["index"]:
        f = d / "results" / f"{i:04d}" / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            m.pop("seed", None)
            rows.append({"index": i, **{k: v for k, v in m.items() if k not in index.columns}})
    finished = pd.DataFrame(rows) if rows else pd.DataFrame({"index": []})
    table = index.merge(finished, on="index", how="left")
    table.insert(1, "done", table["index"].isin(finished["index"]))
    return table
