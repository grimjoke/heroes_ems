"""Collect a sweep's metrics.json files into one table: sweeps/<name>/summary.csv.

    uv run python scripts/aggregate.py sweeps/<name> [--metrics tracking_rmse_r_elbow_flex ...]

Also prints mean +- std across seeds per grid point for the chosen metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from heroes_sim.sweep import aggregate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--metrics", nargs="*", help="metrics to print (default: a few key ones)")
    args = ap.parse_args()

    table = aggregate(args.sweep_dir)
    out = args.sweep_dir / "summary.csv"
    table.to_csv(out, index=False)
    done = int(table["done"].sum())
    print(f"{done}/{len(table)} runs finished -> {out}")
    if not done:
        return
    grid = list(yaml.safe_load((args.sweep_dir / "sweep.yaml").read_text())["grid"])
    wanted = args.metrics or [
        c
        for c in table.columns
        if c.startswith(("tracking_rmse", "ref_rmse", "ref_max_excursion", "stim_dose"))
    ]
    wanted = [c for c in wanted if c in table.columns]
    stats = table[table["done"]].groupby(grid)[wanted].agg(["mean", "std"])
    with_n = stats.assign(n_seeds=table[table["done"]].groupby(grid).size())
    print(with_n.round(4).to_string())


if __name__ == "__main__":
    main()
