"""M5 damping sweep analysis: controller comparison + no-fault detector statistics.

    uv run python scripts/analyze_m5_damping.py            # after the three m5_damping_* sweeps

Part 1 compares reference damping x gain across seeds (mean +- std, and worst case for the
safety metrics). Part 2 pools every no-fault run and reports the distributions that set
detector thresholds (D10 rail count, D10 dead channel, D11 repeated angle readings).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from heroes_sim.sweep import aggregate

J = "r_elbow_flex"
ROOT = Path("sweeps")
DAMP, GAIN, PAT = (
    "controller.joints.r_elbow_flex.damping",
    "controller.joints.r_elbow_flex.gain",
    "patient",
)


def table(name: str) -> pd.DataFrame:
    t = aggregate(ROOT / name)
    t = t[t["done"]].copy()
    if PAT in t:
        t[PAT] = t[PAT].str.extract(r"patients/(\w+)\.yaml")[0]
    return t


def summarize_grid(t: pd.DataFrame, keys: list[str], cols: dict[str, str]) -> pd.DataFrame:
    g = t.groupby(keys)
    out = pd.DataFrame(index=g.size().index)
    for col, how in cols.items():
        if col not in t:
            continue
        if how == "mean":
            out[col] = (
                g[col].mean().round(3).astype(str) + " ± " + g[col].std().round(3).astype(str)
            )
        else:
            out[col + " (worst)"] = g[col].max().round(3)
    out["n"] = g.size()
    return out


def rolling_min_of_mean(x: np.ndarray, n: int) -> float:
    if len(x) < n:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(x)])
    return float(((c[n:] - c[:-n]) / n).min())


def detector_stats(dirs: list[Path], n_dead: int = 10) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for res in sorted((d / "results").glob("[0-9]*")):
            log_f, meta_f = res / "log.parquet", res / "meta.json"
            if not log_f.exists():
                continue
            log = pd.read_parquet(log_f)
            meta = json.loads(meta_f.read_text())
            rest = meta.get("emg_rest_std", {})
            row = {"run": f"{d.name}/{res.name}", "q_meas_repeat_max": _repeat(log[f"q_meas_{J}"])}
            for ch in ("biceps", "triceps"):
                row[f"rail_max_{ch}"] = int(log[f"emg_rail_{ch}"].max())
                if ch in rest:
                    ratio = log[f"emg_std_{ch}"].to_numpy() / rest[ch]
                    row[f"std_ratio_min{n_dead}_{ch}"] = rolling_min_of_mean(ratio, n_dead)
            rows.append(row)
    return pd.DataFrame(rows)


def _repeat(x: pd.Series) -> int:
    same = np.r_[False, np.diff(x.to_numpy()) == 0]
    best = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return best


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    step = table("m5_damping_step")
    cols = {
        f"tracking_rmse_{J}": "mean",
        f"ref_rmse_{J}": "mean",
        f"overshoot_max_{J}": "mean",
        f"steps_reached_{J}": "mean",
        f"beyond_limit_max_{J}": "worst",
        f"time_beyond_limit_s_{J}": "worst",
        "stim_dose_biceps_stim": "mean",
        "stim_dose_triceps_stim": "mean",
    }
    print(
        f"== step_targets: damping x gain, {step['seed'].nunique()} seeds (mean ± std; worst case)"
    )
    print(summarize_grid(step, [PAT, DAMP, GAIN], cols).to_string())
    beyond = step.assign(beyond=step[f"beyond_limit_max_{J}"] > 0)
    print("\n== runs that passed the joint limit (fraction of seeds)")
    print(beyond.groupby([PAT, DAMP, GAIN])["beyond"].mean().unstack(GAIN).round(2).to_string())

    ni = table("m5_damping_no_intent")
    print(f"\n== no_intent: {ni['seed'].nunique()} seeds")
    print(
        summarize_grid(
            ni,
            [PAT, DAMP],
            {
                f"ref_max_excursion_{J}": "worst",
                f"controller_excursion_{J}": "worst",
                f"stim_off_max_excursion_{J}": "worst",
                "stim_dose_biceps_stim": "mean",
                "stim_dose_triceps_stim": "mean",
            },
        ).to_string()
    )

    fh = table("m5_damping_fatigue")
    print(f"\n== fatigue_hold (sci_c5): {fh['seed'].nunique()} seeds")
    late = []
    for i in fh["index"]:
        log = pd.read_parquet(ROOT / "m5_damping_fatigue" / "results" / f"{i:04d}" / "log.parquet")
        late.append(float(log.loc[log["t"] > 80, f"q_{J}"].std()))
    fh["q_std_after_80s"] = late
    print(
        summarize_grid(
            fh,
            [DAMP],
            {
                f"tracking_rmse_{J}": "mean",
                "q_std_after_80s": "mean",
                f"beyond_limit_max_{J}": "worst",
                "final_capacity_min": "mean",
            },
        ).to_string()
    )

    stats = detector_stats([ROOT / f"m5_damping_{s}" for s in ("step", "no_intent", "fatigue")])
    print(f"\n== no-fault detector statistics over {len(stats)} runs")
    q = stats.describe(percentiles=[0.5, 0.9, 0.99]).T[["min", "50%", "90%", "99%", "max"]]
    print(q.round(3).to_string())
    counts = stats["q_meas_repeat_max"].value_counts().sort_index()
    print("\nlongest identical angle-reading run per run (repeats: runs):", dict(counts))


if __name__ == "__main__":
    main()
