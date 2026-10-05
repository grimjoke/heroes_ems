"""Post-hoc metrics from a controller-rate log (and the safety event table).

`summarize` returns one flat {name: number} dict per run; it is what run.py prints,
what goes into metrics.json, and one row of a sweep table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from heroes_safety import RULES
from heroes_sim.config import MetricsConfig, MovementScenario, SimConfig


def _dt(log: pd.DataFrame) -> float:
    return float(log["t"].iloc[1] - log["t"].iloc[0])


def _step_mask(log: pd.DataFrame, joint: str, cfg: MetricsConfig) -> np.ndarray:
    """Samples with a target, minus `settle_s` after each target step."""
    tgt, t = log[f"target_{joint}"].to_numpy(), log["t"].to_numpy()
    keep = ~np.isnan(tgt)
    for tj in t[1:][np.abs(np.diff(tgt)) > cfg.step_jump_rad]:
        keep &= ~((t >= tj) & (t < tj + cfg.settle_s))
    return keep


def tracking_rmse(log: pd.DataFrame, joint: str, cfg: MetricsConfig, of: str = "q") -> float:
    """RMSE of `of` (q, or ref for the controller's reference) vs target, skipping `settle_s`
    after each target step (reaction + transit).

    A step is a single-sample target change larger than `step_jump_rad`; ramps and sines
    change smoothly and are scored throughout.
    """
    keep = _step_mask(log, joint, cfg)
    err = log[f"{of}_{joint}"].to_numpy()[keep] - log[f"target_{joint}"].to_numpy()[keep]
    return float(np.sqrt(np.mean(err**2)))


def max_excursion(log: pd.DataFrame, joint: str, of: str = "q") -> float:
    """Largest |x - x(0)| of q (or ref): movement without intent (no_intent scenario)."""
    x = log[f"{of}_{joint}"].to_numpy()
    return float(np.max(np.abs(x - x[0])))


def excursion_vs_baseline(log: pd.DataFrame, baseline: pd.DataFrame, joint: str) -> float:
    """Largest |q - q_baseline|: arm motion caused by the controller and stim, with the
    patient's own motion (the stim-off run, same seed) subtracted out."""
    q, qb = log[f"q_{joint}"].to_numpy(), baseline[f"q_{joint}"].to_numpy()
    return float(np.max(np.abs(q - qb)))


def reference_drift_rate(log: pd.DataFrame, joint: str) -> float:
    """Mean |d ref / dt| in rad/s: how fast the reference wanders."""
    return float(np.mean(np.abs(log[f"ref_v_{joint}"].to_numpy())))


def stim_dose(log: pd.DataFrame, channel: str) -> float:
    """Integral of applied intensity over time (intensity x s)."""
    return float(log[f"stim_{channel}"].sum() * _dt(log))


def time_at_cap(log: pd.DataFrame, channel: str, cap: float) -> float:
    """Seconds the applied intensity sat at its cap."""
    return float((log[f"stim_{channel}"] >= cap - 1e-12).sum() * _dt(log))


def step_responses(log: pd.DataFrame, joint: str, cfg: MetricsConfig) -> pd.DataFrame:
    """Per target step: time to first come within `tolerance_rad` (NaN if never, before
    the next step) and overshoot past the target in the step's direction (rad, >= 0)."""
    t, q, tgt = (log[c].to_numpy() for c in ("t", f"q_{joint}", f"target_{joint}"))
    jumps = np.nonzero(np.abs(np.diff(tgt)) > cfg.step_jump_rad)[0] + 1
    ends = [*jumps[1:], len(t)] if len(jumps) else []
    rows = []
    for a, b in zip(jumps, ends, strict=True):
        direction = np.sign(tgt[a] - tgt[a - 1])
        err = q[a:b] - tgt[a:b]
        inside = np.nonzero(np.abs(err) < cfg.tolerance_rad)[0]
        rows.append(
            {
                "t_step": t[a],
                "time_to_target": t[a + inside[0]] - t[a] if len(inside) else np.nan,
                "overshoot": float(max(0.0, np.max(direction * err))),
            }
        )
    return pd.DataFrame(rows, columns=["t_step", "time_to_target", "overshoot"])


def limit_excursion(log: pd.DataFrame, joint: str, limits: tuple[float, float]) -> dict:
    """How far and how long q went beyond the joint limits (safety.joint_limits)."""
    q, lo, hi = log[f"q_{joint}"].to_numpy(), limits[0], limits[1]
    beyond = np.maximum(np.maximum(q - hi, lo - q), 0.0)
    return {
        "q_max": float(q.max()),
        "q_min": float(q.min()),
        "beyond_limit_max": float(beyond.max()),
        "time_beyond_limit_s": float((beyond > 0).sum() * _dt(log)),
    }


def longest_repeat(x: np.ndarray) -> int:
    """Longest run of consecutive identical values, counted in repeats (0 = none)."""
    same = np.r_[False, np.diff(x) == 0]
    best = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return best


def safety_event_counts(events: pd.DataFrame) -> dict[str, int]:
    if events.empty:
        return {}
    return {str(k): int(v) for k, v in events["rule"].value_counts().sort_index().items()}


def summarize(cfg: SimConfig, result: Any, events: pd.DataFrame) -> dict[str, float]:
    """Flat metrics for one run (`result` is a runner.RunResult)."""
    sc, log = cfg.scenario, result.log
    m: dict[str, float] = {}
    if not isinstance(sc, MovementScenario):
        return m
    for j in cfg.plant.joints:
        if sc.target is None:
            m[f"max_excursion_{j}"] = max_excursion(log, j)
            if result.baseline_log is not None:
                m[f"stim_off_max_excursion_{j}"] = max_excursion(result.baseline_log, j)
                m[f"controller_excursion_{j}"] = excursion_vs_baseline(log, result.baseline_log, j)
        else:
            m[f"tracking_rmse_{j}"] = tracking_rmse(log, j, cfg.metrics)
            steps = step_responses(log, j, cfg.metrics)
            if len(steps):
                reached = steps["time_to_target"].notna()
                m[f"steps_reached_{j}"] = float(reached.mean())
                m[f"time_to_target_mean_{j}"] = (
                    float(steps.loc[reached, "time_to_target"].mean()) if reached.any() else np.nan
                )
                m[f"overshoot_mean_{j}"] = float(steps["overshoot"].mean())
                m[f"overshoot_max_{j}"] = float(steps["overshoot"].max())
        for key, v in limit_excursion(log, j, cfg.safety.joint_limits[j]).items():
            m[f"{key}_{j}"] = v
        m[f"q_meas_repeat_max_{j}"] = float(longest_repeat(log[f"q_meas_{j}"].to_numpy()))
        if sc.closed_loop:
            if sc.target is None:
                m[f"ref_max_excursion_{j}"] = max_excursion(log, j, of="ref")
                m[f"ref_drift_rate_{j}"] = reference_drift_rate(log, j)
            else:
                m[f"ref_rmse_{j}"] = tracking_rmse(log, j, cfg.metrics, of="ref")
    if sc.closed_loop:
        mvc = getattr(result, "mvc", None)
        if mvc is not None and mvc.deadband is not None:
            m.update({f"deadband_{j}": float(d) for j, d in zip(cfg.plant.joints, mvc.deadband)})
        for ch in cfg.stim.channels:
            m[f"stim_dose_{ch.name}"] = stim_dose(log, ch.name)
            m[f"time_at_cap_{ch.name}"] = time_at_cap(log, ch.name, cfg.safety.cap[ch.name])
        caps = {mu: float(log[f"cap_{mu}"].iloc[-1]) for mu in cfg.plant.muscles}
        m["final_capacity_min"] = min(caps.values())
        m.update({f"final_capacity_{mu}": v for mu, v in caps.items()})
        counts = safety_event_counts(events)
        m.update({f"safety_{r}": float(counts.get(r, 0)) for r in RULES})
    return m
