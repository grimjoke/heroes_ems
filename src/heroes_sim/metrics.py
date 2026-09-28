"""Post-hoc metrics from a controller-rate log (and the safety event table)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heroes_sim.config import MetricsConfig


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


def reference_drift_rate(log: pd.DataFrame, joint: str) -> float:
    """Mean |d ref / dt| in rad/s: how fast the reference wanders."""
    return float(np.mean(np.abs(log[f"ref_v_{joint}"].to_numpy())))


def stim_dose(log: pd.DataFrame, channel: str) -> float:
    """Integral of applied intensity over time (intensity x s)."""
    return float(log[f"stim_{channel}"].sum() * _dt(log))


def time_at_cap(log: pd.DataFrame, channel: str, cap: float) -> float:
    """Seconds the applied intensity sat at its cap."""
    return float((log[f"stim_{channel}"] >= cap - 1e-12).sum() * _dt(log))


def safety_event_counts(events: pd.DataFrame) -> dict[str, int]:
    if events.empty:
        return {}
    return {str(k): int(v) for k, v in events["rule"].value_counts().sort_index().items()}
