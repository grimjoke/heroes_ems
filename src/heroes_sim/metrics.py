"""Post-hoc metrics from a controller-rate log. More arrive with the closed loop (M2+)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heroes_sim.config import MetricsConfig


def tracking_rmse(log: pd.DataFrame, joint: str, cfg: MetricsConfig) -> float:
    """RMSE of q vs target, skipping `settle_s` after each target step (reaction + transit).

    A step is a single-sample target change larger than `step_jump_rad`; ramps and sines
    change smoothly and are scored throughout.
    """
    tgt, q, t = (log[c].to_numpy() for c in (f"target_{joint}", f"q_{joint}", "t"))
    keep = ~np.isnan(tgt)
    for tj in t[1:][np.abs(np.diff(tgt)) > cfg.step_jump_rad]:
        keep &= ~((t >= tj) & (t < tj + cfg.settle_s))
    return float(np.sqrt(np.mean((q[keep] - tgt[keep]) ** 2)))


def max_excursion(log: pd.DataFrame, joint: str) -> float:
    """Largest |q - q(0)|: movement without intent (no_intent safety scenario)."""
    q = log[f"q_{joint}"].to_numpy()
    return float(np.max(np.abs(q - q[0])))
