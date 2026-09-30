import numpy as np
import pandas as pd
import pytest

from heroes_sim.config import MetricsConfig
from heroes_sim.metrics import step_responses

CFG = MetricsConfig(settle_s=1.0, step_jump_rad=0.05, tolerance_rad=0.1)


def synthetic():
    """Step 0 -> 1 at t=1 (first order, tau 0.5 s, 20% overshoot bump at 3.2 s), then 1 -> 0.5 at
    t=4 that is never reached."""
    t = np.arange(0, 6, 0.01)
    tgt = np.where(t < 1, 0.0, np.where(t < 4, 1.0, 0.5))
    q = np.where(t < 1, 0.0, 1 - np.exp(-(t - 1) / 0.5))
    q = q + 0.2 * np.exp(-(((t - 3.2) / 0.1) ** 2))  # overshoot bump after settling
    q = np.where(t < 4, q, 0.9)
    return pd.DataFrame({"t": t, "q_j": q, "target_j": tgt})


def test_step_responses():
    s = step_responses(synthetic(), "j", CFG)
    assert len(s) == 2
    assert s.loc[0, "time_to_target"] == pytest.approx(0.5 * np.log(10), abs=0.02)
    assert s.loc[0, "overshoot"] == pytest.approx(0.2, abs=0.02)
    assert np.isnan(s.loc[1, "time_to_target"])  # never within 0.1 of 0.5
    assert s.loc[1, "overshoot"] == 0.0  # stuck above the lower target: not overshoot


def test_no_steps():
    log = pd.DataFrame({"t": [0.0, 0.01], "q_j": [0.0, 0.0], "target_j": [0.5, 0.5]})
    assert step_responses(log, "j", CFG).empty
