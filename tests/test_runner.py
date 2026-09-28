"""Episode-level sanity checks: open loop (M1) and closed loop (M2)."""

import numpy as np
import pandas as pd
import pytest

from heroes_sim import metrics
from heroes_sim.config import load_run_config
from heroes_sim.recorder import write_run
from heroes_sim.runner import run

J = "r_elbow_flex"
SCI = "configs/patients/sci_c5.yaml"
# Calibrated values (seed 0, full noise model) so closed-loop tests skip calibration.
HEALTHY_MVC = {
    "scenario.mvc": {"biceps": 0.5833, "triceps": 0.4640},
    "scenario.mvc_rest": {"biceps": 0.0180, "triceps": 0.0171},
}
SCI_MVC = {
    "scenario.mvc": {"biceps": 0.2052, "triceps": 0.0260},
    "scenario.mvc_rest": {"biceps": 0.0164, "triceps": 0.0136},
}


def cfg_for(name, patient=None, **overrides):
    return load_run_config(
        f"configs/scenarios/{name}.yaml", overrides=overrides, patient_path=patient
    )


def q_at(log, t):
    return log.loc[(log.t - t).abs().idxmin(), f"q_{J}"]


def test_healthy_reaches_step_targets_open_loop():
    cfg = cfg_for("step_targets", **{"scenario.duration_s": 7.0, "scenario.closed_loop": False})
    log = run(cfg, seed=0).log
    for t_end, tgt in ((3.95, 1.2), (6.95, 1.8)):  # end of each 3 s hold
        assert q_at(log, t_end) == pytest.approx(tgt, abs=0.05)
    assert "stim_biceps_stim" not in log  # no controller, no stim


def test_mvc_calibration_values_and_impairment():
    out = {}
    for patient in (None, SCI):
        res = run(cfg_for("mvc_calibration", patient), seed=0)
        cal = res.calibration
        out[patient] = ({r.group: r for r in cal.trials}, cal.mvc)
        assert res.log[f"q_{J}"].eq(res.log[f"q_{J}"].iloc[0]).all()  # isometric
        # Rest floor (sensor noise, mains, crosstalk, tone) is below MVC on every channel;
        # for the near-paralysed sci_c5 triceps it is about half of it, which is why
        # normalization subtracts it.
        assert all(0 < r < m for r, m in zip(cal.mvc.rest, cal.mvc.envelope))
    (h_trials, h_mvc), (s_trials, s_mvc) = out[None], out[SCI]
    assert all(r < 0.1 * m for r, m in zip(h_mvc.rest, h_mvc.envelope))
    assert h_trials["elbow_flexors"].torque[0] > 30 and h_trials["elbow_extensors"].torque[0] < -20
    assert 0 < s_trials["elbow_flexors"].torque[0] < h_trials["elbow_flexors"].torque[0]
    # Near-paralysed triceps: much smaller MVC envelope than healthy.
    assert s_mvc.envelope[1] < 0.2 * h_mvc.envelope[1]


def test_calibration_runs_before_closed_loop_episode():
    short = {"calibration.rest_s": 0.5, "calibration.window_s": 0.25}
    cfg = cfg_for("no_intent", **{"scenario.duration_s": 1.0, **short})
    res = run(cfg, seed=0)
    assert res.calibration is not None and res.mvc == res.calibration.mvc


@pytest.mark.parametrize("patient,mvc", [(None, HEALTHY_MVC), (SCI, SCI_MVC)])
def test_no_intent_closed_loop_is_quiet(patient, mvc):
    """Key safety scenario: no intent -> reference holds, little stim, no oscillation."""
    log = run(cfg_for("no_intent", patient, **mvc), seed=0).log
    assert metrics.max_excursion(log, J, of="ref") < 0.05
    assert metrics.max_excursion(log, J) < 0.25
    for ch in ("biceps_stim", "triceps_stim"):
        assert metrics.stim_dose(log, ch) < 1.0  # intensity-seconds over 10 s
        assert metrics.time_at_cap(log, ch, 0.8) == 0.0


@pytest.mark.parametrize("patient,mvc", [(None, HEALTHY_MVC), (SCI, SCI_MVC)])
def test_step_targets_closed_loop_tracks(patient, mvc):
    cfg = cfg_for("step_targets", patient, **mvc)
    res = run(cfg, seed=0)
    log = res.log
    assert metrics.tracking_rmse(log, J, cfg.metrics) < 0.25
    assert log["stim_biceps_stim"].max() > 0.05  # stim actually participates
    assert not any(e.rule in ("watchdog", "sensor_sanity") for e in res.events)


def test_closed_loop_helps_impaired_hold():
    """sci_c5 alone sags below the 1.8 rad target; stim assistance holds it higher."""
    base = {"scenario.duration_s": 7.0, **SCI_MVC}
    open_log = run(cfg_for("step_targets", SCI, **base, **{"scenario.closed_loop": False}), 0).log
    closed_log = run(cfg_for("step_targets", SCI, **base), 0).log
    hold = slice(6.0, 7.0)
    q_open = open_log.set_index("t").loc[hold, f"q_{J}"].mean()
    q_closed = closed_log.set_index("t").loc[hold, f"q_{J}"].mean()
    assert abs(1.8 - q_closed) < abs(1.8 - q_open)


def test_same_seed_identical_parquet(tmp_path):
    cfg = cfg_for("step_targets", **{"scenario.duration_s": 2.0, **HEALTHY_MVC})
    runs = [(s, run(cfg, seed=s)) for s in (3, 3, 4)]
    paths = [write_run(tmp_path / str(i), r.log, cfg, s) for i, (s, r) in enumerate(runs)]
    blobs = [(p / "log.parquet").read_bytes() for p in paths]
    assert blobs[0] == blobs[1]
    a, c = (pd.read_parquet(p / "log.parquet") for p in (paths[0], paths[2]))
    assert not np.array_equal(a["u_vol_BIClong"], c["u_vol_BIClong"])


@pytest.mark.parametrize(
    "blanking,runaway",
    [
        ({"controller.blanking_ms": 0}, True),
        ({"controller.blanking_fill": "zero"}, True),
        ({}, False),
    ],
    ids=["no_blanking", "zero_fill", "hold_fill"],
)
def test_stim_artifact_feedback_and_blanking(blanking, runaway):
    """M3 failure mode: stim -> artifact/M-wave -> envelope -> intent -> more stim.

    sci_c5 with no intent: without blanking (or with zero-fill blanking, whose notches in
    the baseline read as EMG) the reference runs away into the ROM clamp. Sample-and-hold
    blanking keeps it still.
    """
    log = run(cfg_for("no_intent", SCI, **SCI_MVC, **blanking), seed=0).log
    drift = metrics.max_excursion(log, J, of="ref")
    if runaway:
        assert drift > 0.3
    else:
        assert drift < 0.05
