"""Episode-level sanity checks: open loop (M1) and closed loop (M2)."""

import json

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
# Refresh from `scripts/run.py configs/scenarios/mvc_calibration.yaml` -> meta.json "calibrated".
HEALTHY_MVC = {
    "scenario.calibrated": {
        "envelope": {"biceps": 0.5807, "triceps": 0.4634},
        "rest": {"biceps": 0.0141, "triceps": 0.0120},
        "deadband": {"r_elbow_flex": 0.05},
        "emg_rest_std": {"biceps": 0.0208, "triceps": 0.0201},
    }
}
SCI_MVC = {
    "scenario.calibrated": {
        "envelope": {"biceps": 0.2032, "triceps": 0.0234},
        "rest": {"biceps": 0.0119, "triceps": 0.0050},
        "deadband": {"r_elbow_flex": 0.1631},
        "emg_rest_std": {"biceps": 0.0176, "triceps": 0.0128},
    }
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
    assert res.mvc.deadband is not None and res.mvc.deadband[0] >= 0.05  # floor


@pytest.mark.parametrize("patient,mvc", [(None, HEALTHY_MVC), (SCI, SCI_MVC)])
def test_no_intent_closed_loop_is_quiet(patient, mvc):
    """Key safety scenario: no intent -> reference holds, little stim, no oscillation."""
    log = run(cfg_for("no_intent", patient, **mvc), seed=0).log
    assert metrics.max_excursion(log, J, of="ref") < 0.05
    assert metrics.max_excursion(log, J) < 0.25
    for ch in ("biceps_stim", "triceps_stim"):
        assert metrics.stim_dose(log, ch) < 1.0  # intensity-seconds over 10 s
        assert metrics.time_at_cap(log, ch, 0.8) == 0.0


@pytest.mark.parametrize("patient,mvc,max_rmse", [(None, HEALTHY_MVC, 0.15), (SCI, SCI_MVC, 0.6)])
def test_step_targets_closed_loop_tracks(patient, mvc, max_rmse):
    """Sanity: stable, stim participates, no faults. Pure double integration (D6) tracks
    sci_c5 poorly at the default gain (RMSE ~0.5 rad); that is a finding, not a bug."""
    cfg = cfg_for("step_targets", patient, **mvc)
    res = run(cfg, seed=0)
    log = res.log
    assert metrics.tracking_rmse(log, J, cfg.metrics) < max_rmse
    assert log["stim_biceps_stim"].max() > 0.05  # stim actually participates
    faults = ("watchdog", "sensor_sanity", "emg_rail", "emg_dead", "angle_stale", "angle_frozen")
    assert not any(e.rule in (*faults, "impedance") for e in res.events)  # no false trips


@pytest.mark.xfail(
    strict=True,
    reason="D17 (open): the D14 stim-on deadband (0.16 for sci_c5, set by stim leaking into "
    "the near-paralysed triceps channel) keeps the reference from following the patient",
)
def test_damped_reference_helps_impaired_hold():
    """sci_c5 alone sags below the 1.8 rad target. With reference damping (the D6 candidate
    fix) stim holds it higher. With pure double integration at the default gain it does
    not (closed 1.56 vs open 1.61 rad): see README findings."""
    base = {
        "scenario.duration_s": 7.0,
        **SCI_MVC,
        "controller.joints.r_elbow_flex.damping": 10.0,
    }
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
    [({"controller.blanking_ms": 0}, True), ({}, False)],
    ids=["no_blanking", "hold_fill"],
)
def test_stim_artifact_feedback_and_blanking(blanking, runaway):
    """M3 failure mode: stim -> artifact/M-wave -> envelope -> intent -> more stim.

    sci_c5 with no intent: without blanking the reference runs away into the ROM clamp.
    Sample-and-hold blanking keeps it still.
    """
    log = run(cfg_for("no_intent", SCI, **SCI_MVC, **blanking), seed=0).log
    drift = metrics.max_excursion(log, J, of="ref")
    if runaway:
        assert drift > 0.3
    else:
        assert drift < 0.05


def test_no_intent_stim_off_baseline():
    """D9: no_intent also runs stim-off with the same seed, so the patient's own drift
    (sci_c5: toward flexion, from flexor-dominated resting tone) is separated from
    what the controller does."""
    cfg = cfg_for("no_intent", SCI, **SCI_MVC)
    res = run(cfg, seed=0)
    base = res.baseline_log
    assert base is not None and "stim_triceps_stim" not in base
    assert base[f"q_{J}"].iloc[-1] - base[f"q_{J}"].iloc[0] > 0.1  # flexion, not extension
    np.testing.assert_array_equal(  # same patient noise: stim is the only difference
        res.log["u_vol_BIClong"].to_numpy(), base["u_vol_BIClong"].to_numpy()
    )
    assert metrics.excursion_vs_baseline(res.log, base, J) > 0


def test_faults_and_perturbations_switch_on_and_off():
    """Injected at t_start, removed at t_end, and logged."""
    faults = [
        {"kind": "emg_dropout", "channel": "biceps", "t_start": 0.5, "t_end": 1.0},
        {"kind": "electrode_detach", "channel": "biceps_stim", "t_start": 0.5},
    ]
    perturb = [{"t_start": 0.3, "t_end": 0.6, "torque": {J: 5.0}}]
    ov = {"scenario.duration_s": 1.5, "scenario.faults": faults, "scenario.perturbations": perturb}
    log = run(cfg_for("step_targets", **HEALTHY_MVC, **ov), seed=0).log.set_index("t")
    on = log["fault_0_emg_dropout"].astype(bool)
    assert not on.loc[:0.49].any() and on.loc[0.51:0.99].all() and not on.loc[1.01:].any()
    assert log["fault_1_electrode_detach"].loc[0.51:].all()
    assert (log["perturb_" + J].loc[0.31:0.59] == 5.0).all() and (
        log["perturb_" + J].loc[0.61:] == 0
    ).all()
    env = log["env_biceps"]
    # dropout: only the amplifier noise floor remains (~20% of the resting envelope)
    assert env.loc[0.95:0.99].max() < 0.3 * env.loc[0.45:0.5].mean()


def test_perturbation_moves_the_arm():
    """+torque = flexion while it acts (no_intent, open loop: nothing resists but tone)."""
    ov = {"scenario.duration_s": 1.0, "scenario.closed_loop": False}
    push = {"scenario.perturbations": [{"t_start": 0.3, "t_end": 0.8, "torque": {J: 3.0}}]}
    a = run(cfg_for("no_intent", **ov), seed=0).log.set_index("t")
    b = run(cfg_for("no_intent", **ov, **push), seed=0).log.set_index("t")
    assert b.loc[0.79, f"q_{J}"] > a.loc[0.79, f"q_{J}"] + 0.2
    np.testing.assert_array_equal(a.loc[:0.3, f"q_{J}"], b.loc[:0.3, f"q_{J}"])


def test_fatigue_logged_in_closed_loop():
    log = run(cfg_for("step_targets", SCI, **SCI_MVC, **{"scenario.duration_s": 3.0}), 0).log
    cap = log["cap_BIClong"].to_numpy()
    assert cap[0] <= 1.0 and cap[-1] < cap[0] and np.all(np.diff(cap) <= 1e-12 + 1e-4)


def test_summarize_and_metrics_json(tmp_path):
    cfg = cfg_for("step_targets", **HEALTHY_MVC, **{"scenario.duration_s": 5.0})
    res = run(cfg, seed=0)
    events = pd.DataFrame([vars(e) for e in res.events], columns=["t", "rule", "channel"])
    m = metrics.summarize(cfg, res, events)
    for key in ("tracking_rmse", "steps_reached", "time_to_target_mean", "overshoot_max"):
        assert f"{key}_{J}" in m
    assert 0 < m["final_capacity_min"] <= 1 and "safety_watchdog" in m
    out = write_run(tmp_path, res.log, cfg, 0, metrics=m)
    assert (
        json.loads((out / "metrics.json").read_text())["final_capacity_min"]
        == m["final_capacity_min"]
    )


@pytest.mark.parametrize(
    "fault,rule",
    [
        ({"kind": "emg_saturation", "channel": "triceps"}, "emg_rail"),
        ({"kind": "emg_dropout", "channel": "triceps"}, "emg_dead"),
        ({"kind": "angle_stale"}, "angle_stale"),
        ({"kind": "angle_freeze", "joint": J}, "angle_frozen"),
        ({"kind": "electrode_detach", "channel": "triceps_stim", "contact": 0.3}, "impedance"),
    ],
)
def test_each_fault_trips_its_rule(fault, rule):
    """D10-D12: the supervisor sees each fault class soon after onset. The impedance rule
    needs a stimulator that reports impedance, which this hardware lacks (H7): enabled here
    to test the capability."""
    ov = {
        "scenario.duration_s": 3.0,
        "scenario.faults": [{"t_start": 1.5, **fault}],
        "safety.impedance_max_ohm": 2000.0,
    }
    res = run(cfg_for("step_targets", SCI, **SCI_MVC, **ov), seed=0)
    first = min((e.t for e in res.events if e.rule == rule), default=None)
    assert first is not None and 1.5 <= first < 1.5 + 0.5, (rule, first)
    assert not any(e.rule == rule and e.t < 1.5 for e in res.events)
    after = res.log[res.log["t"] > first + 0.02]
    if rule == "impedance":  # only that channel is cut
        assert (after["stim_triceps_stim"] == 0).all()
    else:  # sensor faults: every channel is cut
        assert (after[["stim_biceps_stim", "stim_triceps_stim"]] == 0).all().all()


def test_partial_contact_raises_current_density():
    faults = [
        {"kind": "electrode_detach", "channel": "biceps_stim", "contact": 0.3, "t_start": 1.0}
    ]
    ov = {"scenario.duration_s": 1.5, "scenario.faults": faults, "safety.impedance_max_ohm": None}
    log = run(cfg_for("step_targets", SCI, **SCI_MVC, **ov), seed=0).log.set_index("t")
    assert log.loc[1.05:, "impedance_biceps_stim"].eq(1000.0 / 0.3).all()
    on = log.loc[1.05:, "stim_biceps_stim"] > 0
    ratio = (
        log.loc[1.05:, "current_density_biceps_stim"][on] / log.loc[1.05:, "stim_biceps_stim"][on]
    )
    np.testing.assert_allclose(ratio, 1 / 0.3)


def test_partial_contact_undetected_without_impedance_reporting():
    """H7: with the real stimulator (no impedance reporting) a partly detached electrode is
    not detected: current density rises unchecked. This is on the hardware risk list."""
    faults = [
        {"kind": "electrode_detach", "channel": "triceps_stim", "contact": 0.3, "t_start": 1.0}
    ]
    ov = {"scenario.duration_s": 3.0, "scenario.faults": faults}
    res = run(cfg_for("step_targets", SCI, **SCI_MVC, **ov), seed=0)
    assert not any(e.rule == "impedance" for e in res.events)
    after = res.log[res.log["t"] > 1.05]
    on = after["stim_triceps_stim"] > 0
    assert (
        on.any()
        and (
            after.loc[on, "current_density_triceps_stim"] > 3 * after.loc[on, "stim_triceps_stim"]
        ).all()
    )


def test_lead_off_latch_and_operator_reset_in_closed_loop():
    """D16: an intermittent lead (off 1-1.5 s) keeps stim off until an operator reset, which
    is refused while the lead is off and accepted once it has been live for 3 s."""
    faults = [{"kind": "emg_dropout", "channel": "triceps", "t_start": 1.0, "t_end": 1.5}]
    ov = {
        "scenario.duration_s": 6.0,
        "scenario.faults": faults,
        "scenario.operator_resets": [1.4, 5.0],
    }
    log = run(cfg_for("step_targets", SCI, **SCI_MVC, **ov), seed=0).log.set_index("t")
    assert log.loc[1.6:4.9, "safety_emg_dead"].astype(bool).all()  # still latched after lead back
    assert (log.loc[1.6:4.9, ["stim_biceps_stim", "stim_triceps_stim"]] == 0).all().all()
    attempts = log["operator_reset_accepted"].dropna()
    assert list(attempts.round(0)) == [0.0, 1.0]  # refused at 1.4 s, accepted at 5.0 s
    assert not log.loc[5.05:, "safety_emg_dead"].astype(bool).any()
