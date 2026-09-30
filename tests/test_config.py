import pytest
from pydantic import ValidationError

from heroes_sim.config import load_run_config

SC = "configs/scenarios/step_targets.yaml"


def test_compose_scenario_patient():
    cfg = load_run_config(SC)
    assert cfg.scenario.name == "step_targets" and cfg.patient.name == "healthy"
    sci = load_run_config(SC, patient_path="configs/patients/sci_c5.yaml")
    assert sci.patient.name == "sci_c5"


def test_override_into_patient():
    cfg = load_run_config(SC, overrides={"patient.onset_delay_ms": 0})
    assert cfg.patient.onset_delay_ms == 0


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"patient.strength": {"BIClong": 1.0}}, "patient.strength muscles"),
        ({"patient.strength.BIClong": 1.5}, "strength must be in"),
        ({"scenario.initial_state.q": [0.5, 0.5]}, "length 1"),
        ({"plant.muscle_groups.elbow_flexors.muscles": ["NOPE"]}, "unknown muscle"),
        ({"plant.muscle_groups.elbow_flexors.muscles": ["TRIlong"]}, "in groups"),
    ],
)
def test_invalid_rejected(overrides, match):
    with pytest.raises(ValidationError, match=match):
        load_run_config(SC, overrides=overrides)


def test_mvc_group_must_exist():
    with pytest.raises(ValidationError, match="not in plant.muscle_groups"):
        load_run_config(
            "configs/scenarios/mvc_calibration.yaml",
            overrides={"calibration.trials": [{"group": "hamstrings", "effort_s": 2.0}]},
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"controller.joints.r_elbow_flex.agonist": "deltoid"}, "EMG channel"),
        ({"safety.cap": {"biceps_stim": 0.8}}, "every stim channel"),
        ({"stim.channels.0.joint": "knee"}, "unknown"),
        (
            {"scenario.calibrated": {"envelope": {"biceps": 0.5}, "rest": {}, "deadband": {}}},
            "every EMG channel",
        ),
        ({"controller.allocation": {"offset": 0.3, "epsilon": 0.0}}, "epsilon"),
        ({"timing.emg_hz": 1000}, "emg_hz must equal"),
    ],
)
def test_cross_section_names_checked(overrides, match):
    # stim.channels is a list; the dotted-override helper indexes dicts only.
    if "stim.channels.0.joint" in overrides:
        cfg = load_run_config(SC)
        raw = cfg.model_dump()
        raw["stim"]["channels"][0]["joint"] = "knee"
        raw["scenario"] = cfg.scenario.model_dump()
        with pytest.raises(ValidationError, match=match):
            type(cfg).model_validate(raw)
        return
    with pytest.raises(ValidationError, match=match):
        load_run_config(SC, overrides=overrides)


@pytest.mark.parametrize(
    "fault,match",
    [
        ({"kind": "electrode_detach", "channel": "nope", "t_start": 1.0}, "unknown"),
        ({"kind": "emg_dropout", "channel": "biceps_stim", "t_start": 1.0}, "unknown"),
        ({"kind": "angle_freeze", "joint": "knee", "t_start": 1.0}, "unknown"),
        ({"kind": "artifact_increase", "factor": 5, "t_start": 2.0, "t_end": 1.0}, "t_end"),
        ({"kind": "artifact_increase", "factor": 0.5, "t_start": 1.0}, "factor"),
    ],
)
def test_faults_validated(fault, match):
    with pytest.raises(ValidationError, match=match):
        load_run_config(SC, overrides={"scenario.faults": [fault]})


def test_saturation_fault_needs_a_rail():
    fault = {"kind": "emg_saturation", "channel": "biceps", "t_start": 1.0}
    with pytest.raises(ValidationError, match="saturation_mv"):
        load_run_config(SC, overrides={"scenario.faults": [fault], "emg.saturation_mv": None})


def test_fatigue_hold_scenario_loads():
    cfg = load_run_config("configs/scenarios/fatigue_hold.yaml")
    assert cfg.patient.name == "sci_c5" and cfg.scenario.duration_s >= 60


def test_pinned_calibration_needs_rest_std_when_dead_detector_on():
    pinned = {
        "envelope": {"biceps": 0.5, "triceps": 0.4},
        "rest": {"biceps": 0.01, "triceps": 0.01},
        "deadband": {"r_elbow_flex": 0.05},
    }
    with pytest.raises(ValidationError, match="emg_rest_std"):
        load_run_config(SC, overrides={"scenario.calibrated": pinned})
    load_run_config(
        SC, overrides={"scenario.calibrated": pinned, "safety.emg_dead_ratio": 0.0}
    )  # detector off -> not needed
