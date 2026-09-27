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
            overrides={"scenario.trials": [{"group": "hamstrings", "effort_s": 2.0}]},
        )
