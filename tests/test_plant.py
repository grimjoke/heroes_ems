import numpy as np
import pytest

from heroes_sim.config import load_config
from heroes_sim.plant import Plant

MUSCLES = ["BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed"]


@pytest.fixture
def cfg():
    return load_config("configs/base.yaml")


def run(plant, u, seconds):
    for _ in range(int(seconds / plant.dt)):
        plant.step(u)


def test_model_muscles_present(cfg):
    plant = Plant(cfg.plant, cfg.timing)
    assert plant.n_muscles == 6 and plant.n_joints == 1
    assert plant.dt == pytest.approx(0.0005)


def test_unknown_muscle_fails_loudly(cfg):
    bad = cfg.plant.model_copy(update={"muscles": ["NOPE"]})
    with pytest.raises(ValueError, match="NOPE"):
        Plant(bad, cfg.timing)


def test_passive_drop_settles_hanging(cfg):
    plant = Plant(cfg.plant, cfg.timing)
    zero = np.zeros(6)
    ends = []
    for q0 in (1.5, 0.1):  # released from above and below: same hanging posture
        plant.reset(np.array([q0]), np.zeros(1))
        run(plant, zero, 4.0)
        q, qd = plant.joint_state()
        assert abs(qd[0]) < 1e-2
        ends.append(q[0])
    assert ends[0] == pytest.approx(ends[1], abs=0.05)
    assert ends[0] < 1.0  # gravity pulled it down from 1.5


def test_biceps_flexes(cfg):
    plant = Plant(cfg.plant, cfg.timing)
    run(plant, np.zeros(6), 3.0)
    q_rest = plant.joint_state()[0][0]
    u = np.zeros(6)
    u[[0, 1, 2]] = 1.0
    run(plant, u, 1.5)
    assert plant.joint_state()[0][0] > q_rest + 0.5


def test_muscle_state_shapes(cfg):
    plant = Plant(cfg.plant, cfg.timing)
    run(plant, np.ones(6) * 0.5, 0.2)
    s = plant.muscle_state()
    assert s.activation.shape == s.force.shape == s.length.shape == (6,)
    assert np.all(s.activation > 0)


def test_exo_payload_lowers_rest_angle_shift(cfg):
    heavy = cfg.plant.model_copy(
        update={"exo_payload": cfg.plant.exo_payload.model_copy(update={"mass_kg": 2.0})}
    )
    base, loaded = Plant(cfg.plant, cfg.timing), Plant(heavy, cfg.timing)
    for p in (base, loaded):
        run(p, np.zeros(6), 4.0)
    assert loaded.joint_state()[0][0] < base.joint_state()[0][0]
