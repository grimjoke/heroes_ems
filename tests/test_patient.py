import numpy as np
import pytest

from heroes_sim.config import load_run_config
from heroes_sim.patient import NO_INTENT, Intent, Patient

DT = 0.0005
FLEX, EXT = [0, 1, 2], [3, 4, 5]
ACTION = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])


@pytest.fixture
def cfg():
    return load_run_config("configs/scenarios/step_targets.yaml")


def make(cfg, seed=0, **patient):
    pc = cfg.patient.model_copy(update=patient)
    return Patient(pc, cfg.plant, ACTION, DT, np.random.default_rng(seed))


def test_no_intent_is_cocontraction_only(cfg):
    p = make(cfg, noise_std=0.0, onset_delay_ms=0.0)
    u = p.step(NO_INTENT, np.array([1.0]), np.zeros(1))
    np.testing.assert_allclose(u, cfg.patient.cocontraction)


def test_agonist_antagonist_split(cfg):
    p = make(cfg, noise_std=0.0, onset_delay_ms=0.0, cocontraction=0.0)
    up = p.step(Intent(target=np.array([1.5])), np.array([1.0]), np.zeros(1))
    assert np.all(up[FLEX] > 0) and np.all(up[EXT] == 0)
    down = p.step(Intent(target=np.array([0.5])), np.array([1.0]), np.zeros(1))
    assert np.all(down[EXT] > 0) and np.all(down[FLEX] == 0)


def test_paralysed_muscle_never_fires(cfg):
    strength = dict(cfg.patient.strength, BIClong=0.0)
    p = make(cfg, strength=strength)
    for _ in range(200):
        u = p.step(Intent(mvc_group="elbow_flexors"), np.array([1.0]), np.zeros(1))
        assert u[0] == 0.0 and np.all(u >= 0) and np.all(u <= 1)


def test_mvc_drives_group_at_strength(cfg):
    strength = dict(cfg.patient.strength, BIClong=0.3)
    p = make(cfg, strength=strength, noise_std=0.0, onset_delay_ms=0.0, cocontraction=0.1)
    u = p.step(Intent(mvc_group="elbow_flexors"), np.array([1.0]), np.zeros(1))
    np.testing.assert_allclose(u, [0.3, 1.0, 1.0, 0.1, 0.1, 0.1])


def test_onset_delay(cfg):
    p = make(cfg, noise_std=0.0, onset_delay_ms=10.0, cocontraction=0.0)
    delay = round(10e-3 / DT)
    intent = Intent(target=np.array([1.5]))
    outs = [p.step(intent, np.array([1.0]), np.zeros(1))[0] for _ in range(delay + 1)]
    assert all(o == 0.0 for o in outs[:delay]) and outs[delay] > 0


def test_noise_deterministic_per_seed(cfg):
    runs = []
    for seed in (1, 1, 2):
        p = make(cfg, seed=seed)
        runs.append([p.step(NO_INTENT, np.zeros(1), np.zeros(1)) for _ in range(50)])
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])
