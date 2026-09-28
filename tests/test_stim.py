import numpy as np
import pytest

from heroes_sim.config import RecruitmentConfig, load_config
from heroes_sim.stim import StimModel, probabilistic_sum, quantize, recruitment

REC = RecruitmentConfig(threshold=0.35, slope=12.0, saturation=0.9)


@pytest.fixture
def cfg():
    return load_config("configs/base.yaml")


def test_zero_intensity_zero_excitation(cfg):
    stim = StimModel(cfg.stim, cfg.plant.muscles, 0.0005)
    stim.pulse(0.0, np.zeros(2))
    for _ in range(200):
        np.testing.assert_array_equal(stim.step(), 0.0)
    assert recruitment(np.array([0.0]), REC)[0] == 0.0


def test_recruitment_monotonic_and_saturates():
    s = np.linspace(0, 1, 1001)
    r = recruitment(s, REC)
    assert np.all(np.diff(r) > 0)
    assert r[-1] == pytest.approx(0.9)


def test_quantization_respected(cfg):
    q = quantize(np.linspace(0, 1, 997), 128)
    levels = np.arange(128) / 127
    assert np.all(np.isin(np.round(q * 127), np.round(levels * 127)))
    assert np.max(np.abs(q - np.linspace(0, 1, 997))) <= 0.5 / 127 + 1e-12
    stim = StimModel(cfg.stim, cfg.plant.muscles, 0.0005)
    ev = stim.pulse(0.0, np.array([0.3333, 0.0]))
    assert ev.intensity[0] * 127 == pytest.approx(round(0.3333 * 127))


def test_electrode_matrix_and_crosstalk(cfg):
    stim = StimModel(cfg.stim, cfg.plant.muscles, 0.0005)
    stim.pulse(0.0, np.array([1.0, 0.0]))
    for _ in range(round(cfg.stim.em_delay_ms * 1e-3 / 0.0005) + 1):
        u = stim.step()
    flex = [cfg.plant.muscles.index(m) for m in ("BIClong", "BICshort", "BRA")]
    ext = [cfg.plant.muscles.index(m) for m in ("TRIlong", "TRIlat", "TRImed")]
    assert np.all(u[ext] == 0) and u[flex[0]] > u[flex[2]] > 0  # BRA via crosstalk only


def test_pulse_hold_and_em_delay(cfg):
    stim = StimModel(cfg.stim, cfg.plant.muscles, 0.0005)
    delay = round(cfg.stim.em_delay_ms * 1e-3 / 0.0005)
    stim.pulse(0.0, np.array([0.8, 0.0]))
    outs = [stim.step()[0] for _ in range(delay + 5)]
    assert all(o == 0.0 for o in outs[:delay]) and outs[delay] > 0
    assert len(set(outs[delay:])) == 1  # held constant between pulses


def test_probabilistic_sum():
    np.testing.assert_allclose(probabilistic_sum(np.array([0.5]), np.array([0.5])), [0.75])
    np.testing.assert_allclose(probabilistic_sum(np.array([0.3]), np.array([0.0])), [0.3])
