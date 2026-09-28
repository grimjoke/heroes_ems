import numpy as np
import pytest

from heroes_sim.config import load_config
from heroes_sim.sensors.emg import EMGSensor
from heroes_sim.sensors.kinematics import AngleSensor


@pytest.fixture
def cfg():
    return load_config("configs/base.yaml")


def sensor(cfg, emg_cfg=None, seed=0):
    rng = np.random.default_rng
    return EMGSensor(emg_cfg or cfg.emg, cfg.plant.muscles, 2000.0, rng(seed), rng(seed + 1))


def test_emg_amplitude_follows_excitation(cfg):
    u = np.zeros((20000, 6))
    u[:10000, :3] = 0.2
    u[10000:, :3] = 0.8
    emg = sensor(cfg).chunk(u)
    lo, hi = emg[2000:10000, 0].std(), emg[12000:, 0].std()
    assert hi / lo == pytest.approx(4.0, rel=0.1)
    # triceps channel only sees biceps crosstalk (weights 0.05 vs 1.0)
    assert emg[12000:, 1].std() < 0.1 * hi


def test_emg_unit_carrier_rms(cfg):
    u = np.zeros((40000, 6))
    u[:, 0] = 1.0
    no_noise = cfg.emg.model_copy(
        update={"white_noise": cfg.emg.white_noise.model_copy(update={"enabled": False})}
    )
    emg = sensor(cfg, no_noise).chunk(u)
    assert emg[2000:, 0].std() == pytest.approx(
        cfg.emg.volitional.amplitude_mv["BIClong"], rel=0.05
    )


def test_emg_components_toggle(cfg):
    off = cfg.emg.model_copy(
        update={
            "volitional": cfg.emg.volitional.model_copy(update={"enabled": False}),
            "white_noise": cfg.emg.white_noise.model_copy(update={"enabled": False}),
        }
    )
    np.testing.assert_array_equal(sensor(cfg, off).chunk(np.ones((100, 6))), 0.0)


def test_angle_sensor_latency_and_quantization(cfg):
    clean = cfg.angle_sensor.model_copy(
        update={"noise_std_rad": 0.0, "bias_walk_std_rad_per_sqrt_s": 0.0}
    )
    s = AngleSensor(clean, 1, 0.0005, 0.01, np.random.default_rng(0))
    s.reset(np.array([0.5]))
    delay = round(clean.latency_ms * 1e-3 / 0.0005)
    for _ in range(delay):
        s.push(np.array([1.23456]))
    assert s.sample()[0] == 0.5  # not arrived yet
    s.push(np.array([1.23456]))
    assert s.sample()[0] == pytest.approx(1.235)  # 1 mrad resolution


def test_angle_sensor_noise_and_bias(cfg):
    s = AngleSensor(cfg.angle_sensor, 1, 0.0005, 0.01, np.random.default_rng(0))
    s.reset(np.array([1.0]))
    x = np.array([s.sample()[0] for _ in range(1000)])
    assert 0.001 < x.std() < 0.05 and abs(x.mean() - 1.0) < 0.05
