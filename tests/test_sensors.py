import numpy as np
import pytest

from heroes_sim.config import AmplifierConfig, load_config
from heroes_sim.sensors.emg import EMGSensor
from heroes_sim.sensors.kinematics import AngleSensor
from heroes_sim.stim import PulseEvent

CONTAMINATION = ("stim_artifact", "m_wave", "powerline", "baseline_wander", "motion")


@pytest.fixture
def cfg():
    return load_config("configs/base.yaml")


def only(cfg, *enabled, saturation=None):
    """EMG config with just the named components on (volitional/white_noise included)."""
    update = {}
    for name in ("volitional", "white_noise", *CONTAMINATION):
        sub = getattr(cfg.emg, name)
        update[name] = sub.model_copy(update={"enabled": name in enabled})
    update["amplifier"] = (
        None
        if saturation is None
        # rail = supply / 2 / gain: pick the supply that gives the requested rail, ideal recovery
        else AmplifierConfig(
            supply_v=2 * saturation * 200.0 / 1000.0, raw_gain=200.0, overload_recovery_ms=0.0
        )
    )
    return cfg.emg.model_copy(update=update)


def pulse(intensity, muscle_recruitment):
    return PulseEvent(
        0.0, np.asarray(intensity, float), np.zeros(2), np.asarray(muscle_recruitment, float)
    )


def sensor(cfg, emg_cfg=None, seed=0):
    streams = {}

    def rng(name):
        streams.setdefault(name, np.random.default_rng([seed, len(streams)]))
        return streams[name]

    stim = [ch.name for ch in cfg.stim.channels]
    return EMGSensor(emg_cfg or cfg.emg, cfg.plant.muscles, cfg.plant.joints, stim, 2000.0, rng)


def test_emg_amplitude_follows_excitation(cfg):
    u = np.zeros((20000, 6))
    u[:10000, :3] = 0.2
    u[10000:, :3] = 0.8
    emg = sensor(cfg, only(cfg, "volitional", "white_noise")).chunk(u)
    lo, hi = emg[2000:10000, 0].std(), emg[12000:, 0].std()
    assert hi / lo == pytest.approx(4.0, rel=0.1)
    # triceps channel only sees biceps crosstalk (weights 0.05 vs 1.0)
    assert emg[12000:, 1].std() < 0.1 * hi


def test_emg_unit_carrier_rms(cfg):
    u = np.zeros((40000, 6))
    u[:, 0] = 1.0
    emg = sensor(cfg, only(cfg, "volitional")).chunk(u)
    assert emg[2000:, 0].std() == pytest.approx(
        cfg.emg.volitional.amplitude_mv["BIClong"], rel=0.05
    )


def test_emg_components_toggle(cfg):
    s = sensor(cfg, only(cfg))
    s.add_pulse(0, pulse([1.0, 1.0], np.ones(6)))
    np.testing.assert_array_equal(s.chunk(np.ones((100, 6)), np.ones((100, 1))), 0.0)


def test_stim_artifact_scales_with_intensity_and_proximity(cfg):
    s = sensor(cfg, only(cfg, "stim_artifact"))
    s.add_pulse(3, pulse([0.5, 0.0], np.zeros(6)))  # biceps electrode only
    emg = s.chunk(np.zeros((100, 6)))
    assert np.all(emg[:3] == 0)
    a = cfg.emg.stim_artifact.amplitude_mv
    assert emg[3, 0] == pytest.approx(0.5 * a["biceps"]["biceps_stim"])
    assert emg[3, 1] == pytest.approx(0.5 * a["triceps"]["biceps_stim"])
    decay = round(cfg.emg.stim_artifact.decay_ms * 2)  # samples per decay constant at 2 kHz
    assert emg[3 + decay, 0] == pytest.approx(emg[3, 0] * np.exp(-1), rel=1e-6)


def test_m_wave_latency_and_recruitment(cfg):
    s = sensor(cfg, only(cfg, "m_wave"))
    rec = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # biceps heads half recruited
    s.add_pulse(0, pulse([0.5, 0.0], rec))
    emg = s.chunk(np.zeros((60, 6)))
    lat = round(cfg.emg.m_wave.latency_ms * 2)
    assert np.all(emg[: lat + 1] == 0) and np.abs(emg[lat + 1 :, 0]).max() > 0
    peak = 0.5 * (cfg.emg.m_wave.amplitude_mv["BIClong"] + cfg.emg.m_wave.amplitude_mv["BICshort"])
    assert np.abs(emg[:, 0]).max() == pytest.approx(peak, rel=0.05)
    assert np.abs(emg[:, 1]).max() < 0.1 * peak  # triceps channel: crosstalk only


def test_pulse_waveforms_carry_across_chunks(cfg):
    deterministic = only(cfg, "stim_artifact", "m_wave", "powerline")
    whole, parts = sensor(cfg, deterministic), sensor(cfg, deterministic)
    p = pulse([0.7, 0.2], np.full(6, 0.3))
    whole.add_pulse(5, p)
    parts.add_pulse(5, p)
    a = whole.chunk(np.zeros((100, 6)))
    b = np.concatenate([parts.chunk(np.zeros((k, 6))) for k in (7, 1, 20, 72)])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_powerline_at_mains_frequency(cfg):
    emg = sensor(cfg, only(cfg, "powerline")).chunk(np.zeros((4000, 6)))
    spec = np.abs(np.fft.rfft(emg[:, 0]))
    freqs = np.fft.rfftfreq(4000, 1 / 2000)
    assert freqs[np.argmax(spec)] == pytest.approx(50.0)
    assert spec[np.argmin(np.abs(freqs - 100))] > 0.1 * spec.max()  # harmonic present


def test_baseline_wander_is_slow(cfg):
    emg = sensor(cfg, only(cfg, "baseline_wander")).chunk(np.zeros((40000, 6)))[:, 0]
    assert emg.std() > 0.01
    assert np.abs(np.diff(emg)).max() < 0.01 * emg.std() * 10


def test_motion_artifact_follows_velocity(cfg):
    s = sensor(cfg, only(cfg, "motion"))
    qd = np.linspace(-2, 2, 50)[:, None]
    emg = s.chunk(np.zeros((50, 6)), qd)
    np.testing.assert_allclose(
        emg[:, 0], cfg.emg.motion.gain_mv_per_rad_s["biceps"]["r_elbow_flex"] * qd[:, 0]
    )


def test_amplifier_saturation(cfg):
    s = sensor(cfg, only(cfg, "stim_artifact", saturation=2.0))
    s.add_pulse(0, pulse([1.0, 0.0], np.zeros(6)))
    emg = s.chunk(np.zeros((10, 6)))
    assert emg.max() == 2.0


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


def test_emg_channel_faults(cfg):
    s = sensor(cfg)
    s.channel_fault = {0: "dropout", 1: "saturation"}
    emg = s.chunk(np.full((50, 6), 0.5))
    # dropout: only the amplifier noise floor remains; saturation: stuck at the rail
    assert emg[:, 0].std() == pytest.approx(cfg.emg.white_noise.std_mv, rel=0.3)
    assert np.all(emg[:, 1] == cfg.emg.saturation_mv)


def test_artifact_increase(cfg):
    base, big = sensor(cfg, only(cfg, "stim_artifact")), sensor(cfg, only(cfg, "stim_artifact"))
    big.artifact_gain = 3.0
    p = pulse([0.2, 0.0], np.zeros(6))
    base.add_pulse(0, p)
    big.add_pulse(0, p)
    np.testing.assert_allclose(big.chunk(np.zeros((40, 6))), 3.0 * base.chunk(np.zeros((40, 6))))


def test_angle_sensor_freeze(cfg):
    s = AngleSensor(cfg.angle_sensor, 1, 0.0005, 0.01, np.random.default_rng(0))
    s.reset(np.array([1.0]))
    last = s.sample()
    s.frozen[0] = True
    for q in (1.5, 2.0):
        for _ in range(40):
            s.push(np.array([q]))
        assert s.sample()[0] == last[0]
    s.frozen[0] = False
    assert s.sample()[0] == pytest.approx(2.0, abs=0.05)


def test_angle_sensor_stamps_and_stale(cfg):
    s = AngleSensor(cfg.angle_sensor, 1, 0.0005, 0.01, np.random.default_rng(0))
    s.reset(np.array([1.0]))
    s.sample(0.5)
    assert s.stamp == pytest.approx(0.5 - cfg.angle_sensor.latency_ms * 1e-3)
    last = s.sample(0.51)
    s.stale = True
    for _ in range(40):
        s.push(np.array([2.0]))
    assert s.sample(0.6)[0] == last[0] and s.stamp == pytest.approx(0.51 - 0.01)


def test_rail_from_supply_and_gain(cfg):
    """H1: output 0..Vs centred at Vs/2 -> input-referred rail Vs / 2 / gain."""
    amp = cfg.emg.amplifier
    assert cfg.emg.saturation_mv == pytest.approx(1000 * amp.supply_v / 2 / amp.raw_gain)
    five = cfg.emg.model_copy(update={"amplifier": amp.model_copy(update={"supply_v": 5.0})})
    assert five.saturation_mv == pytest.approx(12.5)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_overload_recovery(cfg, sign):
    """H2: after saturating, the output returns to the signal with time constant tau."""
    rec = cfg.emg.model_copy(
        update={
            "amplifier": cfg.emg.amplifier.model_copy(update={"overload_recovery_ms": 5.0}),
            **{
                k: getattr(cfg.emg, k).model_copy(update={"enabled": False})
                for k in ("volitional", "white_noise", *CONTAMINATION)
            },
        }
    )
    s = sensor(cfg, rec)
    rail = rec.saturation_mv
    x = np.zeros((200, 2))
    x[:10, 0] = sign * 3 * rail  # driven past the rail, then back to 0
    y = s._amplifier(x)
    assert np.all(y[:10, 0] == sign * rail)  # clipped at either rail
    tau = round(5e-3 * 2000)
    assert y[10 + tau - 1, 0] == pytest.approx(sign * rail * np.exp(-1), rel=0.02)
    assert abs(y[-1, 0]) < 1e-3 * rail and np.all(y[:, 1] == 0)
