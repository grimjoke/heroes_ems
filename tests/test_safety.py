"""Each supervisor rule has a test that triggers it (spec 9)."""

import numpy as np
import pytest

from heroes_safety import SafetyConfig, SafetySupervisor


def config(**kw):
    base = {
        "tick_hz": 100.0,
        "channel_joint": (0, 0),
        "channel_sign": (1, -1),
        "cap": (0.8, 0.8),
        "max_rise_per_tick": 0.05,
        "q_limit_min": (0.0,),
        "q_limit_max": (2.27,),
        "rom_margin_rad": 0.1,
        "dose_window_s": 1.0,
        "dose_max_mean": 0.5,
        "watchdog_timeout_s": 0.05,
        "q_plausible_min": (-0.3,),
        "q_plausible_max": (2.6,),
        "emg_plausible_abs": 20.0,
    }
    return SafetyConfig(**{**base, **kw})


EMG = np.zeros((20, 2))
Q = np.array([1.0])


def ramp_to(sup, cmd, n=40, t0=0.0, q=Q):
    out = None
    for i in range(n):
        t = t0 + i * 0.01
        out = sup.step(t, np.asarray(cmd, dtype=float), t, q, EMG)
    return out


def test_passes_safe_command():
    sup = SafetySupervisor(config(max_rise_per_tick=1.0))
    out = sup.step(0.0, np.array([0.3, 0.0]), 0.0, Q, EMG)
    np.testing.assert_allclose(out.intensity, [0.3, 0.0])
    assert not any(f.any() for f in out.fired.values()) and sup.events == []


@pytest.mark.parametrize("cmd", [None, np.array([np.nan, 0.2]), np.array([np.inf, 0.0])])
def test_watchdog_bad_command_zeroes_all(cmd):
    sup = SafetySupervisor(config())
    ramp_to(sup, [0.5, 0.5])
    out = sup.step(1.0, cmd, 1.0, Q, EMG)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["watchdog"].all()
    assert sup.events[-1].rule == "watchdog"


def test_watchdog_stale_command():
    sup = SafetySupervisor(config())
    ramp_to(sup, [0.5, 0.0])
    out = sup.step(1.0, np.array([0.5, 0.0]), 1.0 - 0.06, Q, EMG)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["watchdog"].all()


@pytest.mark.parametrize(
    "q,emg",
    [
        (np.array([np.nan]), EMG),
        (np.array([3.0]), EMG),
        (Q, np.full((20, 2), 25.0)),
        (Q, np.array([[np.nan, 0.0]])),
    ],
)
def test_sensor_sanity_zeroes_all(q, emg):
    sup = SafetySupervisor(config())
    ramp_to(sup, [0.5, 0.5])
    out = sup.step(1.0, np.array([0.5, 0.5]), 1.0, q, emg)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["sensor_sanity"].all()


def test_rom_guard_cuts_only_the_channel_driving_into_the_limit():
    sup = SafetySupervisor(config(max_rise_per_tick=1.0))
    out = sup.step(0.0, np.array([0.5, 0.5]), 0.0, np.array([2.2]), EMG)  # near max
    np.testing.assert_allclose(out.intensity, [0.0, 0.5])
    assert out.fired["rom_guard"].tolist() == [True, False]
    out = sup.step(0.01, np.array([0.5, 0.5]), 0.01, np.array([0.05]), EMG)  # near min
    np.testing.assert_allclose(out.intensity, [0.5, 0.0])


def test_cap():
    sup = SafetySupervisor(config(max_rise_per_tick=1.0, cap=(0.3, 0.8)))
    out = sup.step(0.0, np.array([0.9, 0.9]), 0.0, Q, EMG)
    np.testing.assert_allclose(out.intensity, [0.3, 0.8])
    assert out.fired["cap"].tolist() == [True, True]


def test_rate_limit_rises_slowly_falls_immediately():
    sup = SafetySupervisor(config(dose_max_mean=1.0))
    first = sup.step(0.0, np.array([0.5, 0.0]), 0.0, Q, EMG)
    assert first.intensity[0] == pytest.approx(0.05) and first.fired["rate_limit"][0]
    out = ramp_to(sup, [0.5, 0.0], n=20, t0=0.01)
    assert out.intensity[0] == pytest.approx(0.5)
    drop = sup.step(1.0, np.array([0.0, 0.0]), 1.0, Q, EMG)
    assert drop.intensity[0] == 0.0 and not drop.fired["rate_limit"].any()


def test_dose_limit():
    sup = SafetySupervisor(config(max_rise_per_tick=1.0, dose_window_s=0.5, dose_max_mean=0.25))
    window = 50
    outs = [sup.step(i * 0.01, np.array([0.8, 0.0]), i * 0.01, Q, EMG) for i in range(200)]
    s = np.array([o.intensity[0] for o in outs])
    rolling = np.convolve(s, np.ones(window), "valid") / window
    assert rolling.max() <= 0.25 + 1e-12
    assert any(o.fired["dose"][0] for o in outs)
    assert s[0] == pytest.approx(0.8)  # budget available at first


def test_events_are_rising_edges():
    sup = SafetySupervisor(config(max_rise_per_tick=1.0, cap=(0.3, 0.8)))
    for i in range(5):
        sup.step(i * 0.01, np.array([0.5, 0.0]), i * 0.01, Q, EMG)
    sup.step(0.05, np.array([0.1, 0.0]), 0.05, Q, EMG)
    sup.step(0.06, np.array([0.5, 0.0]), 0.06, Q, EMG)
    caps = [e for e in sup.events if e.rule == "cap"]
    assert [(round(e.t, 2), e.channel) for e in caps] == [(0.0, 0), (0.06, 0)]


def test_config_validated():
    with pytest.raises(ValueError):
        config(cap=(1.5, 0.8))
    with pytest.raises(ValueError):
        config(channel_sign=(1,))


@pytest.mark.parametrize(
    "fault",
    ["watchdog", "sensor_sanity"],
)
def test_fault_paths_bypass_rate_limit(fault):
    """D7: faults cut to zero in one tick, regardless of the rate limit; after the fault
    clears, stim ramps back up from zero at the rate limit (no jump back)."""
    sup = SafetySupervisor(config(dose_max_mean=1.0))
    ramp_to(sup, [0.5, 0.0])
    if fault == "watchdog":
        out = sup.step(1.0, None, None, Q, EMG)
    else:
        out = sup.step(1.0, np.array([0.5, 0.0]), 1.0, np.array([np.nan]), EMG)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert not out.fired["rate_limit"].any()
    back = sup.step(1.01, np.array([0.5, 0.0]), 1.01, Q, EMG)
    assert back.intensity[0] == pytest.approx(0.05) and back.fired["rate_limit"][0]


# ---- D10-D12 detectors -------------------------------------------------------------


def detectors(**kw):
    base = {
        "max_rise_per_tick": 1.0,
        "dose_max_mean": 1.0,
        "emg_rail": 10.0,
        "emg_rail_min_samples": 10,
        "emg_dead_ratio": 0.5,
        "emg_dead_ticks": 10,
        "angle_max_age_s": 0.05,
        "angle_repeat_ticks": 10,
        "impedance_max_ohm": 2000.0,
    }
    return config(**{**base, **kw})


REST_STD = (0.02, 0.02)
Z_OK = np.array([1000.0, 1000.0])


def live_emg(rng, n=20, std=0.02):
    return std * rng.standard_normal((n, 2))


def tick(sup, i, emg, q=None, stamp=None, z=Z_OK, cmd=(0.3, 0.3)):
    t = i * 0.01
    q = np.array([1.0 + 1e-3 * (i % 7)]) if q is None else q  # varying reading
    return sup.step(
        t, np.array(cmd), t, q, emg, q_stamp=t - 0.01 if stamp is None else stamp, impedance=z
    )


def test_detectors_quiet_on_healthy_signals():
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    for i in range(300):
        out = tick(sup, i, live_emg(rng))
        np.testing.assert_allclose(out.intensity, [0.3, 0.3])
    assert sup.events == []


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_emg_rail_either_sign(sign):
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    emg = live_emg(rng)
    emg[:12, 1] = sign * 9.9  # 12 samples at >= 0.98 x rail, below the 20 mV plausible limit
    out = tick(sup, 1, emg)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["emg_rail"].all() and not out.fired["sensor_sanity"].any()


def test_emg_rail_ignores_brief_spikes():
    """A few rail samples per chunk (big stim artifacts) are not a stuck lead."""
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    emg = live_emg(rng)
    emg[:3, 0] = 10.0
    assert not tick(sup, 1, emg).fired["emg_rail"].any()


def test_emg_dead_channel_after_n_quiet_ticks():
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    for i in range(9):
        emg = live_emg(rng)
        emg[:, 1] = 0.005 * rng.standard_normal(20)  # amplifier noise only: ratio 0.25
        assert not tick(sup, i, emg).fired["emg_dead"].any()
    out = tick(sup, 9, emg)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["emg_dead"].all()
    # hysteresis: one live-looking tick does not clear it; N consecutive live ticks do
    outs = [tick(sup, 10 + i, live_emg(rng)) for i in range(10)]
    assert all(o.fired["emg_dead"].all() for o in outs[:9])
    assert not outs[9].fired["emg_dead"].any()


def test_emg_dead_needs_calibrated_rest_std():
    with pytest.raises(ValueError, match="emg_rest_std"):
        SafetySupervisor(detectors())


def test_angle_stale_message_zeroes_all():
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    out = tick(sup, 10, live_emg(rng), stamp=0.10 - 0.06)  # 60 ms old > 50 ms
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["angle_stale"].all()
    missing = sup.step(0.2, np.array([0.3, 0.3]), 0.2, np.array([1.0]), live_emg(rng))
    assert missing.fired["angle_stale"].all()  # configured but no stamp -> stale


def test_angle_frozen_after_n_identical_readings():
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    q = np.array([1.234])
    outs = [tick(sup, i, live_emg(rng), q=q) for i in range(12)]
    assert not outs[9].fired["angle_frozen"].any()  # 9 repeats
    assert outs[10].fired["angle_frozen"].all()  # 10th repeat trips
    np.testing.assert_array_equal(outs[10].intensity, 0.0)


@pytest.mark.parametrize("z", [np.array([2500.0, 1000.0]), np.array([np.inf, 1000.0])])
def test_impedance_cuts_only_that_channel(z):
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    out = tick(sup, 1, live_emg(rng), z=z)
    np.testing.assert_allclose(out.intensity, [0.0, 0.3])
    assert out.fired["impedance"].tolist() == [True, False]


def test_impedance_missing_reading_counts_as_bad():
    rng = np.random.default_rng(0)
    sup = SafetySupervisor(detectors(), REST_STD)
    t = 0.01
    out = sup.step(t, np.array([0.3, 0.3]), t, np.array([1.0]), live_emg(rng), q_stamp=t)
    np.testing.assert_array_equal(out.intensity, 0.0)
    assert out.fired["impedance"].all()


def test_detector_config_validated():
    with pytest.raises(ValueError, match="emg_rail"):
        config(emg_rail_min_samples=5)  # count without a rail
    with pytest.raises(ValueError):
        config(emg_dead_ratio=1.5)
