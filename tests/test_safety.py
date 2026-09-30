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
