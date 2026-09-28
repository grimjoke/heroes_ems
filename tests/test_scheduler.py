import pytest

from heroes_sim.config import TimingConfig
from heroes_sim.scheduler import Scheduler, validate_rates


def timing(**kw):
    base = {
        "physics_hz": 2000,
        "emg_hz": 2000,
        "controller_hz": 100,
        "stim_hz": 25,
        "angle_sensor_hz": 100,
    }
    return TimingConfig(**{**base, **kw})


def test_default_rates_valid():
    r = validate_rates(timing())
    assert r == {"emg_hz": 1, "controller_hz": 20, "angle_sensor_hz": 20}


@pytest.mark.parametrize(
    "field,hz",
    [("controller_hz", 300), ("emg_hz", 4000), ("angle_sensor_hz", 30), ("stim_hz", 4000)],
)
def test_bad_rates_raise(field, hz):
    """Sampled clocks must divide the physics rate; events may not exceed it."""
    with pytest.raises(ValueError, match=field):
        validate_rates(timing(**{field: hz}))


@pytest.mark.parametrize("hz", [30, 33, 25, 17])
def test_stim_event_clock_any_rate(hz):
    """Stim pulses are events: any rate, fired at the nearest physics step (D8)."""
    s = Scheduler(timing(stim_hz=hz))
    steps = [n for n in range(1, 20001) if s.advance().stim]  # 10 s
    assert len(steps) == 10 * hz
    ideal = [k * 2000 / hz for k in range(1, len(steps) + 1)]
    assert max(abs(a - b) for a, b in zip(steps, ideal)) <= 0.5  # <= half a step (0.25 ms)


def test_tick_counts():
    s = Scheduler(timing())
    ticks = [s.advance() for _ in range(2000)]
    assert sum(t.controller for t in ticks) == 100
    assert sum(t.stim for t in ticks) == 25
    assert sum(t.emg for t in ticks) == 2000
