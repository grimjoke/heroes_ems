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
    assert r == {"emg_hz": 1, "controller_hz": 20, "stim_hz": 80, "angle_sensor_hz": 20}


@pytest.mark.parametrize("field,hz", [("stim_hz", 30), ("controller_hz", 300), ("emg_hz", 4000)])
def test_non_dividing_rates_raise(field, hz):
    with pytest.raises(ValueError, match=field):
        validate_rates(timing(**{field: hz}))


def test_tick_counts():
    s = Scheduler(timing())
    ticks = [s.advance() for _ in range(2000)]
    assert sum(t.controller for t in ticks) == 100
    assert sum(t.stim for t in ticks) == 25
    assert sum(t.emg for t in ticks) == 2000
