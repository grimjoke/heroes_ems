import numpy as np
import pytest

from heroes_sim.config import MovementScenario, MVCScenario
from heroes_sim.scenario import MVCSchedule, TargetTrajectory


def traj(segments, q0=0.5):
    sc = MovementScenario.model_validate(
        {
            "kind": "movement",
            "name": "t",
            "duration_s": 1.0,
            "initial_state": {"q": [q0], "qd": [0.0]},
            "target": segments,
        }
    )
    return TargetTrajectory(sc.target, np.array([q0]))


def test_segments():
    tr = traj(
        [
            {"kind": "hold", "duration_s": 1.0},
            {"kind": "step", "duration_s": 1.0, "q": [1.0]},
            {"kind": "ramp", "duration_s": 2.0, "q": [2.0]},
            {"kind": "sine", "duration_s": 1.0, "center": [1.0], "amplitude": [0.5], "freq_hz": 1},
        ]
    )
    assert tr(0.5)[0] == 0.5
    assert tr(1.0)[0] == 1.0 and tr(1.99)[0] == 1.0
    assert tr(3.0)[0] == pytest.approx(1.5)
    assert tr(4.25)[0] == pytest.approx(1.5)
    assert tr(99.0)[0] == pytest.approx(1.0)  # holds the last value (sine end)


def test_mvc_schedule():
    sc = MVCScenario.model_validate(
        {
            "kind": "mvc",
            "name": "m",
            "lock_q": [1.0],
            "rest_s": 1.0,
            "window_s": 0.5,
            "trials": [{"group": "a", "effort_s": 2.0}, {"group": "b", "effort_s": 1.0}],
        }
    )
    s = MVCSchedule(sc)
    assert [(w.group, w.t_start, w.t_end) for w in s.windows] == [("a", 1, 3), ("b", 4, 5)]
    assert s.duration_s == 6.0
    assert [s.group_at(t) for t in (0.5, 1.0, 3.5, 4.5, 5.5)] == [None, "a", None, "b", None]


def test_window_longer_than_effort_rejected():
    with pytest.raises(ValueError, match="window_s"):
        MVCScenario.model_validate(
            {
                "kind": "mvc",
                "name": "m",
                "lock_q": [1.0],
                "rest_s": 0.0,
                "window_s": 2.0,
                "trials": [{"group": "a", "effort_s": 1.0}],
            }
        )
