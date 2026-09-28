"""Scenario-side signals: target trajectories and the MVC trial schedule.

Faults and perturbations (spec 6.8) arrive in M4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heroes_sim.config import (
    CalibrationConfig,
    HoldSegment,
    RampSegment,
    Segment,
    SineSegment,
    StepSegment,
)


class TargetTrajectory:
    """Piecewise target joint angles; holds the last value after the final segment."""

    def __init__(self, segments: list[Segment], q0: np.ndarray):
        self._segments = segments
        self._starts: list[float] = []
        self._q_start: list[np.ndarray] = []
        t, q = 0.0, np.asarray(q0, dtype=np.float64)
        for seg in segments:
            self._starts.append(t)
            self._q_start.append(q)
            q = self._eval(seg, q, seg.duration_s)
            t += seg.duration_s
        self._t_end, self._q_end = t, q

    @staticmethod
    def _eval(seg: Segment, q_prev: np.ndarray, tau: float) -> np.ndarray:
        if isinstance(seg, HoldSegment):
            return q_prev
        if isinstance(seg, StepSegment):
            return np.asarray(seg.q, dtype=np.float64)
        if isinstance(seg, RampSegment):
            frac = tau / seg.duration_s
            return q_prev + frac * (np.asarray(seg.q, dtype=np.float64) - q_prev)
        if isinstance(seg, SineSegment):
            phase = 2.0 * np.pi * seg.freq_hz * tau
            return np.asarray(seg.center) + np.asarray(seg.amplitude) * np.sin(phase)
        raise TypeError(f"unknown segment {seg!r}")

    def __call__(self, t: float) -> np.ndarray:
        if t >= self._t_end:
            return self._q_end
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        return self._eval(self._segments[i], self._q_start[i], t - self._starts[i])


@dataclass(frozen=True)
class MVCWindow:
    group: str
    t_start: float
    t_end: float


class MVCSchedule:
    """Rest, then max effort, for each trial in order."""

    def __init__(self, sc: CalibrationConfig):
        windows, t = [], 0.0
        for tr in sc.trials:
            t += sc.rest_s
            windows.append(MVCWindow(tr.group, t, t + tr.effort_s))
            t += tr.effort_s
        self.windows: list[MVCWindow] = windows
        self.duration_s = t + sc.rest_s

    def group_at(self, t: float) -> str | None:
        for w in self.windows:
            if w.t_start <= t < w.t_end:
                return w.group
        return None
