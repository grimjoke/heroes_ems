"""PD on (reference - measured angle). No integral term, by design.

Clinical safety decision (spec 2.3): an I term would escalate stimulation when a patient
physically cannot reach a target. Do not add one, nor anti-windup or bias correction.
"""

from __future__ import annotations

import numpy as np


class PD:
    """u = kp * e + kd * de/dt, derivative by backward difference, optionally first-order
    lowpassed with time constant `derivative_tau_s` (0 = unfiltered). The first tick after
    reset has zero derivative."""

    def __init__(self, kp: np.ndarray, kd: np.ndarray, derivative_tau_s: float, dt: float):
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._dt = dt
        self._alpha = dt / (derivative_tau_s + dt)
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None
        self._d = np.zeros_like(self._kp)

    def __call__(self, error: np.ndarray) -> np.ndarray:
        raw = np.zeros_like(error) if self._prev is None else (error - self._prev) / self._dt
        self._prev = error.copy()
        self._d = self._d + self._alpha * (raw - self._d)
        return self._kp * error + self._kd * self._d
