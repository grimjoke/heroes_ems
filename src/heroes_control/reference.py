"""Gain + double integration: intent -> reference velocity -> reference angle (ROM-clamped)."""

from __future__ import annotations

import numpy as np


class DoubleIntegratorReference:
    """v' = gain * intent - damping * v;  q_ref' = v.  Semi-implicit Euler at the tick rate.

    damping = 0 is a pure double integrator. At a ROM limit the reference is clamped and
    velocity pointing further into the limit is zeroed, so it cannot wind up there.
    """

    def __init__(
        self,
        gain: np.ndarray,
        damping: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
        dt: float,
    ):
        self._gain = np.asarray(gain, dtype=np.float64)
        self._damping = np.asarray(damping, dtype=np.float64)
        self._q_min = np.asarray(q_min, dtype=np.float64)
        self._q_max = np.asarray(q_max, dtype=np.float64)
        self._dt = dt
        self.reset(np.clip(np.zeros_like(self._gain), self._q_min, self._q_max))

    def reset(self, q0: np.ndarray) -> None:
        self.velocity = np.zeros_like(self._gain)
        self.angle = np.clip(np.asarray(q0, dtype=np.float64), self._q_min, self._q_max)

    def __call__(self, intent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = self._dt
        v = self.velocity + dt * (self._gain * intent - self._damping * self.velocity)
        q = self.angle + dt * v
        low, high = q <= self._q_min, q >= self._q_max
        q = np.clip(q, self._q_min, self._q_max)
        v = np.where(low, np.maximum(v, 0.0), np.where(high, np.minimum(v, 0.0), v))
        self.velocity, self.angle = v, q
        return v.copy(), q.copy()
