"""Intent -> reference velocity -> reference angle (ROM-clamped).

    v' = G * intent - b * v,   q_ref' = v

With damping b > 0 this is a velocity mapping: a steady intent u settles at a reference
velocity of (G / b) * u (rad/s per unit intent), smoothed with time constant 1 / b. b = 0 is
a pure double integrator (intent sets acceleration; velocity persists after intent stops).
Damping removes momentum and overshoot; it does not stop drift from a constant leak, which
still ramps the angle at constant speed (the deadband has to handle that).
"""

from __future__ import annotations

import numpy as np


class DoubleIntegratorReference:
    """Exact zero-order-hold discretisation at the tick rate (intent held over each tick):

        a = exp(-b dt)
        v' = a v + (G / b)(1 - a) u
        q' = q + v (1 - a) / b + (G / b) u (dt - (1 - a) / b)

    with the b -> 0 limits (v' = v + G dt u, q' = q + v dt + G dt^2 u / 2). Exact for any b
    and dt, unlike forward Euler, which overshoots for b * dt near 1. At a ROM limit the
    reference is clamped and velocity pointing further into the limit is zeroed, so it cannot
    wind up there.
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
        b = np.asarray(damping, dtype=np.float64)
        self._q_min = np.asarray(q_min, dtype=np.float64)
        self._q_max = np.asarray(q_max, dtype=np.float64)
        self._dt = dt
        a = np.exp(-b * dt)
        safe_b = np.where(b > 0, b, 1.0)
        self._a = a
        # (1 - a) / b and (dt - (1 - a) / b) / b, with their b -> 0 limits dt and dt^2 / 2.
        self._k1 = np.where(b > 0, (1.0 - a) / safe_b, dt)
        self._k2 = np.where(b > 0, (dt - (1.0 - a) / safe_b) / safe_b, 0.5 * dt * dt)
        self.reset(np.clip(np.zeros_like(self._gain), self._q_min, self._q_max))

    def reset(self, q0: np.ndarray) -> None:
        self.velocity = np.zeros_like(self._gain)
        self.angle = np.clip(np.asarray(q0, dtype=np.float64), self._q_min, self._q_max)

    def __call__(self, intent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        acc = self._gain * intent  # G * u
        v0 = self.velocity
        v = self._a * v0 + self._k1 * acc
        q = self.angle + self._k1 * v0 + self._k2 * acc
        low, high = q <= self._q_min, q >= self._q_max
        q = np.clip(q, self._q_min, self._q_max)
        v = np.where(low, np.maximum(v, 0.0), np.where(high, np.minimum(v, 0.0), v))
        self.velocity, self.angle = v, q
        return v.copy(), q.copy()
