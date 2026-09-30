"""IMU-like joint angle sensor: latency, white noise, bias random walk, quantization."""

from __future__ import annotations

from collections import deque

import numpy as np

from heroes_sim.config import AngleSensorConfig


class AngleSensor:
    """`push` the true angle every physics step; `sample` at the sensor rate."""

    def __init__(
        self,
        cfg: AngleSensorConfig,
        n_joints: int,
        physics_dt: float,
        sample_dt: float,
        rng: np.random.Generator,
    ):
        self.cfg = cfg
        self._n = n_joints
        self._delay = round(cfg.latency_ms * 1e-3 / physics_dt)
        self._walk = cfg.bias_walk_std_rad_per_sqrt_s * np.sqrt(sample_dt)
        self._rng = rng
        self.reset(np.zeros(n_joints))

    def reset(self, q0: np.ndarray) -> None:
        q0 = np.asarray(q0, dtype=np.float64).copy()
        self._line: deque[np.ndarray] = deque([q0] * (self._delay + 1), maxlen=self._delay + 1)
        self._bias = np.zeros(self._n)
        self.frozen = np.zeros(self._n, dtype=bool)  # fault: repeat the last reading
        self._out = q0.copy()

    def push(self, q_true: np.ndarray) -> None:
        self._line.append(np.asarray(q_true, dtype=np.float64).copy())

    def sample(self) -> np.ndarray:
        self._bias = self._bias + self._walk * self._rng.standard_normal(self._n)
        noise = self.cfg.noise_std_rad * self._rng.standard_normal(self._n)
        q = self._line[0] + self._bias + noise
        res = self.cfg.resolution_rad
        q = np.round(q / res) * res if res > 0 else q
        self._out = np.where(self.frozen, self._out, q)
        return self._out.copy()
