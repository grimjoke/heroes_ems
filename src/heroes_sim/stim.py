"""EMS stimulation model: channel intensity s[K] -> stimulated excitation u_stim[M].

Per pulse: quantize -> recruitment sigmoid -> electrode-to-muscle matrix E[M, K].
v1 holds the recruitment as continuous excitation between pulses; `PulseEvent`s keep the
event interface for twitch modeling and for the EMG artifact/M-wave model (M3). The
result passes through a fixed electromechanical delay line. Fatigue arrives in M4.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from heroes_sim.config import RecruitmentConfig, StimConfig


def probabilistic_sum(u_vol: np.ndarray, u_stim: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - (1.0 - u_vol) * (1.0 - u_stim), 0.0, 1.0)


COMBINE: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "probabilistic_sum": probabilistic_sum,
}


def quantize(s: np.ndarray, n_steps: int) -> np.ndarray:
    """Snap to `n_steps` evenly spaced levels in [0, 1] (0 and 1 included)."""
    levels = n_steps - 1
    return np.round(np.clip(s, 0.0, 1.0) * levels) / levels


def recruitment(s: np.ndarray, cfg: RecruitmentConfig) -> np.ndarray:
    """Logistic in intensity, shifted and scaled so r(0) = 0 and r(1) = saturation."""

    def sig(x):
        return 1.0 / (1.0 + np.exp(-cfg.slope * (x - cfg.threshold)))

    lo, hi = sig(0.0), sig(1.0)
    return cfg.saturation * (sig(s) - lo) / (hi - lo)


@dataclass(frozen=True)
class PulseEvent:
    t: float
    intensity: np.ndarray  # [K] quantized
    recruitment: np.ndarray  # [K]


class StimModel:
    def __init__(self, cfg: StimConfig, muscles: list[str], physics_dt: float):
        self.cfg = cfg
        self.E = np.array([[ch.muscles.get(m, 0.0) for ch in cfg.channels] for m in muscles])
        self.combine = COMBINE[cfg.combine]
        self._delay = round(cfg.em_delay_ms * 1e-3 / physics_dt)
        self._m, self._k = len(muscles), len(cfg.channels)
        self.reset()

    def reset(self) -> None:
        self._held = np.zeros(self._k)
        self.capacity = np.ones(self._m)  # fatigue state (M4); 1 = unfatigued
        zero = np.zeros(self._m)
        self._line: deque[np.ndarray] = deque([zero] * (self._delay + 1), maxlen=self._delay + 1)

    def pulse(self, t: float, s: np.ndarray) -> PulseEvent:
        """A stimulator pulse at time t with commanded intensity s[K]."""
        q = quantize(np.asarray(s, dtype=np.float64), self.cfg.n_steps)
        self._held = recruitment(q, self.cfg.recruitment)
        return PulseEvent(t, q, self._held.copy())

    def step(self) -> np.ndarray:
        """Advance one physics step; returns delayed u_stim[M]."""
        self._line.append(np.clip(self.E @ self._held, 0.0, 1.0) * self.capacity)
        return self._line[0]
