"""EMS stimulation model: channel intensity s[K] -> stimulated excitation u_stim[M].

Per pulse: quantize -> electrode gain (1, or 0 when detached) -> recruitment sigmoid ->
electrode-to-muscle matrix E[M, K]. v1 holds the recruitment as continuous excitation
between pulses; `PulseEvent`s keep the event interface for twitch modeling and for the EMG
artifact/M-wave model. Every physics step, fatigue updates per-muscle capacity from the
held recruitment, and u_stim = recruitment * capacity passes through a fixed
electromechanical delay line.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from heroes_sim.config import FatigueConfig, RecruitmentConfig, StimConfig


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


class NoFatigue:
    def __init__(self, cfg: FatigueConfig, n_muscles: int):
        self._c = np.ones(n_muscles)

    def reset(self) -> None:
        self._c[:] = 1.0

    def step(self, dt: float, recruitment: np.ndarray) -> np.ndarray:
        return self._c


class ExponentialFatigue:
    """dC/dt = -f * r * C + k * (1 - C): recruited fibres fatigue, rest recovers.

    Semi-implicit update, so C stays in [0, 1] for any dt."""

    def __init__(self, cfg: FatigueConfig, n_muscles: int):
        self._f, self._k = cfg.fatigue_rate_per_s, cfg.recovery_rate_per_s
        self._c = np.ones(n_muscles)

    def reset(self) -> None:
        self._c = np.ones_like(self._c)

    def step(self, dt: float, recruitment: np.ndarray) -> np.ndarray:
        self._c = (self._c + dt * self._k) / (1.0 + dt * (self._f * recruitment + self._k))
        return self._c


FATIGUE = {"none": NoFatigue, "exponential": ExponentialFatigue}


@dataclass(frozen=True)
class PulseEvent:
    t: float
    intensity: np.ndarray  # [K] quantized and delivered (0 on a detached electrode)
    recruitment: np.ndarray  # [K]
    muscle_recruitment: np.ndarray  # [M] effective (x capacity); drives M-waves

    @property
    def active(self) -> bool:
        """A pulse is delivered only if some channel has nonzero intensity."""
        return bool(np.any(self.intensity > 0))


class StimModel:
    def __init__(self, cfg: StimConfig, muscles: list[str], physics_dt: float):
        self.cfg = cfg
        self.E = np.array([[ch.muscles.get(m, 0.0) for ch in cfg.channels] for m in muscles])
        self.combine = COMBINE[cfg.combine]
        self._dt = physics_dt
        self._delay = round(cfg.em_delay_ms * 1e-3 / physics_dt)
        self._m, self._k = len(muscles), len(cfg.channels)
        self.fatigue = FATIGUE[cfg.fatigue.model](cfg.fatigue, self._m)
        self.reset()

    def reset(self) -> None:
        self._held = np.zeros(self._k)
        self._raw = np.zeros(self._m)  # recruited fraction per muscle, before capacity
        self.fatigue.reset()
        self.capacity = np.ones(self._m)
        self.contact = np.ones(self._k)  # electrode contact fraction; 0 = off (fault)
        zero = np.zeros(self._m)
        self._line: deque[np.ndarray] = deque([zero] * (self._delay + 1), maxlen=self._delay + 1)

    def pulse(self, t: float, s: np.ndarray) -> PulseEvent:
        """A stimulator pulse at time t with commanded intensity s[K]."""
        q = quantize(np.asarray(s, dtype=np.float64), self.cfg.n_steps) * (self.contact > 0)
        self._held = recruitment(q, self.cfg.recruitment)
        self._raw = np.clip(self.E @ self._held, 0.0, 1.0)
        return PulseEvent(t, q, self._held.copy(), self._raw * self.capacity)

    def impedance(self) -> np.ndarray:
        """Per-channel electrode impedance as the stimulator reports it: nominal / contact
        (inf when fully off)."""
        with np.errstate(divide="ignore"):
            return self.cfg.impedance_ohm / self.contact

    def current_density(self, intensity: np.ndarray) -> np.ndarray:
        """Delivered intensity / contact area, relative to full contact (1 = nominal)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.contact > 0, intensity / self.contact, 0.0)

    def step(self) -> np.ndarray:
        """Advance one physics step: fatigue update, then delayed u_stim[M]."""
        self.capacity = self.fatigue.step(self._dt, self._raw)
        self._line.append(self._raw * self.capacity)
        return self._line[0]
