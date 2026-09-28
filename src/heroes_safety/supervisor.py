"""Safety supervisor: the last stage before stimulation. Pure numpy; cannot be disabled.

Rules, in order, each tick:
  watchdog       controller command missing, non-finite, or older than timeout -> all zero
  sensor_sanity  angle or EMG non-finite / outside plausible range -> all zero
  rom_guard      channel driving toward a joint limit within `rom_margin_rad` -> zero
  cap            per-channel intensity cap
  rate_limit     max intensity *rise* per tick (decreases are always immediate)
  dose           rolling mean intensity over `dose_window_s` capped per channel

A rule "fires" on a channel when it changes that channel's output. Rising edges are
recorded as `SafetyEvent`s; the per-tick firing mask is in every `SafetyOutput`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RULES = ("watchdog", "sensor_sanity", "rom_guard", "cap", "rate_limit", "dose")


@dataclass(frozen=True)
class SafetyConfig:
    tick_hz: float
    channel_joint: tuple[int, ...]  # per stim channel
    channel_sign: tuple[int, ...]  # +1 drives q up, -1 down
    cap: tuple[float, ...]  # per stim channel, in [0, 1]
    max_rise_per_tick: float
    q_limit_min: tuple[float, ...]  # per joint: ROM guard limits
    q_limit_max: tuple[float, ...]
    rom_margin_rad: float
    dose_window_s: float
    dose_max_mean: float
    watchdog_timeout_s: float
    q_plausible_min: tuple[float, ...]  # per joint: sensor sanity
    q_plausible_max: tuple[float, ...]
    emg_plausible_abs: float

    def __post_init__(self) -> None:
        k, n = len(self.channel_joint), len(self.q_limit_min)
        if not (len(self.channel_sign) == len(self.cap) == k):
            raise ValueError("channel_joint, channel_sign, cap must have equal length")
        if not (
            len(self.q_limit_max) == len(self.q_plausible_min) == len(self.q_plausible_max) == n
        ):
            raise ValueError("per-joint limit tuples must have equal length")
        if any(not 0 <= j < n for j in self.channel_joint):
            raise ValueError("channel_joint index out of range")
        if any(s not in (1, -1) for s in self.channel_sign):
            raise ValueError("channel_sign entries must be +1 or -1")
        if any(not 0 <= c <= 1 for c in self.cap):
            raise ValueError("cap must be in [0, 1]")
        positive = ("tick_hz", "max_rise_per_tick", "dose_window_s", "watchdog_timeout_s")
        if any(getattr(self, f) <= 0 for f in positive):
            raise ValueError(f"{positive} must be > 0")
        if (
            not 0 < self.dose_max_mean <= 1
            or self.rom_margin_rad < 0
            or self.emg_plausible_abs <= 0
        ):
            raise ValueError("dose_max_mean in (0, 1], rom_margin_rad >= 0, emg_plausible_abs > 0")


@dataclass(frozen=True)
class SafetyEvent:
    t: float
    rule: str
    channel: int


@dataclass(frozen=True)
class SafetyOutput:
    intensity: np.ndarray  # [K], what goes to the stimulator
    fired: dict[str, np.ndarray]  # rule -> bool[K]


class SafetySupervisor:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self._k = len(cfg.channel_joint)
        self._joint = np.asarray(cfg.channel_joint, dtype=int)
        self._sign = np.asarray(cfg.channel_sign)
        self._cap = np.asarray(cfg.cap, dtype=np.float64)
        self._window = max(1, round(cfg.dose_window_s * cfg.tick_hz))
        self.reset()

    def reset(self) -> None:
        self._prev = np.zeros(self._k)
        self._dose_buf = np.zeros((self._window, self._k))
        self._dose_idx = 0
        self._was_firing = {r: np.zeros(self._k, dtype=bool) for r in RULES}
        self.events: list[SafetyEvent] = []

    def step(
        self,
        t: float,
        command: np.ndarray | None,
        command_t: float | None,
        q_meas: np.ndarray,
        emg_chunk: np.ndarray,
    ) -> SafetyOutput:
        cfg, k = self.cfg, self._k
        fired = {r: np.zeros(k, dtype=bool) for r in RULES}
        q = np.atleast_1d(np.asarray(q_meas, dtype=np.float64))
        emg = np.asarray(emg_chunk, dtype=np.float64)

        stale = (
            command is None
            or command_t is None
            or not np.all(np.isfinite(command))
            or t - command_t > cfg.watchdog_timeout_s
        )
        insane = (
            not np.all(np.isfinite(q))
            or np.any(q < cfg.q_plausible_min)
            or np.any(q > cfg.q_plausible_max)
            or not np.all(np.isfinite(emg))
            or (emg.size > 0 and np.max(np.abs(emg)) > cfg.emg_plausible_abs)
        )
        if stale or insane:
            s = np.zeros(k)
            fired["watchdog"][:] = stale
            fired["sensor_sanity"][:] = insane
        else:
            s = np.clip(np.asarray(command, dtype=np.float64), 0.0, 1.0)

            qj = q[self._joint]
            lim_hi = np.asarray(cfg.q_limit_max)[self._joint] - cfg.rom_margin_rad
            lim_lo = np.asarray(cfg.q_limit_min)[self._joint] + cfg.rom_margin_rad
            toward = ((self._sign > 0) & (qj >= lim_hi)) | ((self._sign < 0) & (qj <= lim_lo))
            s = self._apply(fired, "rom_guard", s, np.where(toward, 0.0, s))
            s = self._apply(fired, "cap", s, np.minimum(s, self._cap))
            s = self._apply(
                fired, "rate_limit", s, np.minimum(s, self._prev + cfg.max_rise_per_tick)
            )
            others = self._dose_buf.sum(axis=0) - self._dose_buf[self._dose_idx]
            budget = np.maximum(cfg.dose_max_mean * self._window - others, 0.0)
            s = self._apply(fired, "dose", s, np.minimum(s, budget))

        self._prev = s
        self._dose_buf[self._dose_idx] = s
        self._dose_idx = (self._dose_idx + 1) % self._window
        for rule in RULES:
            for ch in np.nonzero(fired[rule] & ~self._was_firing[rule])[0]:
                self.events.append(SafetyEvent(t, rule, int(ch)))
            self._was_firing[rule] = fired[rule]
        return SafetyOutput(intensity=s.copy(), fired=fired)

    @staticmethod
    def _apply(fired: dict, rule: str, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        fired[rule] |= after < before
        return after
