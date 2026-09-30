"""Safety supervisor: the last stage before stimulation. Pure numpy; cannot be disabled.

Rules, in order, each tick. The first group zeroes every channel (the loop can't be
trusted); the rest act per channel.

  watchdog       controller command missing, non-finite, or older than timeout
  sensor_sanity  angle or EMG non-finite / outside plausible range
  emg_rail       an EMG channel has >= `emg_rail_min_samples` samples in the chunk at
                 either amplifier rail (|x| >= rail_fraction * rail): stuck/saturated lead
  emg_dead       an EMG channel's chunk std stays below `emg_dead_ratio` x its calibrated
                 resting std for `emg_dead_ticks` ticks: detached recording electrode. With
                 agonist/antagonist differencing, a dead antagonist turns agonist noise into
                 intent, so this matters as much as saturation
  angle_stale    angle message older than `angle_max_age_s` (driver stuck; primary check)
  angle_frozen   `angle_repeat_ticks` consecutive identical angle readings (fresh messages
                 carrying a frozen value; fallback check)
  impedance      stim channel impedance above `impedance_max_ohm` -> that channel zero.
                 Partial electrode contact raises impedance and current density: this is a
                 skin-safety check, not just wasted stimulation
  rom_guard      channel driving toward a joint limit within `rom_margin_rad` -> zero
  cap            per-channel intensity cap
  rate_limit     max intensity *rise* per tick (decreases are always immediate)
  dose           rolling mean intensity over `dose_window_s` capped per channel

A detector whose parameter is None (or 0 for counts/ratios) is off; the sim config sets all
of them. A rule "fires" on a channel when it changes that channel's output. Rising edges are
recorded as `SafetyEvent`s; the per-tick firing mask is in every `SafetyOutput`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ZERO_ALL = (
    "watchdog",
    "sensor_sanity",
    "emg_rail",
    "emg_dead",
    "angle_stale",
    "angle_frozen",
)
RULES = (*ZERO_ALL, "impedance", "rom_guard", "cap", "rate_limit", "dose")


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
    emg_rail: float | None = None  # amplifier rail (same units as EMG)
    emg_rail_fraction: float = 0.98
    emg_rail_min_samples: int = 0  # per chunk; 0 = off
    emg_dead_ratio: float = 0.0  # 0 = off; needs emg_rest_std at construction
    emg_dead_ticks: int = 1
    angle_max_age_s: float | None = None
    angle_repeat_ticks: int = 0  # 0 = off
    impedance_max_ohm: float | None = None

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
        if self.emg_rail_min_samples > 0 and (self.emg_rail is None or self.emg_rail <= 0):
            raise ValueError("emg_rail_min_samples > 0 needs emg_rail > 0")
        if not 0 < self.emg_rail_fraction <= 1:
            raise ValueError("emg_rail_fraction must be in (0, 1]")
        if not 0 <= self.emg_dead_ratio < 1 or self.emg_dead_ticks < 1:
            raise ValueError("emg_dead_ratio in [0, 1), emg_dead_ticks >= 1")
        if self.angle_repeat_ticks < 0 or self.emg_rail_min_samples < 0:
            raise ValueError("counts must be >= 0")


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
    def __init__(self, cfg: SafetyConfig, emg_rest_std: tuple[float, ...] | None = None):
        """`emg_rest_std`: calibrated raw-EMG chunk std at rest, per channel. Required when
        the dead-channel detector is on: a supervisor can't judge "too quiet" without it."""
        rest_ok = emg_rest_std is not None and all(np.isfinite(v) and v > 0 for v in emg_rest_std)
        if cfg.emg_dead_ratio > 0 and not rest_ok:
            raise ValueError("emg_dead_ratio > 0 needs a calibrated emg_rest_std > 0")
        self.cfg = cfg
        self._k = len(cfg.channel_joint)
        self._joint = np.asarray(cfg.channel_joint, dtype=int)
        self._sign = np.asarray(cfg.channel_sign)
        self._cap = np.asarray(cfg.cap, dtype=np.float64)
        self._window = max(1, round(cfg.dose_window_s * cfg.tick_hz))
        self._rest_std = None if emg_rest_std is None else np.asarray(emg_rest_std, dtype=float)
        self.reset()

    def reset(self) -> None:
        self._prev = np.zeros(self._k)
        self._dose_buf = np.zeros((self._window, self._k))
        self._dose_idx = 0
        self._was_firing = {r: np.zeros(self._k, dtype=bool) for r in RULES}
        self._dead_run: np.ndarray | None = None  # per EMG channel: consecutive quiet ticks
        self._last_q: np.ndarray | None = None
        self._repeat_run: np.ndarray | None = None  # per joint: consecutive identical readings
        self.events: list[SafetyEvent] = []

    def step(
        self,
        t: float,
        command: np.ndarray | None,
        command_t: float | None,
        q_meas: np.ndarray,
        emg_chunk: np.ndarray,
        *,
        q_stamp: float | None = None,
        impedance: np.ndarray | None = None,
    ) -> SafetyOutput:
        """`q_stamp`: acquisition time of `q_meas` (required when angle_max_age_s is set;
        missing counts as stale). `impedance[K]`: latest per-channel stimulator reading
        (required when impedance_max_ohm is set; missing counts as over the limit)."""
        cfg, k = self.cfg, self._k
        fired = {r: np.zeros(k, dtype=bool) for r in RULES}
        q = np.atleast_1d(np.asarray(q_meas, dtype=np.float64))
        emg = np.asarray(emg_chunk, dtype=np.float64)
        emg2 = emg.reshape(len(emg), -1) if emg.size else emg.reshape(0, 0)

        zero_all = {
            "watchdog": (
                command is None
                or command_t is None
                or not np.all(np.isfinite(command))
                or t - command_t > cfg.watchdog_timeout_s
            ),
            "sensor_sanity": (
                not np.all(np.isfinite(q))
                or np.any(q < cfg.q_plausible_min)
                or np.any(q > cfg.q_plausible_max)
                or not np.all(np.isfinite(emg))
                or (emg.size > 0 and np.max(np.abs(emg)) > cfg.emg_plausible_abs)
            ),
            "emg_rail": self._rail(emg2),
            "emg_dead": self._dead(emg2),
            "angle_stale": cfg.angle_max_age_s is not None
            and (q_stamp is None or t - q_stamp > cfg.angle_max_age_s),
            "angle_frozen": self._frozen(q),
        }
        if any(zero_all.values()):
            s = np.zeros(k)
            for rule, hit in zero_all.items():
                fired[rule][:] = bool(hit)
        else:
            s = np.clip(np.asarray(command, dtype=np.float64), 0.0, 1.0)
            if cfg.impedance_max_ohm is not None:
                z = np.full(k, np.inf) if impedance is None else np.asarray(impedance, float)
                bad = ~(z <= cfg.impedance_max_ohm)  # NaN counts as bad
                s = self._apply(fired, "impedance", s, np.where(bad, 0.0, s))

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

    def _rail(self, emg: np.ndarray) -> bool:
        cfg = self.cfg
        if cfg.emg_rail_min_samples <= 0 or emg.size == 0:
            return False
        at_rail = np.abs(emg) >= cfg.emg_rail_fraction * cfg.emg_rail  # both rails
        return bool(np.any(at_rail.sum(axis=0) >= cfg.emg_rail_min_samples))

    def _dead(self, emg: np.ndarray) -> bool:
        cfg = self.cfg
        if cfg.emg_dead_ratio <= 0 or len(emg) < 2:
            return False
        quiet = emg.std(axis=0) < cfg.emg_dead_ratio * self._rest_std
        if self._dead_run is None:
            self._dead_run = np.zeros(len(quiet), dtype=int)
        self._dead_run = np.where(quiet, self._dead_run + 1, 0)
        return bool(np.any(self._dead_run >= cfg.emg_dead_ticks))

    def _frozen(self, q: np.ndarray) -> bool:
        cfg = self.cfg
        if cfg.angle_repeat_ticks <= 0:
            return False
        if self._last_q is None:
            self._last_q, self._repeat_run = q.copy(), np.zeros(len(q), dtype=int)
            return False
        self._repeat_run = np.where(q == self._last_q, self._repeat_run + 1, 0)
        self._last_q = q.copy()
        return bool(np.any(self._repeat_run >= cfg.angle_repeat_ticks))

    @staticmethod
    def _apply(fired: dict, rule: str, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        fired[rule] |= after < before
        return after
