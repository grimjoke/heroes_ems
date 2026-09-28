"""The controller: raw EMG + measured angle -> per-channel stim intensity.

Pure numpy/scipy so the future ROS node wraps this exact code. Every stage is an object
passed in by composition; defaults are built from `ControllerConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heroes_control.allocation import SignedAllocation
from heroes_control.filters import EMGFrontEnd, Rectify, bandpass, lowpass
from heroes_control.intent import AgonistAntagonistIntent
from heroes_control.normalization import MVCNormalizer, MVCValues
from heroes_control.pd import PD
from heroes_control.reference import DoubleIntegratorReference


@dataclass(frozen=True)
class ControllerConfig:
    """Index-based (no names): EMG channel j, joint i, stim channel k. Per-joint tuples
    have one entry per joint; per-stim-channel tuples one per stim channel."""

    emg_fs_hz: float
    tick_hz: float
    n_emg: int
    bandpass_hz: tuple[float, float]
    bandpass_order: int
    envelope_hz: float
    envelope_order: int
    agonist: tuple[int, ...]  # EMG channel per joint
    antagonist: tuple[int, ...]
    deadband: float
    gain: tuple[float, ...]  # rad/s^2 per unit intent
    damping: tuple[float, ...]  # 1/s
    q_min: tuple[float, ...]
    q_max: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    derivative_tau_s: float
    channel_joint: tuple[int, ...]
    channel_sign: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.agonist)
        per_joint = ("antagonist", "gain", "damping", "q_min", "q_max", "kp", "kd")
        bad = [f for f in per_joint if len(getattr(self, f)) != n]
        if bad:
            raise ValueError(f"per-joint fields must have length {n}: {bad}")
        if len(self.channel_sign) != len(self.channel_joint):
            raise ValueError("channel_sign and channel_joint lengths differ")
        if any(s not in (1, -1) for s in self.channel_sign):
            raise ValueError("channel_sign entries must be +1 or -1")
        if any(not 0 <= j < n for j in self.channel_joint):
            raise ValueError("channel_joint index out of range")
        if any(not 0 <= c < self.n_emg for c in self.agonist + self.antagonist):
            raise ValueError("agonist/antagonist EMG index out of range")
        lo, hi = self.bandpass_hz
        if not 0 < lo < hi < self.emg_fs_hz / 2:
            raise ValueError(f"bandpass {self.bandpass_hz} must satisfy 0 < lo < hi < fs/2")
        if not 0 < self.envelope_hz < self.emg_fs_hz / 2:
            raise ValueError("envelope_hz must be in (0, fs/2)")
        if not 0 <= self.deadband < 1:
            raise ValueError("deadband must be in [0, 1)")
        if any(a >= b for a, b in zip(self.q_min, self.q_max, strict=True)):
            raise ValueError("q_min must be < q_max")
        nonneg = ("gain", "damping", "kp", "kd")
        if any(v < 0 for f in nonneg for v in getattr(self, f)) or self.derivative_tau_s < 0:
            raise ValueError(f"{nonneg} and derivative_tau_s must be >= 0")

    @property
    def dt(self) -> float:
        return 1.0 / self.tick_hz

    @property
    def n_joints(self) -> int:
        return len(self.agonist)


def default_front_end(cfg: ControllerConfig) -> EMGFrontEnd:
    lo, hi = cfg.bandpass_hz
    return EMGFrontEnd(
        [
            bandpass(lo, hi, cfg.bandpass_order, cfg.emg_fs_hz, cfg.n_emg),
            Rectify(),
            lowpass(cfg.envelope_hz, cfg.envelope_order, cfg.emg_fs_hz, cfg.n_emg),
        ]
    )


@dataclass(frozen=True)
class ControllerOutput:
    t: float
    envelope: np.ndarray  # [J]
    normalized: np.ndarray  # [J]
    intent: np.ndarray  # [N]
    ref_velocity: np.ndarray  # [N]
    ref_angle: np.ndarray  # [N]
    error: np.ndarray  # [N]
    pd_output: np.ndarray  # [N]
    intensity: np.ndarray  # [K] in [0, 1], before the safety supervisor


class ControllerPipeline:
    def __init__(
        self,
        cfg: ControllerConfig,
        mvc: MVCValues,
        *,
        front_end: EMGFrontEnd | None = None,
        normalizer=None,
        intent=None,
        reference=None,
        pd=None,
        allocator=None,
    ):
        if len(mvc.envelope) != cfg.n_emg:
            raise ValueError(f"MVC has {len(mvc.envelope)} channels, config has {cfg.n_emg}")
        self.cfg = cfg
        self.front_end = front_end or default_front_end(cfg)
        self.normalizer = normalizer or MVCNormalizer(mvc)
        self.intent = intent or AgonistAntagonistIntent(cfg.agonist, cfg.antagonist, cfg.deadband)
        self.reference = reference or DoubleIntegratorReference(
            np.array(cfg.gain),
            np.array(cfg.damping),
            np.array(cfg.q_min),
            np.array(cfg.q_max),
            cfg.dt,
        )
        self.pd = pd or PD(np.array(cfg.kp), np.array(cfg.kd), cfg.derivative_tau_s, cfg.dt)
        self.allocator = allocator or SignedAllocation(cfg.channel_joint, cfg.channel_sign)
        self.reset(np.zeros(cfg.n_joints))

    def reset(self, q0: float | np.ndarray) -> None:
        for stage in (self.front_end, self.normalizer, self.intent, self.pd, self.allocator):
            stage.reset()
        self.reference.reset(np.atleast_1d(np.asarray(q0, dtype=np.float64)))
        self._envelope = np.zeros(self.cfg.n_emg)

    def step(self, t: float, emg_chunk: np.ndarray, q_meas: float | np.ndarray) -> ControllerOutput:
        """One controller tick. `emg_chunk[n, J]` is all raw EMG since the previous tick."""
        chunk = np.asarray(emg_chunk, dtype=np.float64).reshape(-1, self.cfg.n_emg)
        env = self.front_end.process(t, chunk)
        if len(env):
            self._envelope = env[-1].copy()
        normalized = self.normalizer(self._envelope)
        intent = self.intent(normalized)
        ref_v, ref_q = self.reference(intent)
        error = ref_q - np.atleast_1d(np.asarray(q_meas, dtype=np.float64))
        u = self.pd(error)
        return ControllerOutput(
            t=t,
            envelope=self._envelope.copy(),
            normalized=normalized,
            intent=intent,
            ref_velocity=ref_v,
            ref_angle=ref_q,
            error=error,
            pd_output=u,
            intensity=self.allocator(u),
        )
