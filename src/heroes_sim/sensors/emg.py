"""Synthetic surface EMG at the physics rate.

emg = volitional @ S^T + stim artifact + M-waves + white noise + powerline
      + baseline wander + motion artifact, then amplifier saturation.

Every component toggles in config for ablations. Pulse-locked waveforms (artifact,
M-wave) are overlap-added into a pending buffer, so they carry across chunk boundaries.
The stim artifact and M-waves are what the controller must not mistake for intent:
stim -> M-wave -> envelope up -> more stim is a positive-feedback path (spec 6.4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from scipy import signal

from heroes_sim.config import EMGConfig
from heroes_sim.stim import PulseEvent


class VolitionalSource(Protocol):
    """Per-muscle volitional EMG (mV at the muscle, before the sensor matrix).

    Swap in a motor-unit (Fuglevand) model later without changing callers.
    """

    def reset(self) -> None: ...

    def generate(self, u_vol: np.ndarray) -> np.ndarray:  # [n, M] -> [n, M]
        ...


def _unit_gain(sos: np.ndarray, fs_hz: float) -> float:
    """Scale that gives unit output variance for unit-variance white input."""
    impulse = np.zeros(int(10 * fs_hz))
    impulse[0] = 1.0
    return 1.0 / np.sqrt(np.sum(signal.sosfilt(sos, impulse) ** 2))


class BandlimitedNoiseSource:
    """Unit-variance bandlimited Gaussian carrier, amplitude-modulated by neural drive.

    Modulated by volitional excitation u_vol, not by MuJoCo muscle activation: sEMG tracks
    motor-unit firing (it leads force), and the plant's activation also contains the
    stim-evoked part, which appears in the EMG as M-waves, not as volitional EMG.
    """

    def __init__(self, cfg: EMGConfig, muscles: list[str], fs_hz: float, rng: np.random.Generator):
        v = cfg.volitional
        self._sos = signal.butter(v.order, v.band_hz, btype="bandpass", fs=fs_hz, output="sos")
        self._norm = _unit_gain(self._sos, fs_hz)
        self._amp = np.array([v.amplitude_mv.get(m, 0.0) for m in muscles])
        self._m = len(muscles)
        self._rng = rng
        self.reset()

    def reset(self) -> None:
        self._zi = np.zeros((self._sos.shape[0], 2, self._m))

    def generate(self, u_vol: np.ndarray) -> np.ndarray:
        white = self._rng.standard_normal((len(u_vol), self._m))
        carrier, self._zi = signal.sosfilt(self._sos, white, axis=0, zi=self._zi)
        return self._amp * u_vol * (self._norm * carrier)


class EMGSensor:
    def __init__(
        self,
        cfg: EMGConfig,
        muscles: list[str],
        joints: list[str],
        stim_channels: list[str],
        fs_hz: float,
        rng: Callable[[str], np.random.Generator],
        source: VolitionalSource | None = None,
    ):
        self.cfg = cfg
        self.names = [c.name for c in cfg.channels]
        self.fs = fs_hz
        j = len(self.names)
        self.S = np.array([[c.weights.get(m, 0.0) for m in muscles] for c in cfg.channels])
        self.source = source or BandlimitedNoiseSource(cfg, muscles, fs_hz, rng("volitional"))
        self._rng_noise = rng("white_noise")

        a = cfg.stim_artifact
        self.A = np.array(
            [[a.amplitude_mv.get(c, {}).get(k, 0.0) for k in stim_channels] for c in self.names]
        )  # [J, K]
        t_art = np.arange(int(np.ceil(10 * a.decay_ms * 1e-3 * fs_hz))) / fs_hz
        self._artifact_kernel = np.exp(-t_art / (a.decay_ms * 1e-3))

        mw = cfg.m_wave
        self._mwave_amp = np.array([mw.amplitude_mv.get(m, 0.0) for m in muscles])
        lat = round(mw.latency_ms * 1e-3 * fs_hz)
        dur = max(1, round(mw.duration_ms * 1e-3 * fs_hz))
        self._mwave_kernel = np.zeros(lat + dur)
        self._mwave_kernel[lat:] = np.sin(2 * np.pi * np.arange(dur) / dur)

        pl = cfg.powerline
        self._pl_phase = rng("powerline").uniform(0, 2 * np.pi, (len(pl.amplitude_mv), j))

        bw = cfg.baseline_wander
        self._bw_sos = signal.butter(2, bw.cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
        self._bw_norm = _unit_gain(self._bw_sos, fs_hz)
        self._rng_wander = rng("baseline_wander")

        self.M = np.array(
            [
                [cfg.motion.gain_mv_per_rad_s.get(c, {}).get(q, 0.0) for q in joints]
                for c in self.names
            ]
        )  # [J, N]
        self._pulse_len = max(len(self._artifact_kernel), len(self._mwave_kernel))
        self.reset()

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def reset(self) -> None:
        self.source.reset()
        self.artifact_gain = 1.0  # fault: artifact increase
        self.channel_fault: dict[int, str] = {}  # fault: channel -> "dropout" | "saturation"
        self._sample = 0
        self._amp_prev_y = np.zeros(self.n_channels)  # overload recovery state (H2)
        self._amp_gap = np.zeros(self.n_channels)  # decaying gap after leaving the rail
        self._pending = np.zeros((self._pulse_len, self.n_channels))
        self._bw_zi = np.zeros((self._bw_sos.shape[0], 2, self.n_channels))

    def add_pulse(self, offset: int, pulse: PulseEvent) -> None:
        """Schedule a pulse's artifact and M-wave, starting `offset` samples into the next
        chunk (0 = the next sample produced)."""
        need = offset + self._pulse_len
        if need > len(self._pending):
            grown = np.zeros((need, self.n_channels))
            grown[: len(self._pending)] = self._pending
            self._pending = grown
        if self.cfg.stim_artifact.enabled:
            amp = self.artifact_gain * (self.A @ pulse.intensity)  # [J]
            k = self._artifact_kernel
            self._pending[offset : offset + len(k)] += k[:, None] * amp[None, :]
        if self.cfg.m_wave.enabled:
            amp = self.S @ (self._mwave_amp * pulse.muscle_recruitment)  # [J]
            k = self._mwave_kernel
            self._pending[offset : offset + len(k)] += k[:, None] * amp[None, :]

    def _amplifier(self, x: np.ndarray) -> np.ndarray:
        """Clip at +-rail (either sign); after saturation, the output returns to the input
        with `overload_recovery_ms` (H2): y = x + g, where the gap g starts as the distance
        from the rail to the input and decays as exp(-t / tau). Only the gap is filtered,
        so once it has decayed the output is the input again, sample for sample."""
        rail = self.cfg.saturation_mv
        if rail is None:
            return x
        y = np.clip(x, -rail, rail)
        tau = self.cfg.amplifier.overload_recovery_ms * 1e-3
        if tau <= 0 or len(x) == 0:
            return y
        a = np.exp(-1.0 / (self.fs * tau))
        sat = np.abs(x) >= rail
        for j in range(x.shape[1]):
            if not (self._amp_gap[j] or sat[:, j].any()):
                continue
            g, prev_y = self._amp_gap[j], self._amp_prev_y[j]
            for i in range(len(x)):
                if sat[i, j]:
                    g = 0.0
                else:
                    if prev_y in (rail, -rail) and abs(x[i, j]) < rail and g == 0.0:
                        g = prev_y - x[i, j]  # just left the rail
                    g *= a
                    if abs(g) < 1e-6 * rail:
                        g = 0.0
                    y[i, j] = np.clip(x[i, j] + g, -rail, rail)
                prev_y = y[i, j]
            self._amp_gap[j] = g
        self._amp_prev_y = y[-1].copy()
        return y

    def chunk(self, u_vol: np.ndarray, qd: np.ndarray | None = None) -> np.ndarray:
        """Raw EMG for n consecutive samples given volitional excitation (and joint
        velocity, for the motion artifact) at each."""
        n, cfg = len(u_vol), self.cfg
        emg = np.zeros((n, self.n_channels))
        if cfg.volitional.enabled:
            emg += self.source.generate(u_vol) @ self.S.T
        if cfg.white_noise.enabled:
            emg += cfg.white_noise.std_mv * self._rng_noise.standard_normal(emg.shape)

        take = min(n, len(self._pending))
        emg[:take] += self._pending[:take]
        rest = self._pending[take:]
        self._pending = np.zeros((max(self._pulse_len, len(rest)), self.n_channels))
        self._pending[: len(rest)] = rest

        if cfg.powerline.enabled:
            t = (self._sample + np.arange(n)) / self.fs
            for h, amp in enumerate(cfg.powerline.amplitude_mv):
                w = 2 * np.pi * (h + 1) * cfg.powerline.freq_hz
                emg += amp * np.sin(w * t[:, None] + self._pl_phase[h][None, :])
        if cfg.baseline_wander.enabled:
            white = self._rng_wander.standard_normal(emg.shape)
            drift, self._bw_zi = signal.sosfilt(self._bw_sos, white, axis=0, zi=self._bw_zi)
            emg += cfg.baseline_wander.std_mv * self._bw_norm * drift
        if cfg.motion.enabled and qd is not None:
            emg += np.asarray(qd) @ self.M.T
        emg = self._amplifier(emg)
        for j, mode in self.channel_fault.items():
            if mode == "dropout":  # lead off: only the amplifier's own noise floor remains
                emg[:, j] = cfg.white_noise.std_mv * self._rng_noise.standard_normal(n)
            else:
                emg[:, j] = cfg.saturation_mv
        self._sample += n
        return emg
