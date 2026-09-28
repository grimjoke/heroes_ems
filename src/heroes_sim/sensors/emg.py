"""Synthetic surface EMG at the physics rate. M2: clean model (volitional + white noise).

Stim artifact, M-waves, powerline, baseline wander and motion artifact arrive in M3.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import signal

from heroes_sim.config import EMGConfig


class VolitionalSource(Protocol):
    """Per-muscle volitional EMG (mV at the muscle, before the sensor matrix).

    Swap in a motor-unit (Fuglevand) model later without changing callers.
    """

    def reset(self) -> None: ...

    def generate(self, u_vol: np.ndarray) -> np.ndarray:  # [n, M] -> [n, M]
        ...


class BandlimitedNoiseSource:
    """Unit-variance bandlimited Gaussian carrier, amplitude-modulated by neural drive.

    Modulated by volitional excitation u_vol, not by MuJoCo muscle activation: sEMG tracks
    motor-unit firing (it leads force), and the plant's activation also contains the
    stim-evoked part, which appears in the EMG as M-waves (M3), not as volitional EMG.
    """

    def __init__(self, cfg: EMGConfig, muscles: list[str], fs_hz: float, rng: np.random.Generator):
        v = cfg.volitional
        self._sos = signal.butter(v.order, v.band_hz, btype="bandpass", fs=fs_hz, output="sos")
        impulse = np.zeros(int(2 * fs_hz))
        impulse[0] = 1.0
        self._norm = 1.0 / np.sqrt(np.sum(signal.sosfilt(self._sos, impulse) ** 2))
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
    """emg[n, J] = volitional[n, M] @ S^T + sensor white noise. Components toggle in config."""

    def __init__(
        self,
        cfg: EMGConfig,
        muscles: list[str],
        fs_hz: float,
        rng_volitional: np.random.Generator,
        rng_noise: np.random.Generator,
        source: VolitionalSource | None = None,
    ):
        self.cfg = cfg
        self.names = [c.name for c in cfg.channels]
        self.S = np.array([[c.weights.get(m, 0.0) for m in muscles] for c in cfg.channels])
        self.source = source or BandlimitedNoiseSource(cfg, muscles, fs_hz, rng_volitional)
        self._rng_noise = rng_noise

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def reset(self) -> None:
        self.source.reset()

    def chunk(self, u_vol: np.ndarray) -> np.ndarray:
        """Raw EMG for n consecutive samples given the volitional excitation at each."""
        n = len(u_vol)
        emg = np.zeros((n, self.n_channels))
        if self.cfg.volitional.enabled:
            emg += self.source.generate(u_vol) @ self.S.T
        if self.cfg.white_noise.enabled:
            emg += self.cfg.white_noise.std_mv * self._rng_noise.standard_normal(emg.shape)
        return emg
