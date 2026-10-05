"""Causal EMG front-end stages: bandpass, full-wave rectification, lowpass envelope.

Every stage is streaming: `process(t, chunk[n, J]) -> chunk[n, J]` with state carried
between calls, so any chunking of the same signal gives the same output. Never filtfilt.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import signal


class EMGStage(Protocol):
    """A streaming stage on raw-rate EMG. `t` is the time of the chunk's last sample."""

    def reset(self) -> None: ...

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray: ...


class CausalSOS:
    """Per-channel SOS IIR filter with carried state (zero initial conditions)."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        self._zi = np.zeros((self.sos.shape[0], 2, self.n_channels))

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return np.empty((0, self.n_channels))
        y, self._zi = signal.sosfilt(self.sos, chunk, axis=0, zi=self._zi)
        return y


def bandpass(lo_hz: float, hi_hz: float, order: int, fs_hz: float, n_channels: int) -> CausalSOS:
    sos = signal.butter(order, [lo_hz, hi_hz], btype="bandpass", fs=fs_hz, output="sos")
    return CausalSOS(sos, n_channels)


def notch(freq_hz: float, q: float, fs_hz: float, n_channels: int) -> CausalSOS:
    """Causal IIR notch (mains). Runs after blanking: before it, each artifact spike would
    make it ring for tens of ms, smearing the artifact outside the blank."""
    b, a = signal.iirnotch(freq_hz, q, fs=fs_hz)
    return CausalSOS(signal.tf2sos(b, a), n_channels)


def lowpass(cutoff_hz: float, order: int, fs_hz: float, n_channels: int) -> CausalSOS:
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
    return CausalSOS(sos, n_channels)


class Rectify:
    """Full-wave rectification (stateless)."""

    def reset(self) -> None:
        pass

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        return np.abs(chunk)


class EMGFrontEnd:
    """Ordered EMG stages. Insert a stage (e.g. stim-artifact blanking) by composing a list."""

    def __init__(self, stages: list[EMGStage]):
        self.stages = list(stages)

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float64)
        for s in self.stages:
            x = s.process(t, x)
        return x
