"""Stim-artifact blanking: replace raw EMG for a window after every delivered stim pulse.

The stimulator's sync signal tells the controller when it fired (`on_pulse(t)`), as on
hardware. Blanking runs first in the front end, so the artifact and M-wave never enter
the bandpass, where they would ring and inflate the envelope.

Fill modes:
  hold  sample-and-hold the last pre-pulse sample (default). Keeps slow offsets
        (baseline wander, electrode DC) continuous, so the bandpass sees no edge.
  zero  set the window to 0. Cuts a rectangular notch out of any offset; the bandpass
        turns each notch into a transient that reads as EMG. Kept for comparison.
Cost either way: volitional EMG inside the window is lost. With hold, the held sample
keeps the envelope roughly level. With zero, the envelope reads low by the blanked
fraction, and asymmetrically (only the stimulated side's channels see many zeros), so zero
fill must be paired with `BlankingCorrection`; `ControllerConfig` enforces that.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal

FILL_MODES = ("hold", "zero")


class StimBlanking:
    def __init__(self, fs_hz: float, window_s: float, n_channels: int, fill: str = "hold"):
        if fill not in FILL_MODES:
            raise ValueError(f"fill must be one of {FILL_MODES}, got {fill!r}")
        self._fs = fs_hz
        self._window = window_s
        self._n = n_channels
        self._fill = fill
        self.reset()

    def reset(self) -> None:
        self._pulses: deque[float] = deque()
        self._last = np.zeros(self._n)  # last unblanked sample (hold value)
        self.last_mask = np.zeros(0, dtype=bool)  # blanked samples of the last chunk

    def on_pulse(self, t: float) -> None:
        """Stimulator fired at t; the samples after t within the window are blanked."""
        self._pulses.append(t)

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if n == 0:
            return chunk
        mask = np.zeros(n, dtype=bool)
        if self._pulses:
            # Work in whole samples: sample i of this chunk is `ago[i]` samples before t.
            ago = n - 1 - np.arange(n)
            width = round(self._window * self._fs)
            for p in self._pulses:
                since = round((t - p) * self._fs) - ago  # samples since the pulse
                mask |= (since >= 1) & (since <= width)
            while self._pulses and round((t - self._pulses[0]) * self._fs) >= width:
                self._pulses.popleft()
        self.last_mask = mask
        if not mask.any():
            self._last = chunk[-1].copy()
            return chunk
        out = chunk.copy()
        if self._fill == "zero":
            out[mask] = 0.0
            last_kept = np.nonzero(~mask)[0]
            if len(last_kept):
                self._last = chunk[last_kept[-1]].copy()
            return out
        hold = self._last
        for i in range(n):  # sequential: a hold value carries across blanked runs
            if mask[i]:
                out[i] = hold
            else:
                hold = chunk[i]
        self._last = hold.copy()
        return out


class BlankingCorrection:
    """Divide the envelope by the recently kept fraction of samples (zero-fill only).

    The kept-sample indicator goes through the same lowpass as the envelope, so the
    correction has the same dynamics as the bias it removes. It starts at steady state
    (fraction 1), and is floored so a fully blanked stretch cannot divide by ~0.
    """

    def __init__(self, blanking: StimBlanking, sos: np.ndarray, floor: float = 0.2):
        self._blanking = blanking
        self._sos = np.asarray(sos, dtype=np.float64)
        self._floor = floor
        self.reset()

    def reset(self) -> None:
        self._zi = signal.sosfilt_zi(self._sos)[:, :, None]  # steady state for input 1

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if n == 0:
            return chunk
        mask = self._blanking.last_mask
        kept = np.ones((n, 1)) if len(mask) != n else (~mask).astype(np.float64)[:, None]
        frac, self._zi = signal.sosfilt(self._sos, kept, axis=0, zi=self._zi)
        return chunk / np.maximum(frac, self._floor)
