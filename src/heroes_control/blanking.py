"""Stim-artifact blanking: replace raw EMG for a window after every delivered stim pulse.

The stimulator's sync signal tells the controller when it fired (`on_pulse(t)`), as on
hardware. Blanking runs first in the front end, so the artifact and M-wave never enter
the bandpass, where they would ring and inflate the envelope.

Fill modes:
  interp  linear interpolation from the last sample before the blank to the first sample
          after it (default, D15). Needs that next sample, so the stage delays its output by
          window + 1 samples (constant, so still causal and chunking-independent). No step at
          the end of the blank: the held value would differ from the live signal by the mains
          and wander, and that step rings through the bandpass as fake EMG.
  hold    sample-and-hold the last pre-pulse sample. No delay, but leaves that step.
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

FILL_MODES = ("interp", "hold", "zero")


class StimBlanking:
    def __init__(self, fs_hz: float, window_s: float, n_channels: int, fill: str = "interp"):
        if fill not in FILL_MODES:
            raise ValueError(f"fill must be one of {FILL_MODES}, got {fill!r}")
        self._fs = fs_hz
        self._window = window_s
        self._n = n_channels
        self._fill = fill
        self.delay = round(window_s * fs_hz) + 1 if fill == "interp" else 0  # samples
        self.reset()

    def reset(self) -> None:
        self._pulses: deque[float] = deque()
        self._last = np.zeros(self._n)  # last unblanked sample (hold value)
        self.last_mask = np.zeros(0, dtype=bool)  # blanked samples of the last chunk
        # interp: delay line of (sample, blanked) pairs; starts as unblanked zeros.
        self._buf = np.zeros((self.delay, self._n))
        self._bmask = np.zeros(self.delay, dtype=bool)
        self._anchor = np.zeros(self._n)  # last unblanked sample already output
        self._run_done = 0  # samples of the current blanked run already output

    def on_pulse(self, t: float) -> None:
        """Stimulator fired at t; the samples after t within the window are blanked."""
        self._pulses.append(t)

    def mask(self, t: float, n: int) -> np.ndarray:
        """Which of the n samples ending at t fall in a blank (no state change)."""
        mask = np.zeros(n, dtype=bool)
        # Work in whole samples: sample i of this chunk is `ago[i]` samples before t.
        ago = n - 1 - np.arange(n)
        width = round(self._window * self._fs)
        for p in self._pulses:
            since = round((t - p) * self._fs) - ago  # samples since the pulse
            mask |= (since >= 1) & (since <= width)
        return mask

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if n == 0:
            return chunk
        mask = self.mask(t, n)
        width = round(self._window * self._fs)
        while self._pulses and round((t - self._pulses[0]) * self._fs) >= width:
            self._pulses.popleft()
        if self._fill == "interp":
            return self._interp(chunk, mask)
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

    def _interp(self, chunk: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Append the chunk to the delay line and emit the oldest n samples, with each
        blanked run replaced by a line between its neighbours. A run can straddle calls:
        `_run_done` counts how much of it was already emitted."""
        n = len(chunk)
        buf = np.concatenate([self._buf, chunk])
        bm = np.concatenate([self._bmask, mask])
        out = buf[:n].copy()
        anchor, done = self._anchor, self._run_done
        i = 0
        while i < n:
            if not bm[i]:
                anchor, done = buf[i], 0
                i += 1
                continue
            j = i
            while j < len(bm) and bm[j]:
                j += 1
            end = min(j, n)
            if j < len(bm):  # right neighbour available: interpolate
                span = done + (j - i) + 1
                w = (done + np.arange(1, end - i + 1))[:, None] / span
                out[i:end] = anchor + w * (buf[j] - anchor)
            else:  # run longer than the delay (overlapping pulses): hold
                out[i:end] = anchor
            done = done + (end - i) if j >= n else 0  # run continues into the next call?
            i = end
        self._anchor, self._run_done = anchor, done
        self._buf, self._bmask = buf[n:], bm[n:]
        self.last_mask = bm[:n]
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


class MainsCanceller:
    """Adaptive mains canceller (D17): subtracts a fitted sum of sinusoids at the mains
    frequency and harmonics before blanking, so a blank no longer cuts a gap out of the mains
    (interpolating across 15 ms of 50 Hz leaves a residual at 50 +- k * pulse rate that the
    notch after blanking cannot remove, and on a weak channel it reads as intent).

    LMS on cos/sin references, per channel, one sample at a time (chunking-independent).
    Adaptation is frozen on blanked samples, so artifacts and M-waves never enter the fit
    and, unlike a notch filter, it cannot ring after a pulse. A bias term tracks baseline
    wander for the adaptation error only (not subtracted); without it the wander's gradient
    noise jitters the fit and adds a tail of fake EMG. Time constants: about 2 / mu samples
    for the sinusoids, 1 / mu_bias for the bias. Fast start: the step size begins at the
    running-mean gain (2 / k for the sinusoids, 1 / k for the bias, k = samples fitted) and
    decays to mu, so the fit is converged within a few mains cycles of a reset instead of
    leaving mains in the first ~2 / mu samples, where it would read as intent.
    """

    def __init__(
        self,
        freqs_hz: tuple[float, ...],
        fs_hz: float,
        n_channels: int,
        mu: float,
        mu_bias: float,
        blanking: StimBlanking | None = None,
    ):
        self._w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / fs_hz  # rad/sample
        self._n = n_channels
        self._mu = mu
        self._mu_bias = mu_bias
        self._blanking = blanking
        self.reset()

    def reset(self) -> None:
        self._coef = np.zeros((2 * len(self._w), self._n))
        self._bias = np.zeros(self._n)
        self._k = 0  # samples since reset (phase reference)
        self._fitted = 0  # unblanked samples fitted since reset (fast-start step size)

    def process(self, t: float, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if n == 0:
            return chunk
        blanked = (
            self._blanking.mask(t, n) if self._blanking is not None else np.zeros(n, dtype=bool)
        )
        phase = np.outer(self._k + np.arange(n), self._w)
        ref = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)  # [n, 2H]
        self._k += n
        out = np.empty_like(chunk)
        coef, bias, k = self._coef, self._bias, self._fitted
        for i in range(n):  # sequential: each sample's fit uses the previous update
            out[i] = chunk[i] - ref[i] @ coef
            if not blanked[i]:
                k += 1
                e = out[i] - bias
                bias = bias + max(self._mu_bias, 1.0 / k) * e
                coef = coef + max(self._mu, 2.0 / (k + 1)) * np.outer(ref[i], e)
        self._coef, self._bias, self._fitted = coef, bias, k
        return out
