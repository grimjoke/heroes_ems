"""MVC normalization with resting-baseline subtraction, per EMG channel.

normalized = max(envelope - rest, 0) / (mvc - rest)

Without the baseline, a weak channel's noise floor (sensor noise + crosstalk + resting
tone) divided by its small MVC reads as steady intent: with a near-paralysed triceps the
controller would push extension against every flexion. Both values come from the same
calibration trial (rest periods and max efforts), as on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MVCValues:
    """Calibration results. Per EMG channel: envelope at max voluntary contraction and at
    rest. Per joint: intent deadband derived from resting intent noise (None -> use the
    controller's `deadband_floor`)."""

    envelope: tuple[float, ...]
    rest: tuple[float, ...] | None = None  # None -> zero baseline
    deadband: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) and v > 0 for v in self.envelope):
            raise ValueError(f"MVC envelope values must be finite and > 0: {self.envelope}")
        if self.rest is not None:
            if len(self.rest) != len(self.envelope):
                raise ValueError("rest and envelope must have the same length")
            if not all(0 <= r < m for r, m in zip(self.rest, self.envelope, strict=True)):
                raise ValueError(
                    f"need 0 <= rest < MVC per channel: {self.rest} vs {self.envelope}"
                )
        if self.deadband is not None and not all(0 <= d < 1 for d in self.deadband):
            raise ValueError(f"deadband values must be in [0, 1): {self.deadband}")

    @property
    def baseline(self) -> np.ndarray:
        return np.zeros(len(self.envelope)) if self.rest is None else np.asarray(self.rest)


class MVCNormalizer:
    """normalized = max(envelope - rest, 0) / (mvc - rest); clamped at zero after the
    subtraction, so a below-rest envelope never reads as negative effort."""

    def __init__(self, mvc: MVCValues):
        self._rest = mvc.baseline
        self._span = np.asarray(mvc.envelope, dtype=np.float64) - self._rest

    def reset(self) -> None:
        pass

    def __call__(self, envelope: np.ndarray) -> np.ndarray:
        return np.maximum(envelope - self._rest, 0.0) / self._span
