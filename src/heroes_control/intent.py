"""Agonist/antagonist split: normalized envelopes -> signed intent in [-1, 1] per joint."""

from __future__ import annotations

import numpy as np


class AgonistAntagonistIntent:
    """intent_j = normalized[agonist_j] - normalized[antagonist_j], through a deadband.

    The deadband removes |raw| <= deadband and rescales the rest so the output still
    spans [-1, 1] continuously. deadband = 0 disables it.
    """

    def __init__(
        self,
        agonist: tuple[int, ...],
        antagonist: tuple[int, ...],
        deadband: float | tuple[float, ...],
    ):
        self._ag = np.asarray(agonist, dtype=int)
        self._ant = np.asarray(antagonist, dtype=int)
        self._db = np.broadcast_to(np.asarray(deadband, dtype=np.float64), self._ag.shape)

    def reset(self) -> None:
        pass

    def __call__(self, normalized: np.ndarray) -> np.ndarray:
        raw = normalized[self._ag] - normalized[self._ant]
        shrunk = np.sign(raw) * np.maximum(np.abs(raw) - self._db, 0.0) / (1.0 - self._db)
        return np.clip(shrunk, -1.0, 1.0)


def calibrate_deadband(
    normalized_rest: np.ndarray,
    agonist: tuple[int, ...],
    antagonist: tuple[int, ...],
    percentile: float,
    floor: float,
) -> tuple[float, ...]:
    """Per-joint deadband = max(floor, `percentile` of |raw intent|) over a rest recording.

    `normalized_rest[n, J]`: normalized envelopes at rest, recorded with stimulation running
    at a typical level (D14), so the deadband covers the artifact leak too. Rest intent is
    rectified and heavy-tailed, so a tail percentile is used, not k * sigma.
    """
    x = np.asarray(normalized_rest, dtype=np.float64)
    raw = np.abs(x[:, list(agonist)] - x[:, list(antagonist)])
    return tuple(float(max(floor, min(p, 0.99))) for p in np.percentile(raw, percentile, axis=0))
