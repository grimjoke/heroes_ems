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
    k: float,
    floor: float,
) -> tuple[float, ...]:
    """Per-joint deadband = max(floor, k * std of raw resting intent).

    `normalized_rest[n, J]`: normalized envelopes during calibration rest (no effort, no
    stim). Sets the deadband relative to this patient's own noise floor, since in a double
    integrator it is what stands between resting noise and reference drift.
    """
    x = np.asarray(normalized_rest, dtype=np.float64)
    raw = x[:, list(agonist)] - x[:, list(antagonist)]
    return tuple(float(max(floor, min(k * sd, 0.99))) for sd in raw.std(axis=0))
