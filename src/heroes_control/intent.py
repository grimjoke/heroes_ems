"""Agonist/antagonist split: normalized envelopes -> signed intent in [-1, 1] per joint."""

from __future__ import annotations

import numpy as np


class AgonistAntagonistIntent:
    """intent_j = normalized[agonist_j] - normalized[antagonist_j], through a deadband.

    The deadband removes |raw| <= deadband and rescales the rest so the output still
    spans [-1, 1] continuously. deadband = 0 disables it.
    """

    def __init__(self, agonist: tuple[int, ...], antagonist: tuple[int, ...], deadband: float):
        self._ag = np.asarray(agonist, dtype=int)
        self._ant = np.asarray(antagonist, dtype=int)
        self._db = deadband

    def reset(self) -> None:
        pass

    def __call__(self, normalized: np.ndarray) -> np.ndarray:
        raw = normalized[self._ag] - normalized[self._ant]
        shrunk = np.sign(raw) * np.maximum(np.abs(raw) - self._db, 0.0) / (1.0 - self._db)
        return np.clip(shrunk, -1.0, 1.0)
