"""Signed PD output per joint -> stim intensity per channel in [0, 1]."""

from __future__ import annotations

import numpy as np


class SignedAllocation:
    """Channel k drives joint `channel_joint[k]` in direction `channel_sign[k]` (+1 / -1).

    Positive PD output goes to the channels with sign +1 (agonist), negative to sign -1
    (antagonist): s_k = clip(sign_k * u[joint_k], 0, 1).
    """

    def __init__(self, channel_joint: tuple[int, ...], channel_sign: tuple[int, ...]):
        self._joint = np.asarray(channel_joint, dtype=int)
        self._sign = np.asarray(channel_sign, dtype=np.float64)

    def reset(self) -> None:
        pass

    def __call__(self, pd_output: np.ndarray) -> np.ndarray:
        return np.clip(self._sign * pd_output[self._joint], 0.0, 1.0)
