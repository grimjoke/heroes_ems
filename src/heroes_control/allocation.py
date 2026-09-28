"""Signed PD output per joint -> stim intensity per channel in [0, 1]."""

from __future__ import annotations

import numpy as np


class SignedAllocation:
    """Channel k drives joint `channel_joint[k]` in direction `channel_sign[k]` (+1 / -1).

    Positive PD output goes to the channels with sign +1 (agonist), negative to sign -1
    (antagonist): v_k = sign_k * u[joint_k].

    offset = 0 (default): s_k = clip(v_k, 0, 1), the plain split.
    offset > 0: threshold compensation, s_k = offset + (1 - offset) * v_k, applied only
    when v_k > epsilon, else 0. The epsilon gate keeps tiny noisy outputs from jumping
    straight to threshold-level stimulation.
    """

    def __init__(
        self,
        channel_joint: tuple[int, ...],
        channel_sign: tuple[int, ...],
        offset: float = 0.0,
        epsilon: float = 0.0,
    ):
        if not 0 <= offset < 1:
            raise ValueError("offset must be in [0, 1)")
        if offset > 0 and epsilon <= 0:
            raise ValueError("a threshold offset needs an epsilon gate > 0")
        self._joint = np.asarray(channel_joint, dtype=int)
        self._sign = np.asarray(channel_sign, dtype=np.float64)
        self._offset, self._eps = offset, epsilon

    def reset(self) -> None:
        pass

    def __call__(self, pd_output: np.ndarray) -> np.ndarray:
        v = self._sign * pd_output[self._joint]
        if self._offset == 0:
            return np.clip(v, 0.0, 1.0)
        s = np.where(v > self._eps, self._offset + (1.0 - self._offset) * v, 0.0)
        return np.clip(s, 0.0, 1.0)
