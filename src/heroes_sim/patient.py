"""Volitional activation model: intent -> excitation u_vol[M].

No activation dynamics here: MuJoCo's muscle model is the only one in the system.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from heroes_sim.config import PatientConfig, PlantConfig


@dataclass(frozen=True)
class Intent:
    """What the patient is trying to do at one instant.

    target: intended joint angles [N], or None for no movement intent.
    mvc_group: muscle group to drive at maximal voluntary effort (overrides target).
    """

    target: np.ndarray | None = None
    mvc_group: str | None = None


NO_INTENT = Intent()


class Patient:
    """Intended velocity toward target, split agonist/antagonist, scaled by impairment.

    Per joint: v_int = clip((target - q) / tau, +-max_speed); drive = gain * (v_int - qd).
    Muscles whose action matches sign(drive) get |drive|; everyone gets `cocontraction`.
    Motor noise is added, then the command is clipped to [0, 1] and scaled by `strength`,
    so a paralysed muscle (strength 0) never fires. `onset_delay_ms` delays the intent
    (reaction to a cue); proprioception of q, qd is not delayed.
    """

    def __init__(
        self,
        cfg: PatientConfig,
        plant_cfg: PlantConfig,
        action: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ):
        self.cfg = cfg
        self.action = np.asarray(action, dtype=np.float64)  # [M, N]
        self.strength = np.array([cfg.strength[m] for m in plant_cfg.muscles])
        self._groups = {
            name: np.isin(plant_cfg.muscles, g.muscles)
            for name, g in plant_cfg.muscle_groups.items()
        }
        self._delay_steps = round(cfg.onset_delay_ms * 1e-3 / dt)
        self._rng = rng
        self.reset()

    def reset(self) -> None:
        n = self._delay_steps + 1
        self._line: deque[Intent] = deque([NO_INTENT] * n, maxlen=n)

    def step(self, intent: Intent, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        self._line.append(intent)
        now = self._line[0]
        d = self.cfg.drive
        cmd = np.full(len(self.strength), self.cfg.cocontraction)
        if now.mvc_group is not None:
            cmd[self._groups[now.mvc_group]] = 1.0
        elif now.target is not None:
            v_int = np.clip(
                (now.target - q) / d.approach_tau_s, -d.max_speed_rad_s, d.max_speed_rad_s
            )
            drive = d.gain * (v_int - qd)
            cmd += np.maximum(self.action @ drive, 0.0)
        cmd += self.cfg.noise_std * self._rng.standard_normal(len(cmd))
        return self.strength * np.clip(cmd, 0.0, 1.0)
