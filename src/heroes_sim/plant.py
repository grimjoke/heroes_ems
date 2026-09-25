"""MuJoCo musculoskeletal plant wrapper (MyoSuite MJCF loaded directly, no gym env)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from heroes_sim.config import PlantConfig, TimingConfig


@dataclass(frozen=True)
class MuscleState:
    activation: np.ndarray
    force: np.ndarray
    length: np.ndarray


def resolve_model_path(cfg: PlantConfig) -> Path:
    if cfg.model_file is not None:
        path = Path(cfg.model_file)
    else:
        import myosuite

        path = Path(myosuite.__file__).parent / cfg.model_relpath
    if not path.is_file():
        raise FileNotFoundError(f"MyoSuite model not found at {path}")
    return path


class Plant:
    def __init__(self, cfg: PlantConfig, timing: TimingConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(resolve_model_path(cfg)))
        # Timestep is set explicitly; the MJCF default (2 ms) is not trusted.
        self.model.opt.timestep = 1.0 / timing.physics_hz
        if not cfg.gravity:
            self.model.opt.gravity[:] = 0.0
        self.data = mujoco.MjData(self.model)

        self._joint_ids = self._lookup(mujoco.mjtObj.mjOBJ_JOINT, cfg.joints, "joint")
        self._act_ids = self._lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.muscles, "muscle")
        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]
        self._dof_adr = self.model.jnt_dofadr[self._joint_ids]
        self.joint_range = self.model.jnt_range[self._joint_ids].copy()
        self._apply_exo_payload()
        self.reset(np.zeros(len(cfg.joints)), np.zeros(len(cfg.joints)))

    @property
    def n_joints(self) -> int:
        return len(self._joint_ids)

    @property
    def n_muscles(self) -> int:
        return len(self._act_ids)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def _lookup(self, obj: mujoco.mjtObj, names: list[str], kind: str) -> np.ndarray:
        ids = []
        for name in names:
            i = mujoco.mj_name2id(self.model, obj, name)
            if i < 0:
                available = [mujoco.mj_id2name(self.model, obj, k) for k in range(self._count(obj))]
                raise ValueError(f"{kind} '{name}' not in model; available: {available}")
            ids.append(i)
        return np.array(ids, dtype=int)

    def _count(self, obj: mujoco.mjtObj) -> int:
        return {
            mujoco.mjtObj.mjOBJ_JOINT: self.model.njnt,
            mujoco.mjtObj.mjOBJ_ACTUATOR: self.model.nu,
        }[obj]

    def _apply_exo_payload(self) -> None:
        """Add a passive point mass to the forearm body (composite mass, COM and inertia)."""
        payload = self.cfg.exo_payload
        if payload.mass_kg == 0.0:
            return
        m = self.model
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.cfg.forearm_body)
        if b < 0:
            raise ValueError(f"forearm_body '{self.cfg.forearm_body}' not in model")
        m0, c0 = m.body_mass[b], m.body_ipos[b].copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[b])
        rot = rot.reshape(3, 3)
        i0 = rot @ np.diag(m.body_inertia[b]) @ rot.T  # inertia about COM, body-aligned axes

        mp, p = payload.mass_kg, np.asarray(payload.pos_m, dtype=float)
        mt = m0 + mp
        ct = (m0 * c0 + mp * p) / mt

        def parallel_axis(mass: float, d: np.ndarray) -> np.ndarray:
            return mass * (d @ d * np.eye(3) - np.outer(d, d))

        it = i0 + parallel_axis(m0, c0 - ct) + parallel_axis(mp, p - ct)
        w, v = np.linalg.eigh(it)
        if np.linalg.det(v) < 0:
            v[:, 0] *= -1
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, v.reshape(-1))
        m.body_mass[b], m.body_ipos[b], m.body_inertia[b], m.body_iquat[b] = mt, ct, w, quat
        m.body_subtreemass[b] = mt  # leaf body; mj_setConst refreshes the rest
        mujoco.mj_setConst(m, mujoco.MjData(m))

    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_adr] = qpos
        self.data.qvel[self._dof_adr] = qvel
        # Muscle activation starts at zero (mj_resetData); forward to populate derived state.
        mujoco.mj_forward(self.model, self.data)

    def step(self, excitation: np.ndarray) -> None:
        """Advance one physics step. Excitation in [0, 1] per muscle -> data.ctrl.

        MuJoCo's built-in muscle activation dynamics are the only activation dynamics.
        """
        excitation = np.asarray(excitation, dtype=np.float64)
        if excitation.shape != (self.n_muscles,):
            raise ValueError(f"excitation shape {excitation.shape}, expected ({self.n_muscles},)")
        self.data.ctrl[self._act_ids] = np.clip(excitation, 0.0, 1.0)
        mujoco.mj_step(self.model, self.data)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self._qpos_adr].copy(),
            self.data.qvel[self._dof_adr].copy(),
        )

    def muscle_state(self) -> MuscleState:
        ids = self._act_ids
        return MuscleState(
            activation=self.data.act[self.model.actuator_actadr[ids]].copy(),
            force=self.data.actuator_force[ids].copy(),
            length=self.data.actuator_length[ids].copy(),
        )
