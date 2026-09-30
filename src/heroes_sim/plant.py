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
        self.action = self._build_action_matrix()
        self._lock: np.ndarray | None = None
        self._apply_exo_payload()
        self._check_action_signs()
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

    def _build_action_matrix(self) -> np.ndarray:
        """A[M, N]: +1 muscle flexes joint, -1 extends, 0 no action (from config groups)."""
        a = np.zeros((self.n_muscles, self.n_joints))
        for g in self.cfg.muscle_groups.values():
            j = self.cfg.joints.index(g.joint)
            for m in g.muscles:
                a[self.cfg.muscles.index(m), j] = g.sign
        return a

    def moment_arms(self) -> np.ndarray:
        """dL/dq [M, N] at the current pose. Negative for a muscle that flexes the joint."""
        m, d = self.model, self.data
        dense = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(
            dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
        return dense[np.ix_(self._act_ids, self._dof_adr)]

    def _check_action_signs(self) -> None:
        """Fail loudly if config muscle groups disagree with the model across the joint range."""
        for frac in (0.1, 0.5, 0.9):
            q = self.joint_range[:, 0] + frac * (self.joint_range[:, 1] - self.joint_range[:, 0])
            self.reset(q, np.zeros(self.n_joints))
            model_sign = -np.sign(self.moment_arms())
            mismatch = (self.action != 0) & (model_sign != self.action)
            if mismatch.any():
                bad = [self.cfg.muscles[i] for i in np.nonzero(mismatch.any(axis=1))[0]]
                raise ValueError(f"muscle_groups sign disagrees with model moment arms: {bad}")

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

    def lock(self, q: np.ndarray) -> None:
        """Hold joints kinematically at q (isometric trials). Muscles still produce force."""
        self._lock = np.asarray(q, dtype=np.float64).copy()
        self._pin()
        mujoco.mj_forward(self.model, self.data)

    def unlock(self) -> None:
        self._lock = None

    def _pin(self) -> None:
        self.data.qpos[self._qpos_adr] = self._lock
        self.data.qvel[self._dof_adr] = 0.0

    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._lock = None
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
        if self._lock is not None:
            self._pin()
        mujoco.mj_step(self.model, self.data)
        if self._lock is not None:
            self._pin()

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self._qpos_adr].copy(),
            self.data.qvel[self._dof_adr].copy(),
        )

    def set_external_torque(self, torque: np.ndarray) -> None:
        """Passive external torque per joint (N m), held until changed; reset clears it."""
        self.data.qfrc_applied[self._dof_adr] = torque

    def actuator_torque(self) -> np.ndarray:
        """Net muscle torque per joint (N m, + flexion) from the last step."""
        return self.data.qfrc_actuator[self._dof_adr].copy()

    def muscle_state(self) -> MuscleState:
        ids = self._act_ids
        return MuscleState(
            activation=self.data.act[self.model.actuator_actadr[ids]].copy(),
            force=self.data.actuator_force[ids].copy(),
            length=self.data.actuator_length[ids].copy(),
        )
