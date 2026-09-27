"""Wires plant, patient and scenario; runs one episode. M1: open loop (no controller/stim)."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from heroes_sim.config import MovementScenario, MVCScenario, SimConfig
from heroes_sim.patient import NO_INTENT, Intent, Patient
from heroes_sim.plant import Plant
from heroes_sim.scenario import MVCSchedule, TargetTrajectory
from heroes_sim.scheduler import Scheduler


def make_rng(seed: int, component: str) -> np.random.Generator:
    """Per-component stream from the run seed. Keyed by name, so adding a component
    never shifts another component's stream."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(component.encode())]))


@dataclass(frozen=True)
class MVCTrialResult:
    """Best `window_s` of one max-effort trial (the window with highest group activation)."""

    group: str
    activation: np.ndarray  # [M] mean activation over the window
    torque: np.ndarray  # [N] mean net muscle torque over the window, N m


@dataclass(frozen=True)
class RunResult:
    log: pd.DataFrame  # controller-rate log
    mvc: list[MVCTrialResult] | None = None


StepHook = Callable[[Plant], None]


def _require(cfg: SimConfig) -> None:
    if cfg.patient is None or cfg.scenario is None:
        raise ValueError("run config needs `patient` and `scenario` (use load_run_config)")


def run(cfg: SimConfig, seed: int, on_step: StepHook | None = None) -> RunResult:
    _require(cfg)
    if isinstance(cfg.scenario, MVCScenario):
        return run_mvc(cfg, seed, on_step)
    return RunResult(log=run_movement(cfg, seed, on_step))


def _episode(
    cfg: SimConfig,
    seed: int,
    duration_s: float,
    intent_at: Callable[[float], Intent],
    q0: np.ndarray,
    qd0: np.ndarray,
    lock: bool,
    on_step: StepHook | None,
) -> pd.DataFrame:
    plant = Plant(cfg.plant, cfg.timing)
    plant.reset(q0, qd0)
    if lock:
        plant.lock(q0)
    patient = Patient(cfg.patient, cfg.plant, plant.action, plant.dt, make_rng(seed, "patient"))
    sched = Scheduler(cfg.timing)

    n_steps = round(duration_s * cfg.timing.physics_hz)
    n_log = n_steps // (cfg.timing.physics_hz // cfg.timing.controller_hz)
    joints, muscles = cfg.plant.joints, cfg.plant.muscles
    nj, nm = len(joints), len(muscles)
    t_log = np.empty(n_log)
    target, q_log, qd_log, tau_log = (np.empty((n_log, nj)) for _ in range(4))
    u_log, act_log = np.empty((n_log, nm)), np.empty((n_log, nm))
    group_log = np.empty(n_log, dtype=object)

    k = 0
    for _ in range(n_steps):
        intent = intent_at(sched.t)
        q, qd = plant.joint_state()
        u_vol = patient.step(intent, q, qd)
        plant.step(u_vol)
        if on_step is not None:
            on_step(plant)
        if sched.advance().controller:
            q, qd = plant.joint_state()
            t_log[k] = sched.t
            target[k] = np.nan if intent.target is None else intent.target
            q_log[k], qd_log[k], tau_log[k] = q, qd, plant.actuator_torque()
            u_log[k], act_log[k] = u_vol, plant.muscle_state().activation
            group_log[k] = intent.mvc_group or ""
            k += 1

    cols: dict[str, np.ndarray] = {"t": t_log}
    for j, name in enumerate(joints):
        cols |= {
            f"target_{name}": target[:, j],
            f"q_{name}": q_log[:, j],
            f"qd_{name}": qd_log[:, j],
            f"torque_{name}": tau_log[:, j],
        }
    for m, name in enumerate(muscles):
        cols |= {f"u_vol_{name}": u_log[:, m], f"act_{name}": act_log[:, m]}
    cols["mvc_group"] = group_log
    return pd.DataFrame(cols)


def run_movement(cfg: SimConfig, seed: int, on_step: StepHook | None = None) -> pd.DataFrame:
    _require(cfg)
    sc = cfg.scenario
    assert isinstance(sc, MovementScenario)
    q0 = np.asarray(sc.initial_state.q, dtype=np.float64)
    qd0 = np.asarray(sc.initial_state.qd, dtype=np.float64)
    if sc.target is None:

        def intent_at(t: float) -> Intent:
            return NO_INTENT
    else:
        traj = TargetTrajectory(sc.target, q0)

        def intent_at(t: float) -> Intent:
            return Intent(target=traj(t))

    return _episode(cfg, seed, sc.duration_s, intent_at, q0, qd0, False, on_step)


def run_mvc(cfg: SimConfig, seed: int, on_step: StepHook | None = None) -> RunResult:
    """Isometric max-effort trial per muscle group with the joint locked, as on hardware.

    M2 adds the EMG envelope to the log; the controller's MVC values come from that.
    """
    _require(cfg)
    sc = cfg.scenario
    assert isinstance(sc, MVCScenario)
    sched = MVCSchedule(sc)
    lock_q = np.asarray(sc.lock_q, dtype=np.float64)

    def intent_at(t: float) -> Intent:
        g = sched.group_at(t)
        return NO_INTENT if g is None else Intent(mvc_group=g)

    log = _episode(
        cfg, seed, sched.duration_s, intent_at, lock_q, np.zeros_like(lock_q), True, on_step
    )
    return RunResult(log=log, mvc=[_best_window(cfg, sc, log, w.group) for w in sched.windows])


def _best_window(cfg: SimConfig, sc: MVCScenario, log: pd.DataFrame, group: str) -> MVCTrialResult:
    rows = log[log["mvc_group"] == group]
    members = cfg.plant.muscle_groups[group].muscles
    act = rows[[f"act_{m}" for m in cfg.plant.muscles]].to_numpy()
    tau = rows[[f"torque_{j}" for j in cfg.plant.joints]].to_numpy()
    w = round(sc.window_s * cfg.timing.controller_hz)
    score = rows[[f"act_{m}" for m in members]].to_numpy().mean(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(score)])
    start = int(np.argmax(csum[w:] - csum[:-w]))
    return MVCTrialResult(
        group=group,
        activation=act[start : start + w].mean(axis=0),
        torque=tau[start : start + w].mean(axis=0),
    )
