"""Wires everything and runs one episode.

Per physics step: intent -> patient u_vol -> stim u_stim -> combine -> plant.
Clock ticks: angle sensor sample; controller tick (EMG chunk -> controller -> latency
queue -> safety supervisor -> applied intensity); stim pulse (latches applied intensity).
Closed-loop movement episodes calibrate MVC first (as on hardware) unless given.
"""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from heroes_control import (
    ControllerConfig,
    ControllerPipeline,
    MVCValues,
    calibrate_deadband,
    default_front_end,
)
from heroes_control.normalization import MVCNormalizer
from heroes_safety import RULES, SafetyConfig, SafetyEvent, SafetySupervisor
from heroes_sim.config import (
    AngleFreeze,
    ArtifactIncrease,
    ElectrodeDetach,
    EMGDropout,
    EMGSaturation,
    Fault,
    MovementScenario,
    MVCScenario,
    Perturbation,
    SimConfig,
)
from heroes_sim.patient import NO_INTENT, Intent, Patient
from heroes_sim.plant import Plant
from heroes_sim.scenario import MVCSchedule, TargetTrajectory
from heroes_sim.scheduler import Scheduler
from heroes_sim.sensors.emg import EMGSensor
from heroes_sim.sensors.kinematics import AngleSensor
from heroes_sim.stim import StimModel


def make_rng(seed: int, component: str) -> np.random.Generator:
    """Per-component stream from the run seed. Keyed by name, so adding a component
    never shifts another component's stream."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(component.encode())]))


def build_controller_config(cfg: SimConfig) -> ControllerConfig:
    """Resolve the name-based YAML sections into the controller's index-based config."""
    c, emg = cfg.controller, [ch.name for ch in cfg.emg.channels]
    joints = [c.joints[j] for j in cfg.plant.joints]
    return ControllerConfig(
        emg_fs_hz=float(cfg.timing.emg_hz),
        tick_hz=float(cfg.timing.controller_hz),
        n_emg=len(emg),
        bandpass_hz=c.bandpass_hz,
        bandpass_order=c.bandpass_order,
        envelope_hz=c.envelope_hz,
        envelope_order=c.envelope_order,
        blanking_s=c.blanking_ms * 1e-3,
        blanking_fill=c.blanking_fill,
        blanking_correction=c.blanking_correction,
        agonist=tuple(emg.index(j.agonist) for j in joints),
        antagonist=tuple(emg.index(j.antagonist) for j in joints),
        deadband_floor=c.deadband_floor,
        deadband_k=c.deadband_k,
        gain=tuple(j.gain for j in joints),
        damping=tuple(j.damping for j in joints),
        q_min=tuple(j.rom[0] for j in joints),
        q_max=tuple(j.rom[1] for j in joints),
        kp=tuple(j.kp for j in joints),
        kd=tuple(j.kd for j in joints),
        derivative_tau_s=c.derivative_tau_s,
        channel_joint=tuple(cfg.plant.joints.index(ch.joint) for ch in cfg.stim.channels),
        channel_sign=tuple(ch.sign for ch in cfg.stim.channels),
        allocation_offset=c.allocation.offset,
        allocation_epsilon=c.allocation.epsilon,
    )


def build_safety_config(cfg: SimConfig) -> SafetyConfig:
    s, joints, chans = cfg.safety, cfg.plant.joints, cfg.stim.channels
    return SafetyConfig(
        tick_hz=float(cfg.timing.controller_hz),
        channel_joint=tuple(joints.index(ch.joint) for ch in chans),
        channel_sign=tuple(ch.sign for ch in chans),
        cap=tuple(s.cap[ch.name] for ch in chans),
        max_rise_per_tick=s.max_rise_per_tick,
        q_limit_min=tuple(s.joint_limits[j][0] for j in joints),
        q_limit_max=tuple(s.joint_limits[j][1] for j in joints),
        rom_margin_rad=s.rom_margin_rad,
        dose_window_s=s.dose_window_s,
        dose_max_mean=s.dose_max_mean,
        watchdog_timeout_s=s.watchdog_timeout_s,
        q_plausible_min=tuple(s.q_plausible[j][0] for j in joints),
        q_plausible_max=tuple(s.q_plausible[j][1] for j in joints),
        emg_plausible_abs=s.emg_plausible_abs_mv,
    )


@dataclass(frozen=True)
class MVCTrialResult:
    """Best `window_s` of one max-effort trial (the window with highest group activation)."""

    group: str
    activation: np.ndarray  # [M] mean activation over the window
    torque: np.ndarray  # [N] mean net muscle torque over the window, N m


@dataclass(frozen=True)
class Calibration:
    log: pd.DataFrame
    trials: list[MVCTrialResult]
    mvc: MVCValues  # MVC + rest per EMG channel, deadband per joint


@dataclass(frozen=True)
class RunResult:
    log: pd.DataFrame  # controller-rate log
    mvc: MVCValues | None = None
    calibration: Calibration | None = None  # when calibration ran in this run
    events: list[SafetyEvent] = field(default_factory=list)
    baseline_log: pd.DataFrame | None = None  # stim-off run, same seed (stim_off_baseline)


StepHook = Callable[[Plant], None]


def run(cfg: SimConfig, seed: int, on_step: StepHook | None = None) -> RunResult:
    if cfg.patient is None or cfg.scenario is None:
        raise ValueError("run config needs `patient` and `scenario` (use load_run_config)")
    sc = cfg.scenario
    if isinstance(sc, MVCScenario):
        cal = calibrate(cfg, seed, "", on_step)
        return RunResult(log=cal.log, mvc=cal.mvc, calibration=cal)
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

    cal, mvc = None, None
    if sc.closed_loop:
        if sc.calibrated is not None:
            p = sc.calibrated
            names = [ch.name for ch in cfg.emg.channels]
            mvc = MVCValues(
                tuple(p.envelope[n] for n in names),
                tuple(p.rest[n] for n in names),
                tuple(p.deadband[j] for j in cfg.plant.joints),
            )
        else:
            cal = calibrate(cfg, seed, "calibration/", None)
            mvc = cal.mvc
    ep = _Episode(cfg, seed, "", q0, qd0, lock=False, mvc=mvc)
    log = ep.run(sc.duration_s, intent_at, on_step, sc.faults, sc.perturbations)
    baseline = None
    if sc.stim_off_baseline and sc.closed_loop:
        # Same seed and component names -> same patient/sensor noise; only stim differs.
        base_ep = _Episode(cfg, seed, "", q0, qd0, lock=False, mvc=None)
        baseline = base_ep.run(sc.duration_s, intent_at, None, sc.faults, sc.perturbations)
    return RunResult(log=log, mvc=mvc, calibration=cal, events=ep.events, baseline_log=baseline)


def calibrate(cfg: SimConfig, seed: int, prefix: str, on_step: StepHook | None) -> Calibration:
    """MVC trials with the joint locked; EMG through the controller's own front end."""
    cal = cfg.calibration
    sched = MVCSchedule(cal)
    lock_q = np.asarray(cal.lock_q, dtype=np.float64)

    def intent_at(t: float) -> Intent:
        g = sched.group_at(t)
        return NO_INTENT if g is None else Intent(mvc_group=g)

    ep = _Episode(cfg, seed, prefix, lock_q, np.zeros_like(lock_q), lock=True, mvc=None)
    log = ep.run(sched.duration_s, intent_at, on_step)
    trials = [_best_window(cfg, log, w.group) for w in sched.windows]
    w = round(cal.window_s * cfg.timing.controller_hz)
    t = log["t"].to_numpy()
    # Rest baseline: the last window_s before each effort cue (envelope settled, no effort).
    at_rest = np.zeros(len(t), dtype=bool)
    for win in sched.windows:
        at_rest |= (t > win.t_start - cal.window_s) & (t <= win.t_start)
    mvc, rest = [], []
    for ch in cfg.emg.channels:
        env = log[f"env_{ch.name}"].to_numpy()
        mvc.append(float(_best_mean(env[log["mvc_group"] == ch.mvc_group], w)[0]))
        rest.append(float(env[at_rest].mean()))
    # Deadband from resting intent noise, through the controller's own normalization.
    ctrl = build_controller_config(cfg)
    norm = MVCNormalizer(MVCValues(tuple(mvc), tuple(rest)))
    env_rest = log.loc[at_rest, [f"env_{ch.name}" for ch in cfg.emg.channels]].to_numpy()
    deadband = calibrate_deadband(
        np.array([norm(e) for e in env_rest]),
        ctrl.agonist,
        ctrl.antagonist,
        ctrl.deadband_k,
        ctrl.deadband_floor,
    )
    values = MVCValues(tuple(mvc), tuple(rest), deadband)
    return Calibration(log=log, trials=trials, mvc=values)


def _best_mean(x: np.ndarray, w: int) -> tuple[float, int]:
    """Highest mean over any w consecutive samples of x, and where it starts."""
    csum = np.concatenate([[0.0], np.cumsum(x)])
    sums = csum[w:] - csum[:-w]
    start = int(np.argmax(sums))
    return sums[start] / w, start


def _best_window(cfg: SimConfig, log: pd.DataFrame, group: str) -> MVCTrialResult:
    rows = log[log["mvc_group"] == group]
    members = cfg.plant.muscle_groups[group].muscles
    w = round(cfg.calibration.window_s * cfg.timing.controller_hz)
    _, start = _best_mean(rows[[f"act_{m}" for m in members]].to_numpy().mean(axis=1), w)
    act = rows[[f"act_{m}" for m in cfg.plant.muscles]].to_numpy()
    tau = rows[[f"torque_{j}" for j in cfg.plant.joints]].to_numpy()
    return MVCTrialResult(
        group=group,
        activation=act[start : start + w].mean(axis=0),
        torque=tau[start : start + w].mean(axis=0),
    )


def _active(item: Fault | Perturbation, t: float) -> bool:
    return item.t_start <= t and (item.t_end is None or t < item.t_end)


class _Injector:
    """Switches scenario faults and perturbations on and off at their times."""

    def __init__(self, ep: _Episode, faults: list[Fault], perturbations: list[Perturbation]):
        self.ep, self.faults, self.perturbations = ep, faults, perturbations
        self.state = [False] * len(faults)
        cfg = ep.cfg
        self._stim = [ch.name for ch in cfg.stim.channels]
        self._joints = cfg.plant.joints
        self._torques = [
            np.array([p.torque.get(j, 0.0) for j in self._joints]) for p in perturbations
        ]
        self.torque = np.zeros(len(self._joints))

    def update(self, t: float) -> None:
        for i, f in enumerate(self.faults):
            on = _active(f, t)
            if on != self.state[i]:
                self.state[i] = on
                self._apply(f, on)
        if self.perturbations:
            tau = sum(
                (tq for p, tq in zip(self.perturbations, self._torques) if _active(p, t)),
                np.zeros(len(self._joints)),
            )
            if not np.array_equal(tau, self.torque):
                self.torque = tau
                self.ep.plant.set_external_torque(tau)

    def _apply(self, f: Fault, on: bool) -> None:
        ep = self.ep
        if isinstance(f, ElectrodeDetach):
            if ep.closed_loop:
                ep.stim.channel_gain[self._stim.index(f.channel)] = 0.0 if on else 1.0
        elif isinstance(f, EMGDropout | EMGSaturation):
            j = ep.emg.names.index(f.channel)
            if on:
                ep.emg.channel_fault[j] = "dropout" if isinstance(f, EMGDropout) else "saturation"
            else:
                ep.emg.channel_fault.pop(j, None)
        elif isinstance(f, AngleFreeze):
            ep.angle.frozen[self._joints.index(f.joint)] = on
        elif isinstance(f, ArtifactIncrease):
            ep.emg.artifact_gain = f.factor if on else 1.0


class _Log:
    """Preallocated controller-rate columns; unset values stay NaN."""

    def __init__(self, n: int):
        self.n, self.cols = n, {}

    def put(self, k: int, prefix: str, names: list[str], values: np.ndarray) -> None:
        for name, v in zip(names, values, strict=True):
            key = f"{prefix}_{name}"
            if key not in self.cols:
                self.cols[key] = np.full(self.n, np.nan)
            self.cols[key][k] = v


class _Episode:
    def __init__(
        self,
        cfg: SimConfig,
        seed: int,
        prefix: str,
        q0: np.ndarray,
        qd0: np.ndarray,
        lock: bool,
        mvc: MVCValues | None,
    ):
        self.cfg = cfg
        t = cfg.timing
        self.plant = Plant(cfg.plant, t)
        self.plant.reset(q0, qd0)
        if lock:
            self.plant.lock(q0)
        dt = self.plant.dt
        self.patient = Patient(
            cfg.patient, cfg.plant, self.plant.action, dt, make_rng(seed, prefix + "patient")
        )
        self.emg = EMGSensor(
            cfg.emg,
            cfg.plant.muscles,
            cfg.plant.joints,
            [ch.name for ch in cfg.stim.channels],
            float(t.emg_hz),
            lambda name: make_rng(seed, f"{prefix}emg/{name}"),
        )
        self.angle = AngleSensor(
            cfg.angle_sensor,
            self.plant.n_joints,
            dt,
            1.0 / t.angle_sensor_hz,
            make_rng(seed, prefix + "angle_sensor"),
        )
        self.angle.reset(q0)
        self.q_meas = np.asarray(q0, dtype=np.float64).copy()
        self.ctrl_cfg = build_controller_config(cfg)
        self.closed_loop = mvc is not None
        if self.closed_loop:
            self.controller = ControllerPipeline(self.ctrl_cfg, mvc)
            self.controller.reset(q0)
            self.supervisor = SafetySupervisor(build_safety_config(cfg))
            self.stim = StimModel(cfg.stim, cfg.plant.muscles, dt)
            # Output computed at tick n reaches the supervisor at tick n + latency_ticks.
            # Before the first real output arrives the stimulator sees a zero command.
            n_lat = cfg.controller.latency_ticks
            self._queue = deque([(0.0, np.zeros(len(cfg.stim.channels)))] * n_lat)
        else:
            self.front_end = default_front_end(self.ctrl_cfg)
        self.events: list[SafetyEvent] = []

    def _pulse(self, now: float, applied: np.ndarray, offset: int) -> None:
        """Stimulator fires: latch recruitment; EMG sees the artifact and M-wave starting
        `offset` samples into its next chunk; the controller gets the sync signal."""
        ev = self.stim.pulse(now, applied)
        if ev.active:
            self.emg.add_pulse(offset, ev)
            self.controller.on_stim_pulse(now)

    def run(
        self,
        duration_s: float,
        intent_at: Callable[[float], Intent],
        on_step: StepHook | None,
        faults: list[Fault] | None = None,
        perturbations: list[Perturbation] | None = None,
    ) -> pd.DataFrame:
        cfg, plant = self.cfg, self.plant
        inject = _Injector(self, faults or [], perturbations or [])
        fault_names = [f"{i}_{f.kind}" for i, f in enumerate(inject.faults)]
        t = cfg.timing
        sched = Scheduler(t)
        n_steps = round(duration_s * t.physics_hz)
        per_tick = t.physics_hz // t.controller_hz
        log = _Log(n_steps // per_tick)
        joints, muscles = cfg.plant.joints, cfg.plant.muscles
        emg_names = self.emg.names
        stim_names = [ch.name for ch in cfg.stim.channels]
        n_stim = len(stim_names)
        applied = np.zeros(n_stim)
        u_buf = np.zeros((per_tick, len(muscles)))
        qd_buf = np.zeros((per_tick, len(joints)))
        u_stim = np.zeros(len(muscles))
        groups = np.empty(log.n, dtype=object)
        k = i = 0
        for _ in range(n_steps):
            if inject.faults or inject.perturbations:
                inject.update(sched.t)
            intent = intent_at(sched.t)
            q, qd = plant.joint_state()
            u_vol = self.patient.step(intent, q, qd)
            u_buf[i] = u_vol
            i += 1
            if self.closed_loop:
                u_stim = self.stim.step()
                plant.step(self.stim.combine(u_vol, u_stim))
            else:
                plant.step(u_vol)
            if on_step is not None:
                on_step(plant)
            ticks = sched.advance()
            now = sched.t
            q, qd = plant.joint_state()
            qd_buf[i - 1] = qd
            self.angle.push(q)
            if ticks.angle_sensor:
                self.q_meas = self.angle.sample()
            if not ticks.controller:
                if ticks.stim and self.closed_loop:
                    self._pulse(now, applied, offset=i)
                continue

            emg_chunk = self.emg.chunk(u_buf[:i], qd_buf[:i])
            i = 0
            if self.closed_loop:
                out = self.controller.step(now, emg_chunk, self.q_meas)
                self._queue.append((out.t, out.intensity))
                cmd_t, cmd = self._queue.popleft()
                safe = self.supervisor.step(now, cmd, cmd_t, self.q_meas, emg_chunk)
                applied = safe.intensity
                env, norm = out.envelope, out.normalized
                log.put(k, "intent", joints, out.intent)
                log.put(k, "ref_v", joints, out.ref_velocity)
                log.put(k, "ref", joints, out.ref_angle)
                log.put(k, "error", joints, out.error)
                log.put(k, "pd", joints, out.pd_output)
                log.put(k, "norm", emg_names, norm)
                log.put(k, "cmd", stim_names, out.intensity)
                log.put(k, "stim", stim_names, applied)
                log.put(k, "safety", list(RULES), [f.any() for f in safe.fired.values()])
                log.put(k, "cap", muscles, self.stim.capacity)
            else:
                env = self.front_end.process(now, emg_chunk)[-1]
            if ticks.stim and self.closed_loop:
                self._pulse(now, applied, offset=0)

            target = np.full(len(joints), np.nan) if intent.target is None else intent.target
            log.put(k, "t", ["s"], [now])
            log.put(k, "target", joints, target)
            log.put(k, "q", joints, q)
            log.put(k, "qd", joints, qd)
            log.put(k, "q_meas", joints, self.q_meas)
            log.put(k, "torque", joints, plant.actuator_torque())
            log.put(k, "env", emg_names, env)
            log.put(k, "u_vol", muscles, u_vol)
            log.put(k, "u_stim", muscles, u_stim)
            log.put(k, "act", muscles, plant.muscle_state().activation)
            if fault_names:
                log.put(k, "fault", fault_names, inject.state)
            if inject.perturbations:
                log.put(k, "perturb", joints, inject.torque)
            groups[k] = intent.mvc_group or ""
            k += 1

        if self.closed_loop:
            self.events = list(self.supervisor.events)
        cols = log.cols
        df = pd.DataFrame({"t": cols.pop("t_s"), **cols})
        df["mvc_group"] = groups
        return df
