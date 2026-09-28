"""Pydantic schemas for the sim. Every parameter lives in YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimingConfig(_Strict):
    physics_hz: int = Field(gt=0)
    emg_hz: int = Field(gt=0)
    controller_hz: int = Field(gt=0)
    stim_hz: int = Field(gt=0)
    angle_sensor_hz: int = Field(gt=0)


class ExoPayloadConfig(_Strict):
    mass_kg: float = Field(ge=0.0)
    pos_m: tuple[float, float, float]


class MuscleGroupConfig(_Strict):
    """Muscles acting on one joint in one direction. sign: +1 flexion, -1 extension."""

    joint: str
    sign: Literal[1, -1]
    muscles: list[str] = Field(min_length=1)


class PlantConfig(_Strict):
    model_file: Path | None
    model_relpath: str
    joints: list[str] = Field(min_length=1)
    muscles: list[str] = Field(min_length=1)
    muscle_groups: dict[str, MuscleGroupConfig]
    forearm_body: str
    gravity: bool
    exo_payload: ExoPayloadConfig

    @model_validator(mode="after")
    def _groups_consistent(self) -> PlantConfig:
        seen: dict[str, str] = {}
        for name, g in self.muscle_groups.items():
            if g.joint not in self.joints:
                raise ValueError(f"muscle group '{name}': unknown joint '{g.joint}'")
            for m in g.muscles:
                if m not in self.muscles:
                    raise ValueError(f"muscle group '{name}': unknown muscle '{m}'")
                if m in seen:
                    raise ValueError(f"muscle '{m}' in groups '{seen[m]}' and '{name}'")
                seen[m] = name
        return self


class DriveConfig(_Strict):
    """Intent-to-excitation mapping: intended velocity toward target, split by muscle action."""

    approach_tau_s: float = Field(gt=0)  # intended velocity = error / tau (then capped)
    max_speed_rad_s: float = Field(gt=0)
    gain: float = Field(ge=0)  # excitation per rad/s of (intended - actual) velocity


class PatientConfig(_Strict):
    name: str
    strength: dict[str, float]  # per muscle, 0 = paralysed, 1 = healthy
    onset_delay_ms: float = Field(ge=0)
    cocontraction: float = Field(ge=0, le=1)
    noise_std: float = Field(ge=0)  # motor noise on excitation, per physics step
    drive: DriveConfig

    @model_validator(mode="after")
    def _strength_range(self) -> PatientConfig:
        bad = {m: s for m, s in self.strength.items() if not 0.0 <= s <= 1.0}
        if bad:
            raise ValueError(f"strength must be in [0, 1]: {bad}")
        return self


class HoldSegment(_Strict):
    kind: Literal["hold"]
    duration_s: float = Field(gt=0)


class StepSegment(_Strict):
    kind: Literal["step"]
    duration_s: float = Field(gt=0)
    q: list[float]


class RampSegment(_Strict):
    kind: Literal["ramp"]
    duration_s: float = Field(gt=0)
    q: list[float]


class SineSegment(_Strict):
    kind: Literal["sine"]
    duration_s: float = Field(gt=0)
    center: list[float]
    amplitude: list[float]
    freq_hz: float = Field(gt=0)


Segment = Annotated[
    HoldSegment | StepSegment | RampSegment | SineSegment, Field(discriminator="kind")
]


class InitialState(_Strict):
    q: list[float]
    qd: list[float]


class CalibratedValues(_Strict):
    """Pinned calibration results, to skip the MVC trial (e.g. from a previous run's meta)."""

    envelope: dict[str, float]  # EMG channel -> MVC envelope
    rest: dict[str, float]  # EMG channel -> resting envelope
    deadband: dict[str, float]  # joint -> intent deadband


class MovementScenario(_Strict):
    kind: Literal["movement"]
    name: str
    initial_state: InitialState
    target: list[Segment] | None  # None -> patient intends nothing (no_intent)
    duration_s: float = Field(gt=0)
    closed_loop: bool  # false: volitional only, no controller or stim
    calibrated: CalibratedValues | None = None  # None -> run the calibration trial first
    stim_off_baseline: bool = False  # also run stim-off (open loop) for baseline metrics


class EMGChannelConfig(_Strict):
    name: str
    mvc_group: str  # muscle group whose MVC trial calibrates this channel
    weights: dict[str, float]  # muscle -> pickup weight (row of S[J, M]); crosstalk < 1


class VolitionalEMGConfig(_Strict):
    enabled: bool
    band_hz: tuple[float, float]  # bandlimited-noise carrier
    order: int = Field(ge=1)
    amplitude_mv: dict[str, float]  # per muscle: carrier RMS at full excitation


class WhiteNoiseConfig(_Strict):
    enabled: bool
    std_mv: float = Field(ge=0)


class StimArtifactConfig(_Strict):
    """Spike on every pulse: amplitude_mv[emg][stim] * intensity, exponential decay."""

    enabled: bool
    amplitude_mv: dict[str, dict[str, float]]  # EMG channel -> stim channel -> mV at intensity 1
    decay_ms: float = Field(gt=0)


class MWaveConfig(_Strict):
    """Evoked compound response per pulse: one sine cycle after `latency_ms`, peak
    amplitude_mv[muscle] * recruitment, seen through the sensor matrix S."""

    enabled: bool
    amplitude_mv: dict[str, float]  # per muscle, peak at full recruitment
    latency_ms: float = Field(ge=0)
    duration_ms: float = Field(gt=0)


class PowerlineConfig(_Strict):
    enabled: bool
    freq_hz: float = Field(gt=0)  # 50 Hz mains (Greece)
    amplitude_mv: list[float] = Field(min_length=1)  # fundamental, then harmonics


class BaselineWanderConfig(_Strict):
    enabled: bool
    std_mv: float = Field(ge=0)
    cutoff_hz: float = Field(gt=0)


class MotionArtifactConfig(_Strict):
    enabled: bool
    gain_mv_per_rad_s: dict[str, dict[str, float]]  # EMG channel -> joint -> mV per rad/s


class EMGConfig(_Strict):
    channels: list[EMGChannelConfig] = Field(min_length=1)
    volitional: VolitionalEMGConfig
    white_noise: WhiteNoiseConfig
    stim_artifact: StimArtifactConfig
    m_wave: MWaveConfig
    powerline: PowerlineConfig
    baseline_wander: BaselineWanderConfig
    motion: MotionArtifactConfig
    saturation_mv: float | None = Field(gt=0)  # amplifier clipping; None -> no clipping


class AngleSensorConfig(_Strict):
    noise_std_rad: float = Field(ge=0)
    bias_walk_std_rad_per_sqrt_s: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    resolution_rad: float = Field(ge=0)  # 0 -> no quantization


class RecruitmentConfig(_Strict):
    """Sigmoid intensity -> recruitment fraction, shifted so zero intensity recruits nothing."""

    threshold: float = Field(ge=0, le=1)  # intensity at the sigmoid midpoint
    slope: float = Field(gt=0)  # per unit intensity
    saturation: float = Field(gt=0, le=1)  # recruitment at full intensity


class StimChannelConfig(_Strict):
    name: str
    joint: str
    sign: Literal[1, -1]  # direction this electrode drives the joint (+1 flexion)
    muscles: dict[str, float]  # muscle -> recruitment weight (column of E[M, K]); crosstalk < 1


class StimConfig(_Strict):
    n_steps: int = Field(ge=2)  # digital-pot quantization levels
    recruitment: RecruitmentConfig
    em_delay_ms: float = Field(ge=0)
    combine: Literal["probabilistic_sum"]
    channels: list[StimChannelConfig] = Field(min_length=1)


class JointControlConfig(_Strict):
    agonist: str  # EMG channel name
    antagonist: str
    gain: float = Field(ge=0)  # rad/s^2 per unit intent
    damping: float = Field(ge=0)  # 1/s; 0 = pure double integration
    rom: tuple[float, float]  # reference clamp
    kp: float = Field(ge=0)
    kd: float = Field(ge=0)


class AllocationConfig(_Strict):
    offset: float = Field(ge=0, lt=1)  # 0 = plain split; > 0 = recruitment-threshold offset
    epsilon: float = Field(ge=0)  # the offset applies only above this PD output

    @model_validator(mode="after")
    def _gated(self) -> AllocationConfig:
        if self.offset > 0 and self.epsilon <= 0:
            raise ValueError("allocation.offset > 0 needs allocation.epsilon > 0")
        return self


class ControllerSection(_Strict):
    bandpass_hz: tuple[float, float]
    bandpass_order: int = Field(ge=1)
    envelope_hz: float = Field(gt=0)
    envelope_order: int = Field(ge=1)
    blanking_ms: float = Field(ge=0)  # 0 -> no blanking stage
    blanking_fill: Literal["hold", "zero"]
    blanking_correction: bool  # must be true with zero fill, false with hold
    deadband_floor: float = Field(ge=0, lt=1)  # lower bound for the calibrated deadband
    deadband_k: float = Field(ge=0)  # deadband = max(floor, k * sigma of resting intent)
    allocation: AllocationConfig
    derivative_tau_s: float = Field(ge=0)
    latency_ticks: int = Field(ge=0)  # compute latency before output takes effect
    joints: dict[str, JointControlConfig]


class SafetySection(_Strict):
    cap: dict[str, float]  # stim channel -> max intensity
    max_rise_per_tick: float = Field(gt=0)
    joint_limits: dict[str, tuple[float, float]]  # ROM guard limits
    rom_margin_rad: float = Field(ge=0)
    dose_window_s: float = Field(gt=0)
    dose_max_mean: float = Field(gt=0, le=1)
    watchdog_timeout_s: float = Field(gt=0)
    q_plausible: dict[str, tuple[float, float]]
    emg_plausible_abs_mv: float = Field(gt=0)


class MVCTrial(_Strict):
    group: str
    effort_s: float = Field(gt=0)


class CalibrationConfig(_Strict):
    """MVC protocol: isometric max effort per muscle group, joint locked at lock_q."""

    lock_q: list[float]
    rest_s: float = Field(ge=0)
    trials: list[MVCTrial] = Field(min_length=1)
    window_s: float = Field(gt=0)  # MVC value = mean over the best window within each effort

    @model_validator(mode="after")
    def _window_fits(self) -> CalibrationConfig:
        # The rest baseline is the last window_s of each rest; the first half of the rest
        # is left for the previous effort's EMG and envelope to die away.
        if self.rest_s < 2 * self.window_s:
            raise ValueError(f"rest_s {self.rest_s} must be >= 2 * window_s {self.window_s}")
        for tr in self.trials:
            if self.window_s > tr.effort_s:
                raise ValueError(f"window_s {self.window_s} > effort_s {tr.effort_s}")
        return self


class MVCScenario(_Strict):
    """Runs the `calibration` protocol on its own."""

    kind: Literal["mvc"]
    name: str


Scenario = Annotated[MovementScenario | MVCScenario, Field(discriminator="kind")]


class PassiveDropConfig(_Strict):
    duration_s: float = Field(gt=0)
    q0_rad: float


class MetricsConfig(_Strict):
    settle_s: float = Field(ge=0)  # excluded from tracking RMSE after each target step
    step_jump_rad: float = Field(gt=0)  # target change per log sample that counts as a step


class SimConfig(_Strict):
    timing: TimingConfig
    plant: PlantConfig
    passive_drop: PassiveDropConfig | None = None
    metrics: MetricsConfig | None = None
    calibration: CalibrationConfig | None = None
    emg: EMGConfig | None = None
    angle_sensor: AngleSensorConfig | None = None
    stim: StimConfig | None = None
    controller: ControllerSection | None = None
    safety: SafetySection | None = None
    patient: PatientConfig | None = None
    scenario: Scenario | None = None

    @model_validator(mode="after")
    def _rates_divide_physics(self) -> SimConfig:
        # Fail at config load, not mid-run. Scheduler re-validates independently.
        from heroes_sim.scheduler import validate_rates

        validate_rates(self.timing)
        return self

    @model_validator(mode="after")
    def _dims_match(self) -> SimConfig:
        n = len(self.plant.joints)
        if self.patient is not None and set(self.patient.strength) != set(self.plant.muscles):
            raise ValueError(
                f"patient.strength muscles {sorted(self.patient.strength)} "
                f"!= plant.muscles {sorted(self.plant.muscles)}"
            )
        sc = self.scenario
        if isinstance(sc, MovementScenario):
            vecs = [sc.initial_state.q, sc.initial_state.qd]
            for seg in sc.target or []:
                vecs += [getattr(seg, f) for f in ("q", "center", "amplitude") if hasattr(seg, f)]
            if any(len(v) != n for v in vecs):
                raise ValueError(f"scenario joint vectors must have length {n}")
        if self.calibration is not None:
            if len(self.calibration.lock_q) != n:
                raise ValueError(f"calibration.lock_q must have length {n}")
            for tr in self.calibration.trials:
                if tr.group not in self.plant.muscle_groups:
                    raise ValueError(f"MVC trial group '{tr.group}' not in plant.muscle_groups")
        return self

    @model_validator(mode="after")
    def _names_match(self) -> SimConfig:
        """Every name used by emg/stim/controller/safety must exist where it points."""
        joints, muscles = set(self.plant.joints), set(self.plant.muscles)

        def need(names, universe, where: str) -> None:
            missing = set(names) - set(universe)
            if missing:
                raise ValueError(f"{where}: unknown {sorted(missing)}")

        emg_names: list[str] = []
        if self.emg is not None:
            if self.timing.emg_hz != self.timing.physics_hz:
                raise ValueError("timing.emg_hz must equal physics_hz (1 sample per step)")
            emg_names = [c.name for c in self.emg.channels]
            for c in self.emg.channels:
                need(c.weights, muscles, f"emg channel '{c.name}' weights")
                need([c.mvc_group], self.plant.muscle_groups, f"emg channel '{c.name}' mvc_group")
            need(self.emg.volitional.amplitude_mv, muscles, "emg.volitional.amplitude_mv")
            need(self.emg.m_wave.amplitude_mv, muscles, "emg.m_wave.amplitude_mv")
            need(self.emg.stim_artifact.amplitude_mv, emg_names, "emg.stim_artifact channels")
            need(self.emg.motion.gain_mv_per_rad_s, emg_names, "emg.motion channels")
            for c, per_joint in self.emg.motion.gain_mv_per_rad_s.items():
                need(per_joint, joints, f"emg.motion.{c}")
        stim_names: list[str] = []
        if self.stim is not None:
            stim_names = [c.name for c in self.stim.channels]
            if self.emg is not None:
                for c, per_stim in self.emg.stim_artifact.amplitude_mv.items():
                    need(per_stim, stim_names, f"emg.stim_artifact.{c}")
            for c in self.stim.channels:
                need([c.joint], joints, f"stim channel '{c.name}' joint")
                need(c.muscles, muscles, f"stim channel '{c.name}' muscles")
        if self.controller is not None:
            if set(self.controller.joints) != joints:
                raise ValueError("controller.joints must list every plant joint")
            for j, jc in self.controller.joints.items():
                need([jc.agonist, jc.antagonist], emg_names, f"controller.joints.{j} EMG channel")
        if self.safety is not None:
            if set(self.safety.cap) != set(stim_names):
                raise ValueError("safety.cap must list every stim channel")
            if set(self.safety.joint_limits) != joints or set(self.safety.q_plausible) != joints:
                raise ValueError("safety.joint_limits and safety.q_plausible must list every joint")
        sc = self.scenario
        if isinstance(sc, MovementScenario) and sc.calibrated is not None:
            cal = sc.calibrated
            if set(cal.envelope) != set(emg_names) or set(cal.rest) != set(emg_names):
                raise ValueError(
                    "scenario.calibrated must give envelope/rest for every EMG channel"
                )
            if set(cal.deadband) != joints:
                raise ValueError("scenario.calibrated.deadband must give every joint")
        return self


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any] | None) -> None:
    for key, value in (overrides or {}).items():
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SimConfig:
    raw = _read_yaml(path)
    _apply_overrides(raw, overrides)
    return SimConfig.model_validate(raw)


def load_run_config(
    scenario_path: str | Path,
    base_path: str | Path = "configs/base.yaml",
    overrides: dict[str, Any] | None = None,
    patient_path: str | Path | None = None,
) -> SimConfig:
    """Compose base + scenario + a patient profile.

    The patient is `patient_path` if given, else the scenario's `patient_file` resolved
    relative to the scenario file. Overrides use dotted keys into the composed tree,
    e.g. `patient.onset_delay_ms=0`.
    """
    scenario_path = Path(scenario_path)
    raw = _read_yaml(base_path)
    scenario = _read_yaml(scenario_path)
    patient_file = scenario.pop("patient_file")
    raw["scenario"] = scenario
    raw["patient"] = _read_yaml(patient_path or scenario_path.parent / patient_file)
    _apply_overrides(raw, overrides)
    return SimConfig.model_validate(raw)
