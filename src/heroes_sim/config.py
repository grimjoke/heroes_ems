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


class MovementScenario(_Strict):
    kind: Literal["movement"]
    name: str
    initial_state: InitialState
    target: list[Segment] | None  # None -> patient intends nothing (no_intent)
    duration_s: float = Field(gt=0)


class MVCTrial(_Strict):
    group: str
    effort_s: float = Field(gt=0)


class MVCScenario(_Strict):
    kind: Literal["mvc"]
    name: str
    lock_q: list[float]
    rest_s: float = Field(ge=0)
    trials: list[MVCTrial] = Field(min_length=1)
    window_s: float = Field(gt=0)  # MVC value = mean over the best window within each effort

    @model_validator(mode="after")
    def _window_fits(self) -> MVCScenario:
        for tr in self.trials:
            if self.window_s > tr.effort_s:
                raise ValueError(f"window_s {self.window_s} > effort_s {tr.effort_s}")
        return self


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
        elif isinstance(sc, MVCScenario):
            if len(sc.lock_q) != n:
                raise ValueError(f"scenario.lock_q must have length {n}")
            for tr in sc.trials:
                if tr.group not in self.plant.muscle_groups:
                    raise ValueError(f"MVC trial group '{tr.group}' not in plant.muscle_groups")
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
