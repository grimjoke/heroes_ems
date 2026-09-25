"""Pydantic schemas for the sim. Every parameter lives in YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class PlantConfig(_Strict):
    model_file: Path | None
    model_relpath: str
    joints: list[str] = Field(min_length=1)
    muscles: list[str] = Field(min_length=1)
    forearm_body: str
    gravity: bool
    exo_payload: ExoPayloadConfig


class PassiveDropConfig(_Strict):
    duration_s: float = Field(gt=0)
    q0_rad: float


class SimConfig(_Strict):
    timing: TimingConfig
    plant: PlantConfig
    passive_drop: PassiveDropConfig | None = None

    @model_validator(mode="after")
    def _rates_divide_physics(self) -> SimConfig:
        # Fail at config load, not mid-run. Scheduler re-validates independently.
        from heroes_sim.scheduler import validate_rates

        validate_rates(self.timing)
        return self


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SimConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return SimConfig.model_validate(raw)
