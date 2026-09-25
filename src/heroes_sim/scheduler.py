"""Integer-ratio multi-rate clock driven by physics steps."""

from __future__ import annotations

from dataclasses import dataclass

from heroes_sim.config import TimingConfig


def validate_rates(timing: TimingConfig) -> dict[str, int]:
    """Return {clock: steps-per-tick}; raise if any rate doesn't divide the physics rate."""
    ratios: dict[str, int] = {}
    for name in ("emg_hz", "controller_hz", "stim_hz", "angle_sensor_hz"):
        hz = getattr(timing, name)
        if hz > timing.physics_hz or timing.physics_hz % hz != 0:
            raise ValueError(
                f"timing.{name}={hz} does not evenly divide physics_hz={timing.physics_hz}"
            )
        ratios[name] = timing.physics_hz // hz
    return ratios


@dataclass(frozen=True)
class Ticks:
    emg: bool
    controller: bool
    stim: bool
    angle_sensor: bool


class Scheduler:
    def __init__(self, timing: TimingConfig):
        self._ratios = validate_rates(timing)
        self.dt = 1.0 / timing.physics_hz
        self.step_index = 0

    @property
    def t(self) -> float:
        return self.step_index * self.dt

    def advance(self) -> Ticks:
        """Advance one physics step; report which clocks tick at the new step index."""
        self.step_index += 1
        n = self.step_index
        r = self._ratios
        return Ticks(
            emg=n % r["emg_hz"] == 0,
            controller=n % r["controller_hz"] == 0,
            stim=n % r["stim_hz"] == 0,
            angle_sensor=n % r["angle_sensor_hz"] == 0,
        )
