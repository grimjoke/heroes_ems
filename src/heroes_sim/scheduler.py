"""Multi-rate clock driven by physics steps.

Sampled clocks (EMG, controller, angle sensor) must divide the physics rate exactly:
their samples are evenly spaced by construction. Event clocks (stim pulses) may run at
any rate up to the physics rate: pulse k fires at the physics step nearest k / rate, so
timing jitter is at most half a physics step (0.25 ms at 2 kHz), negligible against
muscle activation (~10 ms).
"""

from __future__ import annotations

from dataclasses import dataclass

from heroes_sim.config import TimingConfig

SAMPLED = ("emg_hz", "controller_hz", "angle_sensor_hz")
EVENTS = ("stim_hz",)


def validate_rates(timing: TimingConfig) -> dict[str, int]:
    """Return {sampled clock: steps-per-tick}; raise if a sampled clock doesn't divide the
    physics rate or an event clock exceeds it."""
    ratios: dict[str, int] = {}
    for name in SAMPLED:
        hz = getattr(timing, name)
        if hz > timing.physics_hz or timing.physics_hz % hz != 0:
            raise ValueError(
                f"timing.{name}={hz} does not evenly divide physics_hz={timing.physics_hz}"
            )
        ratios[name] = timing.physics_hz // hz
    for name in EVENTS:
        if getattr(timing, name) > timing.physics_hz:
            raise ValueError(f"timing.{name} exceeds physics_hz={timing.physics_hz}")
    return ratios


def event_step(k: int, event_hz: float, physics_hz: int) -> int:
    """Physics step at which event k fires (nearest to k / event_hz, k >= 1)."""
    return round(k * physics_hz / event_hz)


@dataclass(frozen=True)
class Ticks:
    emg: bool
    controller: bool
    stim: bool
    angle_sensor: bool


class Scheduler:
    def __init__(self, timing: TimingConfig):
        self._ratios = validate_rates(timing)
        self._physics_hz = timing.physics_hz
        self._stim_hz = timing.stim_hz
        self.dt = 1.0 / timing.physics_hz
        self.step_index = 0
        self._pulse_k = 1
        self._next_pulse = event_step(1, self._stim_hz, self._physics_hz)

    @property
    def t(self) -> float:
        return self.step_index * self.dt

    def advance(self) -> Ticks:
        """Advance one physics step; report which clocks tick at the new step index."""
        self.step_index += 1
        n = self.step_index
        r = self._ratios
        stim = n == self._next_pulse
        if stim:
            self._pulse_k += 1
            self._next_pulse = event_step(self._pulse_k, self._stim_hz, self._physics_hz)
        return Ticks(
            emg=n % r["emg_hz"] == 0,
            controller=n % r["controller_hz"] == 0,
            stim=stim,
            angle_sensor=n % r["angle_sensor_hz"] == 0,
        )
