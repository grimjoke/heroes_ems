"""EMG-driven EMS controller. numpy/scipy only; shared by the sim and the hardware node."""

from heroes_control.filters import EMGFrontEnd
from heroes_control.normalization import MVCValues
from heroes_control.pipeline import (
    ControllerConfig,
    ControllerOutput,
    ControllerPipeline,
    default_front_end,
)

__all__ = [
    "ControllerConfig",
    "ControllerOutput",
    "ControllerPipeline",
    "EMGFrontEnd",
    "MVCValues",
    "default_front_end",
]
