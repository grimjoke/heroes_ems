import numpy as np
import pytest

from heroes_control import ControllerPipeline, MVCValues
from heroes_control.blanking import StimBlanking

FS = 2000.0


def feed(stage, x, size, pulses):
    """Stream x in chunks; register each pulse before the chunk that contains its effect."""
    out, pending = [], sorted(pulses)
    for i in range(0, len(x), size):
        t_end = (i + len(x[i : i + size]) - 1) / FS
        while pending and pending[0] < t_end:
            stage.on_pulse(pending.pop(0))
        out.append(stage.process(t_end, x[i : i + size]))
    return np.concatenate(out)


def signal(n=400):
    t = np.arange(n) / FS
    return np.stack([np.sin(2 * np.pi * 7 * t) + 1.0, np.cos(2 * np.pi * 3 * t)], axis=1)


def test_passthrough_without_pulses():
    x = signal()
    np.testing.assert_array_equal(feed(StimBlanking(FS, 0.02, 2), x, 20, []), x)


@pytest.mark.parametrize("fill", ["hold", "zero"])
def test_blanks_window_after_pulse(fill):
    x = signal()
    p = 49 / FS  # pulse at sample 49; samples 50..89 are within 20 ms
    y = feed(StimBlanking(FS, 0.02, 2, fill), x, 20, [p])
    np.testing.assert_array_equal(y[:50], x[:50])
    np.testing.assert_array_equal(y[90:], x[90:])
    expected = x[49] if fill == "hold" else np.zeros(2)
    np.testing.assert_array_equal(y[50:90], np.broadcast_to(expected, (40, 2)))


@pytest.mark.parametrize("fill", ["hold", "zero"])
def test_streaming_equivalence(fill):
    x = signal(1000)
    pulses = [49 / FS, 129 / FS, 209 / FS, 610 / FS]
    outs = [feed(StimBlanking(FS, 0.02, 2, fill), x, size, pulses) for size in (1, 7, 20)]
    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[0], outs[2])


def test_hold_keeps_offset_continuous():
    """Zero-fill cuts a notch out of a DC offset; hold does not (the M3 finding)."""
    x = np.full((200, 1), 0.05)
    held = feed(StimBlanking(FS, 0.02, 1, "hold"), x, 20, [49 / FS])
    zeroed = feed(StimBlanking(FS, 0.02, 1, "zero"), x, 20, [49 / FS])
    assert np.ptp(held) == 0.0 and np.ptp(zeroed) == pytest.approx(0.05)


def test_bad_fill_rejected():
    with pytest.raises(ValueError):
        StimBlanking(FS, 0.02, 2, "interpolate")


def test_pipeline_forwards_stim_sync():
    from tests.test_controller import config

    ctrl = ControllerPipeline(config(blanking_s=0.02), MVCValues((0.5, 0.4)))
    ctrl.reset(0.5)
    ctrl.on_stim_pulse(0.0)
    out = ctrl.step(0.01, np.full((20, 2), 5.0), 0.5)  # samples 1..20 after the pulse
    np.testing.assert_array_equal(out.envelope, 0.0)
