import numpy as np
import pytest

from heroes_control import ControllerPipeline, MVCValues, default_front_end
from heroes_control.blanking import BlankingCorrection, StimBlanking

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


def _envelope(fill, correction, blank=True, seconds=6.0):
    """Mean envelope of steady unit-RMS EMG with a 25 Hz pulse train, after 1 s settling.

    Built by hand: the config (rightly) refuses zero fill without correction."""
    from heroes_control import EMGFrontEnd
    from tests.test_controller import config

    stages = default_front_end(config()).stages  # bandpass, rectify, envelope
    if blank:
        b = StimBlanking(FS, 0.015, 2, fill)
        stages = [b, *stages]
        if correction:
            stages.append(BlankingCorrection(b, stages[-1].sos))
    fe = EMGFrontEnd(stages)
    x = np.random.default_rng(1).standard_normal((int(seconds * FS), 2))
    out, pulse_every = [], int(FS / 25)
    for i in range(0, len(x), 20):
        t = (i + 19) / FS
        if blank:
            for p in range(i, i + 20):
                if p % pulse_every == 0:
                    for st in fe.stages:
                        if hasattr(st, "on_pulse"):
                            st.on_pulse(p / FS)
        out.append(fe.process(t, x[i : i + 20]))
    return np.concatenate(out)[int(FS) :].mean(axis=0)


def test_zero_fill_bias_and_correction():
    """D3: zero fill reads low (~0.70 of the unblanked envelope with a 15 ms window at
    25 Hz); the kept-fraction correction brings it back but overshoots (~1.11), because
    the bandpass smears the gaps, so less is lost than the 37.5% blanked. Hold reads
    ~0.94 with no correction. Characterized here; see README."""
    ref = _envelope("hold", False, blank=False)
    zeroed = _envelope("zero", False) / ref
    corrected = _envelope("zero", True) / ref
    held = _envelope("hold", False) / ref
    assert np.all(zeroed < 0.8)
    assert np.all((corrected > 1.0) & (corrected < 1.2))
    assert np.all(np.abs(corrected - 1) < np.abs(zeroed - 1))
    assert np.all((held > 0.9) & (held < 1.0))


def test_correction_starts_at_steady_state():
    unity_dc = np.array([[0.5, 0, 0, 1, -0.5, 0]])  # y = 0.5 x + 0.5 y[-1], DC gain 1
    corr = BlankingCorrection(StimBlanking(FS, 0.015, 2), unity_dc)
    np.testing.assert_allclose(corr.process(0.0, np.ones((5, 2))), 1.0)
