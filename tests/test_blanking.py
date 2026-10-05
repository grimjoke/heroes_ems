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
    np.testing.assert_array_equal(feed(StimBlanking(FS, 0.02, 2, "hold"), x, 20, []), x)


@pytest.mark.parametrize("fill", ["hold", "zero"])
def test_blanks_window_after_pulse(fill):
    x = signal()
    p = 49 / FS  # pulse at sample 49; samples 50..89 are within 20 ms
    y = feed(StimBlanking(FS, 0.02, 2, fill), x, 20, [p])
    np.testing.assert_array_equal(y[:50], x[:50])
    np.testing.assert_array_equal(y[90:], x[90:])
    expected = x[49] if fill == "hold" else np.zeros(2)
    np.testing.assert_array_equal(y[50:90], np.broadcast_to(expected, (40, 2)))


@pytest.mark.parametrize("fill", ["interp", "hold", "zero"])
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
        StimBlanking(FS, 0.02, 2, "spline")


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
    corr = BlankingCorrection(StimBlanking(FS, 0.015, 2, "zero"), unity_dc)
    np.testing.assert_allclose(corr.process(0.0, np.ones((5, 2))), 1.0)


# ---- D15: interpolating blank -------------------------------------------------------


def test_interp_delays_by_window_plus_one():
    x = signal(400)
    b = StimBlanking(FS, 0.02, 2, "interp")
    assert b.delay == 41
    y = feed(b, x, 20, [])
    np.testing.assert_array_equal(y[:41], 0.0)  # delay line starts empty
    np.testing.assert_array_equal(y[41:], x[: 400 - 41])


def test_interp_draws_a_line_across_the_blank():
    x = signal(400)
    y = feed(StimBlanking(FS, 0.02, 2, "interp"), x, 20, [99 / FS])  # blanks 100..139
    d = 41
    out = y[100 + d : 140 + d]
    w = (np.arange(1, 41) / 41)[:, None]
    np.testing.assert_allclose(out, x[99] + w * (x[140] - x[99]))
    np.testing.assert_array_equal(y[d : 100 + d], x[:100])  # untouched before
    np.testing.assert_array_equal(y[140 + d :], x[140 : 400 - d])  # and after


def test_interp_leaves_no_step_on_a_mains_signal():
    """The hold-end step (held sample vs the live mains) is what leaked on weak channels."""
    t = np.arange(2000) / FS
    mains = 0.02 * np.sin(2 * np.pi * 50 * t)[:, None]
    pulses = [k / 30 for k in range(1, 30)]
    held = feed(StimBlanking(FS, 0.015, 1, "hold"), mains, 20, pulses)
    interp = feed(StimBlanking(FS, 0.015, 1, "interp"), mains, 20, pulses)
    jump = lambda y: np.abs(np.diff(y[:, 0])).max()
    assert jump(interp) < 0.25 * jump(held)


def test_default_front_end_wires_notch_and_correction():
    from heroes_control.blanking import BlankingCorrection as BC
    from heroes_control.filters import CausalSOS
    from tests.test_controller import config

    zero = default_front_end(
        config(blanking_s=0.015, blanking_fill="zero", blanking_correction=True, notch_hz=(50.0,))
    ).stages
    assert isinstance(zero[0], StimBlanking) and isinstance(zero[-1], BC)
    assert isinstance(zero[1], CausalSOS) and zero[1].sos.shape == (1, 6)  # the notch
    interp = default_front_end(config(blanking_s=0.015, blanking_fill="interp")).stages
    assert not any(isinstance(st, BC) for st in interp)


def test_notch_removes_mains():
    from heroes_control.filters import notch

    t = np.arange(8000) / FS
    x = np.stack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 120 * t)], axis=1)
    y = notch(50.0, 30.0, FS, 2).process(0.0, x)
    assert y[4000:, 0].std() < 0.02 * x[:, 0].std()  # 50 Hz gone after settling
    assert y[4000:, 1].std() > 0.95 * x[:, 1].std()  # 120 Hz kept


# ---- D17: adaptive mains canceller ---------------------------------------------------


def _mains_noise_wander(seconds=20.0, seed=0):
    from scipy import signal as sg

    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    mains = sum(
        a * np.sin(2 * np.pi * 50 * (h + 1) * t + rng.uniform(0, 2 * np.pi))
        for h, a in enumerate([0.02, 0.006, 0.003])
    )
    w = sg.sosfilt(sg.butter(2, 0.5, fs=FS, output="sos"), rng.standard_normal(n))
    wander = 0.05 * w / w[n // 10 :].std()
    noise = 0.005 * rng.standard_normal(n)
    return np.stack([noise, noise + mains + wander], axis=1)  # [clean, contaminated]


def _canceller_envelope(x, cancel):
    """Envelope (median, p99) of x with 15 ms interp blanking at 30 Hz, notch after."""
    from tests.test_controller import config

    kw = {"blanking_s": 0.015, "blanking_fill": "interp", "notch_hz": (50.0,)}
    if cancel:
        kw |= {"mains_cancel_hz": (50.0, 100.0, 150.0), "mains_cancel_mu": 0.002}
        kw |= {"mains_cancel_mu_bias": 0.02}
    fe = default_front_end(config(**kw))
    pulses = list(np.arange(0.5, len(x) / FS, 1 / 30))
    env = []
    for i in range(0, len(x), 20):
        t = (i + 19) / FS
        while pulses and pulses[0] <= t:
            p = pulses.pop(0)
            for st in fe.stages:
                if hasattr(st, "on_pulse"):
                    st.on_pulse(p)
        env.append(fe.process(t, x[i : i + 20])[-1])
    env = np.array(env)[200:]  # after 2 s
    return np.median(env, axis=0), np.percentile(env, 99, axis=0)


def test_blanked_mains_leaks_and_canceller_removes_it():
    """Interpolating across 15 ms of 50 Hz leaves a gated residual the notch after
    blanking cannot remove (the envelope reads ~2.6x the noise floor). The canceller, fitted
    on unblanked samples only, takes it back to the floor, median and tail."""
    x = _mains_noise_wander()
    (clean, dirty), _ = _canceller_envelope(x, cancel=False)
    assert dirty > 2 * clean
    (clean_c, dirty_c), (_, dirty_p99) = _canceller_envelope(x, cancel=True)
    assert dirty_c < 1.1 * clean_c
    _, (clean_p99, _) = _canceller_envelope(x, cancel=False)
    assert dirty_p99 < 1.3 * clean_p99


def test_canceller_frozen_during_blanks_and_chunking_independent():
    from heroes_control.blanking import MainsCanceller

    t = np.arange(4000) / FS
    x = (0.02 * np.sin(2 * np.pi * 50 * t))[:, None] * np.ones((1, 2))
    spiky = x.copy()
    pulses = [k / 30 for k in range(1, 60)]
    for p in pulses:  # a big artifact inside each blank
        i = round(p * FS) + 3
        spiky[i : i + 4] += 8.0

    def run(sig, size):
        blank = StimBlanking(FS, 0.015, 2, "interp")
        mc = MainsCanceller((50.0,), FS, 2, 0.002, 0.02, blank)
        out, pending = [], sorted(pulses)
        for i in range(0, len(sig), size):
            t_end = (i + len(sig[i : i + size]) - 1) / FS
            while pending and pending[0] < t_end:
                blank.on_pulse(pending.pop(0))
            out.append(mc.process(t_end, sig[i : i + size]))
            blank.process(t_end, sig[i : i + size])  # pops expired pulses, as in the chain
        return np.concatenate(out), mc

    y, mc = run(spiky, 20)
    _, mc_clean = run(x, 20)
    np.testing.assert_allclose(mc._coef, mc_clean._coef, atol=1e-12)  # artifacts never fitted
    np.testing.assert_array_equal(y, run(spiky, 7)[0])
