import numpy as np
import pytest

from heroes_control import ControllerConfig, ControllerPipeline, MVCValues, default_front_end
from heroes_control.allocation import SignedAllocation
from heroes_control.intent import AgonistAntagonistIntent, calibrate_deadband
from heroes_control.normalization import MVCNormalizer
from heroes_control.pd import PD
from heroes_control.reference import DoubleIntegratorReference

FS, TICK = 2000.0, 100.0


def config(**kw):
    base = {
        "emg_fs_hz": FS,
        "tick_hz": TICK,
        "n_emg": 2,
        "bandpass_hz": (20.0, 450.0),
        "bandpass_order": 4,
        "envelope_hz": 4.0,
        "envelope_order": 2,
        "blanking_s": 0.0,
        "blanking_fill": "hold",
        "blanking_correction": False,
        "notch_hz": (),
        "notch_q": 30.0,
        "mains_cancel_hz": (),
        "mains_cancel_mu": 0.0,
        "mains_cancel_mu_bias": 0.0,
        "agonist": (0,),
        "antagonist": (1,),
        "deadband_floor": 0.05,
        "deadband_percentile": 99.0,
        "gain": (50.0,),
        "damping": (10.0,),
        "q_min": (0.1,),
        "q_max": (2.1,),
        "kp": (0.8,),
        "kd": (0.03,),
        "derivative_tau_s": 0.02,
        "channel_joint": (0, 0),
        "channel_sign": (1, -1),
        "allocation_offset": 0.0,
        "allocation_epsilon": 0.0,
    }
    return ControllerConfig(**{**base, **kw})


MVC = MVCValues((0.5, 0.4), (0.01, 0.01))


def emg_signal(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    amp = np.stack([np.linspace(0, 0.4, n), np.full(n, 0.05)], axis=1)
    return amp * rng.standard_normal((n, 2))


def run_pipeline(emg, q=0.5, per_tick=20, cfg=None):
    ctrl = ControllerPipeline(cfg or config(), MVC)
    ctrl.reset(q)
    return [
        ctrl.step((k + 1) / TICK, emg[k * per_tick : (k + 1) * per_tick], q)
        for k in range(len(emg) // per_tick)
    ]


def test_streaming_equivalence_front_end():
    """Chunk sizes 1, 7 and 20 give identical envelopes (spec 9)."""
    emg = emg_signal()
    outs = []
    for size in (1, 7, 20):
        fe = default_front_end(config())
        outs.append(
            np.concatenate([fe.process(0.0, emg[i : i + size]) for i in range(0, len(emg), size)])
        )
    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[0], outs[2])


def test_causality():
    """Output at tick k is unchanged when samples after tick k are altered (spec 9)."""
    emg = emg_signal()
    altered = emg.copy()
    cut = 1000  # tick 50 ends at sample 1000
    altered[cut:] = 5.0 * np.random.default_rng(9).standard_normal(altered[cut:].shape)
    a, b = run_pipeline(emg), run_pipeline(altered)
    for k in range(cut // 20):
        for field in ("envelope", "intent", "ref_angle", "pd_output", "intensity"):
            np.testing.assert_array_equal(getattr(a[k], field), getattr(b[k], field))
    assert not np.array_equal(a[-1].envelope, b[-1].envelope)


def test_pd_has_no_integral():
    """Constant error -> constant output, no accumulation (spec 2.3, 9)."""
    pd = PD(np.array([0.8]), np.array([0.03]), 0.02, 0.01)
    outs = [pd(np.array([0.3]))[0] for _ in range(500)]
    assert outs[0] == pytest.approx(0.8 * 0.3)
    assert np.ptp(outs) == 0.0


def test_pd_derivative_acts_on_error_change():
    pd = PD(np.array([0.0]), np.array([1.0]), 0.0, 0.01)
    pd(np.array([0.0]))
    assert pd(np.array([0.01]))[0] == pytest.approx(1.0)  # 0.01 rad / 0.01 s


def test_normalization_subtracts_rest():
    norm = MVCNormalizer(MVCValues((0.5, 0.1), (0.1, 0.02)))
    np.testing.assert_allclose(norm(np.array([0.5, 0.01])), [1.0, 0.0])
    np.testing.assert_allclose(norm(np.array([0.3, 0.06])), [0.5, 0.5])


@pytest.mark.parametrize("env,rest", [((0.0, 1.0), None), ((0.5,), (0.6,)), ((np.nan,), None)])
def test_mvc_values_validated(env, rest):
    with pytest.raises(ValueError):
        MVCValues(env, rest)


def test_intent_split_and_deadband():
    intent = AgonistAntagonistIntent((0,), (1,), 0.1)
    assert intent(np.array([0.08, 0.0]))[0] == 0.0
    assert intent(np.array([0.55, 0.0]))[0] == pytest.approx(0.5)
    assert intent(np.array([0.0, 0.55]))[0] == pytest.approx(-0.5)
    assert intent(np.array([3.0, 0.0]))[0] == 1.0


def test_reference_double_integrates_and_clamps():
    ref = DoubleIntegratorReference(
        np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]), 0.01
    )
    ref.reset(np.array([0.5]))
    v, q = ref(np.array([1.0]))
    assert v[0] == pytest.approx(0.1) and q[0] == pytest.approx(0.5005)  # exact: G dt^2 / 2
    for _ in range(200):
        v, q = ref(np.array([1.0]))
    assert q[0] == 1.0 and v[0] == 0.0  # clamped, no velocity wind-up into the limit
    v, q = ref(np.array([-1.0]))
    assert q[0] < 1.0  # leaves the limit immediately


def test_reference_holds_without_intent():
    ref = DoubleIntegratorReference(
        np.array([50.0]), np.array([10.0]), np.array([0.0]), np.array([2.0]), 0.01
    )
    ref.reset(np.array([0.7]))
    for _ in range(100):
        _, q = ref(np.zeros(1))
    assert q[0] == 0.7


def test_allocation_sign_split():
    ctrl = ControllerPipeline(config(), MVC)
    np.testing.assert_allclose(ctrl.allocator(np.array([0.4])), [0.4, 0.0])
    np.testing.assert_allclose(ctrl.allocator(np.array([-2.0])), [0.0, 1.0])


def test_output_exposes_intermediates():
    out = run_pipeline(emg_signal(0.5))[-1]
    for field in ("envelope", "normalized"):
        assert getattr(out, field).shape == (2,)
    for field in ("intent", "ref_velocity", "ref_angle", "error", "pd_output"):
        assert getattr(out, field).shape == (1,)
    assert out.intensity.shape == (2,) and np.all((0 <= out.intensity) & (out.intensity <= 1))


def test_stage_swappable_by_composition():
    """A stage (e.g. blanking) is inserted without subclassing the pipeline."""

    class Zero:
        def reset(self):
            pass

        def process(self, t, chunk):
            return np.zeros_like(chunk)

    cfg = config()
    fe = default_front_end(cfg)
    fe.stages.insert(0, Zero())
    ctrl = ControllerPipeline(cfg, MVC, front_end=fe)
    ctrl.reset(0.5)
    out = ctrl.step(0.01, np.ones((20, 2)), 0.5)
    np.testing.assert_array_equal(out.envelope, 0.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"kp": (0.8, 0.8)},
        {"channel_sign": (1, 0)},
        {"bandpass_hz": (20.0, 1500.0)},
        {"q_min": (2.5,)},
        {"agonist": (5,)},
        {"deadband_floor": 1.0},
        {"allocation_offset": 0.3},  # offset without an epsilon gate
        {"blanking_s": 0.015, "blanking_fill": "zero"},  # zero fill without correction
        {"blanking_s": 0.015, "blanking_correction": True},  # correction with hold
        {"kd": (-0.1,)},
    ],
)
def test_config_validated(kw):
    with pytest.raises(ValueError):
        config(**kw)


def test_reference_velocity_persists_without_damping():
    """Pure double integration (D6): after intent stops, reference velocity persists."""
    ref = DoubleIntegratorReference(
        np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([10.0]), 0.01
    )
    ref.reset(np.array([1.0]))
    for _ in range(10):
        ref(np.array([1.0]))
    v0, q0 = ref(np.zeros(1))
    for _ in range(50):
        v, q = ref(np.zeros(1))
    assert v[0] == v0[0] > 0 and q[0] > q0[0] + 0.4  # still moving, no decay


def test_allocation_offset_is_gated():
    """D1: the threshold offset applies only above epsilon; tiny outputs stay at zero."""
    alloc = SignedAllocation((0, 0), (1, -1), offset=0.35, epsilon=0.02)
    np.testing.assert_allclose(alloc(np.array([0.01])), [0.0, 0.0])  # below the gate
    np.testing.assert_allclose(alloc(np.array([0.1])), [0.35 + 0.65 * 0.1, 0.0])
    np.testing.assert_allclose(alloc(np.array([-0.5])), [0.0, 0.35 + 0.65 * 0.5])
    np.testing.assert_allclose(alloc(np.array([2.0])), [1.0, 0.0])
    plain = SignedAllocation((0, 0), (1, -1))
    np.testing.assert_allclose(plain(np.array([0.01])), [0.01, 0.0])


def test_calibrated_deadband():
    """D14: deadband = max(floor, 99th percentile of |raw intent|) at stim-on rest."""
    rng = np.random.default_rng(0)
    rest = np.stack([0.1 * rng.standard_normal(200000), np.zeros(200000)], axis=1)
    (db,) = calibrate_deadband(rest, (0,), (1,), percentile=99.0, floor=0.05)
    assert db == pytest.approx(0.1 * 2.5758, rel=0.02)  # 99th pct of |N(0, 0.1)|
    quiet = rest * 0.01
    assert calibrate_deadband(quiet, (0,), (1,), percentile=99.0, floor=0.05) == (0.05,)


def test_percentile_deadband_sees_a_one_sided_tail():
    """Rectified, heavy-tailed rest intent: k * sigma would sit well inside the tail."""
    rng = np.random.default_rng(1)
    tail = np.zeros((10000, 2))
    tail[:, 1] = np.maximum(0.02 * rng.standard_cauchy(10000), 0.0)  # one-sided, heavy
    (db,) = calibrate_deadband(tail, (0,), (1,), percentile=99.0, floor=0.0)
    assert np.mean(np.abs(tail[:, 0] - tail[:, 1]) > db) == pytest.approx(0.01, abs=0.002)


def test_pipeline_uses_calibrated_deadband():
    ctrl = ControllerPipeline(config(), MVCValues((0.5, 0.4), (0.0, 0.0), (0.2,)))
    assert ctrl.intent(np.array([0.15, 0.0]))[0] == 0.0  # inside calibrated 0.2, not floor 0.05
    with pytest.raises(ValueError, match="joints"):
        ControllerPipeline(config(), MVCValues((0.5, 0.4), None, (0.2, 0.2)))


def test_normalization_clamps_at_zero():
    """D4: below-rest envelope reads as zero effort, never negative."""
    norm = MVCNormalizer(MVCValues((0.5,), (0.1,)))
    assert norm(np.array([0.02]))[0] == 0.0


def test_damped_reference_is_a_velocity_mapping():
    """D6: steady intent u settles at (G / b) u rad/s with time constant 1 / b."""
    g, b, dt = 50.0, 10.0, 0.01
    ref = DoubleIntegratorReference(
        np.array([g]), np.array([b]), np.array([-1e9]), np.array([1e9]), dt
    )
    ref.reset(np.array([0.0]))
    vs = [ref(np.array([0.2]))[0][0] for _ in range(100)]
    assert vs[-1] == pytest.approx(g / b * 0.2, rel=1e-4)
    tau_tick = round(1 / b / dt)  # after one time constant: 1 - 1/e of the final value
    assert vs[tau_tick - 1] == pytest.approx(g / b * 0.2 * (1 - np.exp(-1)), rel=1e-9)


def test_reference_discretisation_is_exact():
    """Matches the closed-form solution of v' = G u - b v, q' = v for any b * dt."""
    g, b, dt, u = 50.0, 10.0, 0.05, 0.3  # b * dt = 0.5: forward Euler would be 10% off
    ref = DoubleIntegratorReference(
        np.array([g]), np.array([b]), np.array([-1e9]), np.array([1e9]), dt
    )
    ref.reset(np.array([0.0]))
    for _ in range(4):
        v, q = ref(np.array([u]))
    t, vinf = 4 * dt, g / b * u
    assert v[0] == pytest.approx(vinf * (1 - np.exp(-b * t)), rel=1e-12)
    assert q[0] == pytest.approx(vinf * (t - (1 - np.exp(-b * t)) / b), rel=1e-12)
