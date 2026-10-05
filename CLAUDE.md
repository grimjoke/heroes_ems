# HEROES EMS sim

Read `SIM_SPEC.md` fully before generating code. Follow it over instincts; flag disagreements.

## Commands
- `uv sync` — install
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format
- `uv run python scripts/passive_drop.py [--viewer]` — M0 demo (headless by default)
- `uv run python scripts/run.py configs/scenarios/<name>.yaml [--patient P] [--seed N] [--override k=v] [--viewer]` — one run -> `runs/`
- `uv run python scripts/artifact_demo.py [--patient P]` — stim-artifact feedback: blanking off / zero / hold / interp / interp + mains canceller
- `uv run python scripts/fault_demo.py [--patient P] [--override k=v]` — M4 fault matrix: what the supervisor catches
- `uv run python scripts/make_sweep.py configs/sweeps/<spec>.yaml` → `scripts/slurm/array.sbatch` or `scripts/run_sweep.py sweeps/<name>` → `scripts/aggregate.py sweeps/<name>`
- `uv run python scripts/run.py --run-file sweeps/<name>/NNNN.yaml` — rerun one sweep point
- `uv run python scripts/analyze_m5_damping.py` — M5 damping analysis + no-fault detector statistics (after the three `configs/sweeps/m5_damping_*` sweeps)
- Sweep runs import the package fresh per run: don't edit `src/` while a sweep is running.

## Deviations from spec
Design decisions D1–D9 are resolved in README "Design decisions" (PR #3). The short version:
- Stim pulses are an event clock (nearest physics step); only sampled clocks need integer ratios. Physics 2 kHz, EMG 1:1. Stim 30 Hz (D15: off mains/2).
- Reference damping 10 at gain 50 (D6 resolved: G/b = 5 rad/s per intent, exact exp(−b·dt) discretisation). The hardware still runs pure double integration until changed deliberately. The allocation threshold offset stays an off-by-default sweep option. Don't change these defaults to improve results.
- MVC normalization subtracts the stim-off rest baseline (clamped at 0). Deadband (D14) = max(0.05, p99 |intent|) over 30 s of stim-on rest; the noise-floor quality gate rejects a calibration, it does not set the deadband.
- Blanking is a software stage, 15 ms, interp fill (D15; delays EMG 15.5 ms), 50 Hz notch after blanking (never before: it rings). Zero fill requires envelope correction (enforced), and correction overshoots.
- EMG is modulated by volitional excitation, not activation. Safety rate limit caps rises only; fault paths bypass it.
- `no_intent` runs a stim-off baseline (same seed) so patient drift isn't counted as controller drift.
- Fatigue is the exponential model (stim recruitment only); faults/perturbations are scenario lists.
- Supervisor detectors (D10–D12, D16 resolved): emg_rail (either sign; rail = supply/2/gain, H1), emg_dead (latched until an operator reset, accepted after emg_dead_reset_s healthy; auto-clear is a config option), angle_stale (primary) + angle_frozen (fallback), impedance (per channel; off by default: the stimulator has no impedance reporting, H7). Thresholds come from 395 fault-free sweep runs; re-derive them from a multi-seed sweep (`scripts/analyze_m5_damping.py` statistics) if the EMG or angle model changes, never from one seed.
- Open: D17 mains under the blank (adaptive canceller, `controller.mains_canceller`, off by default), D18 inner PD gain (changed 0.8 → 0.3 interim, flagged), D19 calibration gate on a silent muscle. D13 deferred. Don't change controller or supervisor behaviour for these without sign-off.
- Metrics live in `heroes_sim.metrics.summarize` (one flat dict → metrics.json → sweep table); add new metrics there, not in scripts.
- Default PD gains kp 0.3, kd 0.03 (D18 interim): kp ≥ 0.4 limit-cycles past the joint limit in `no_intent_flexed`, kp 0.8 even in `no_intent` (d6_confirm, d18_inner_gain sweeps). Re-check `no_intent` and `no_intent_flexed` for both patients over 10 seeds (never seed 0 alone) after changing loop timing, filtering delay or gains.
- Pinned calibration values in `tests/test_runner.py` must be refreshed (run `mvc_calibration`, copy `meta.json` "calibrated") whenever the EMG model or calibration protocol changes.
