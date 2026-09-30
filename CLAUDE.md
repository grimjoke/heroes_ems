# HEROES EMS sim

Read `SIM_SPEC.md` fully before generating code. Follow it over instincts; flag disagreements.

## Commands
- `uv sync` — install
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format
- `uv run python scripts/passive_drop.py [--viewer]` — M0 demo (headless by default)
- `uv run python scripts/run.py configs/scenarios/<name>.yaml [--patient P] [--seed N] [--override k=v] [--viewer]` — one run -> `runs/`
- `uv run python scripts/artifact_demo.py [--patient P]` — M3 stim-artifact feedback: blanking off / zero / hold

## Deviations from spec
Design decisions D1–D9 are resolved in README "Design decisions" (PR #3). The short version:
- Stim pulses are an event clock (nearest physics step); only sampled clocks need integer ratios. Physics 2 kHz, EMG 1:1.
- Reference is pure double integration (`damping: 0`) as on hardware; damping and the allocation threshold offset exist only as M5 sweep options. Don't change these defaults to improve results.
- MVC normalization subtracts a rest baseline (clamped at 0); the intent deadband is calibrated as max(floor, k·σ_rest).
- Blanking is a software stage, 15 ms, hold fill. Zero fill requires envelope correction (enforced), and correction overshoots.
- EMG is modulated by volitional excitation, not activation. Safety rate limit caps rises only; fault paths bypass it.
- `no_intent` runs a stim-off baseline (same seed) so patient drift isn't counted as controller drift.
- Default PD gains (kp 0.8, kd 0.03) sit below the delay-limited stability bound (~1.2 for sci_c5); re-check `no_intent` for both patients after changing loop timing or gains.
- Pinned calibration values in `tests/test_runner.py` must be refreshed (run `mvc_calibration`, copy `meta.json` "calibrated") whenever the EMG model or calibration protocol changes.
