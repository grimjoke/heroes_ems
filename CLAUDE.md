# HEROES EMS sim

Read `SIM_SPEC.md` fully before generating code. Follow it over instincts; flag disagreements.

## Commands
- `uv sync` — install
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format
- `uv run python scripts/passive_drop.py [--viewer]` — M0 demo (headless by default)
- `uv run python scripts/run.py configs/scenarios/<name>.yaml [--patient P] [--seed N] [--override k=v] [--viewer]` — one run -> `runs/`

## Deviations from spec
- `timing.stim_hz` is 25, not 30: 2000/30 is not an integer, so the spec's own rate check rejects 30.
- MyoSuite muscle order in the MJCF differs from the spec list; the plant maps by name via config.
- Patient drive uses proprioceptive velocity feedback (`gain * (v_int - qd)`); MVC mode is `Intent(mvc_group=...)`.
- EMG is modulated by volitional excitation (not activation); MVC normalization subtracts a rest baseline; intent has a deadband; reference has optional damping; safety rate limit caps rises only. Full list: README "Deviations".
- Default PD gains (kp 0.8, kd 0.03) sit below the delay-limited stability bound (~1.2 for sci_c5); re-check `no_intent` for both patients after changing loop timing or gains.
