# HEROES EMS sim

Read `SIM_SPEC.md` fully before generating code. Follow it over instincts; flag disagreements.

## Commands
- `uv sync` — install
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format
- `uv run python scripts/passive_drop.py [--viewer]` — M0 demo (headless by default)

## Deviations from spec
- `timing.stim_hz` is 25, not 30: 2000/30 is not an integer, so the spec's own rate check rejects 30.
- MyoSuite muscle order in the MJCF differs from the spec list; the plant maps by name via config.
