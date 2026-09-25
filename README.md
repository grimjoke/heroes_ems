# HEROES EMS Sim

Headless, deterministic closed-loop simulation for developing the EMG-driven EMS controller without hardware. Architecture: [SIM_SPEC.md](SIM_SPEC.md).

## Setup

```bash
uv sync                      # installs Python 3.11 env with mujoco + myosuite
```

## Run M0 (passive drop under gravity)

```bash
uv run python scripts/passive_drop.py            # headless, prints q over time
uv run python scripts/passive_drop.py --viewer   # opt-in MuJoCo viewer (needs a display)
```

Expected headless output: the forearm is released at 1.5 rad, swings down and settles near 0.54 rad, at ~36x real time.

## Tests and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Progress

| Milestone | Status | Notes |
|---|---|---|
| M0: scaffold, config schemas, plant, passive drop | Done | 12 tests passing |
| M1: patient model, open-loop movement, MVC trial | Next | |
| M2: clean EMG, controller, safety, stim; closed loop | Not started | |
| M3: stim artifact, M-waves, blanking stage | Not started | |
| M4: fatigue, faults, perturbations, sweeps + SLURM | Not started | |
| M5: first sweep (noise x gain) | Not started | |

### Deviations from spec
- `timing.stim_hz` is 25, not 30: 2000/30 is not an integer, so the spec's own rate check rejects 30.
- The MJCF muscle order differs from the spec's list; the plant maps by name via config.
- Model is loaded from `simhive/myo_sim/elbow/`, not the `envs/myo/assets` copy (which adds a gym target site/tendon).
