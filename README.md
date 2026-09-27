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

## Run a scenario (M1: open-loop volitional movement)

```bash
uv run python scripts/run.py configs/scenarios/step_targets.yaml --seed 0
uv run python scripts/run.py configs/scenarios/no_intent.yaml
uv run python scripts/run.py configs/scenarios/mvc_calibration.yaml --patient configs/patients/sci_c5.yaml
uv run python scripts/run.py <scenario> --override patient.onset_delay_ms=0 --viewer
```

Each run writes `runs/<scenario>_<patient>_s<seed>/{log.parquet, meta.json}`: a controller-rate log, plus the resolved config, seed, git hash and package versions. Seed 0 with the default configs gives:

| Scenario | healthy | sci_c5 |
|---|---|---|
| step_targets: tracking RMSE (1 s settle excluded after steps) | 0.097 rad | 0.196 rad |
| no_intent: max excursion from rest | 0.04 rad | 0.20 rad |
| mvc_calibration: flexor / extensor torque at 90 deg | +51.5 / -36.4 N m | +16.9 / -3.5 N m |

## Tests and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Progress

| Milestone | Status | Notes |
|---|---|---|
| M0: scaffold, config schemas, plant, passive drop | Done | |
| M1: patient model, open-loop movement, MVC trial | Done | 36 tests passing |
| M2: clean EMG, controller, safety, stim; closed loop | Next | MVC values switch to EMG envelope |
| M3: stim artifact, M-waves, blanking stage | Not started | |
| M4: fatigue, faults, perturbations, sweeps + SLURM | Not started | |
| M5: first sweep (noise x gain) | Not started | |

### Deviations from spec
- `timing.stim_hz` is 25, not 30: 2000/30 is not an integer, so the spec's own rate check rejects 30.
- The MJCF muscle order differs from the spec's list; the plant maps by name via config.
- Model is loaded from `simhive/myo_sim/elbow/`, not the `envs/myo/assets` copy (which adds a gym target site/tendon).
- Patient drive is intended velocity toward target, `v_int = clip((target - q)/tau, +-v_max)`, and excitation is `gain * (v_int - qd)`, split by muscle action. The spec gave velocity-proportional as an example. The `- qd` term (the patient's own proprioception) is needed to hold a posture against gravity. `onset_delay_ms` delays the intent only, not the feedback.
- The spec's `mvc_trial(muscle_group)` mode is `Intent(mvc_group=...)`, passed to `Patient.step`. The joint lock for isometric trials is kinematic (`Plant.lock`).
- Until M2 there is no EMG, so MVC results are muscle activation and joint torque over the best window. The controller's MVC values will come from the EMG envelope in the same trial.
- Muscle-to-joint action comes from `plant.muscle_groups` in config. The plant checks each group's sign against the model's moment arms and fails loudly on a mismatch.
