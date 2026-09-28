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

## Run a scenario

```bash
uv run python scripts/run.py configs/scenarios/step_targets.yaml --seed 0
uv run python scripts/run.py configs/scenarios/no_intent.yaml --patient configs/patients/sci_c5.yaml
uv run python scripts/run.py configs/scenarios/mvc_calibration.yaml
uv run python scripts/run.py <scenario> --override scenario.closed_loop=false   # volitional only
uv run python scripts/run.py <scenario> --override controller.joints.r_elbow_flex.kp=0.5 --viewer
```

Movement scenarios run closed loop: patient -> EMG -> `heroes_control` -> `heroes_safety` -> stim -> plant. By default each run does the MVC calibration trial first, as on hardware. To skip it, set `scenario.mvc` (and `scenario.mvc_rest`).

Each run writes `runs/<scenario>_<patient>_s<seed>/` containing:
- `log.parquet`: the controller-rate log, with every controller intermediate, applied stim and safety flags
- `events.parquet`: safety rule firings
- `meta.json`: the resolved config, seed, git hash, package versions, MVC values and metrics

Seed 0 with the default configs (full EMG noise model, sample-and-hold blanking) gives:

| | healthy | sci_c5 |
|---|---|---|
| step_targets tracking RMSE, closed loop (open loop) | 0.094 (0.097) rad | 0.206 (0.196) rad |
| step_targets reference-vs-target RMSE | 0.21 rad | 0.28 rad |
| no_intent: arm excursion / reference drift | 0.03 / 0.00 rad | 0.15 / 0.00 rad |
| MVC envelope (rest), biceps / triceps | 0.583 (0.018) / 0.464 (0.017) mV | 0.205 (0.016) / 0.026 (0.014) mV |

A 16 s closed-loop episode plus its 12 s calibration runs at ~6x real time.

### M3: stim artifact feedback and blanking

```bash
uv run python scripts/artifact_demo.py [--patient configs/patients/healthy.yaml]
```

The EMG now carries everything stimulation puts on it:
- a stim artifact on every delivered pulse, scaled by intensity and electrode proximity, with a 2 ms decay;
- M-waves: one sine cycle 4 ms after each pulse, scaled by the recruitment of each muscle;
- 50 Hz mains plus harmonics, baseline wander, and an optional motion artifact coupled to joint velocity;
- amplifier saturation at ±10 mV.

Every component can be switched off under `emg.*`. The loop that results is stim → artifact/M-wave → envelope up → intent → more stim.

| sci_c5, seed 0 | clean EMG (M2) | no blanking | zero-fill blanking | **hold blanking** (default) |
|---|---|---|---|---|
| no_intent reference drift | 0.00 rad | **0.44 rad** (runs into the ROM clamp) | **0.44 rad** | 0.00 rad |
| step_targets tracking RMSE | 0.20 rad | **0.79 rad** | **0.72 rad** | 0.21 rad |
| step_targets triceps dose (intensity x s) | 0.29 | **4.65** | **4.18** | 0.47 |

- **Without blanking the loop runs away into extension.** Triceps stim → triceps artifact → extension intent → more triceps stim. The healthy patient shows the same runaway: the reference hits the extension clamp and the triceps dose reaches 8.5. Yet their tracking RMSE rises only from 0.095 to 0.114, because they overpower the stim with their own muscles. **Tracking error alone would hide this; watch the reference and the dose.**
- **Zero-fill blanking is not enough.** Zeroing a 20 ms window cuts a rectangular notch out of the slow baseline wander. The bandpass turns each notch into a transient that reads as EMG. That is enough to drive the near-paralysed triceps channel, whose MVC envelope is 0.026 mV.
- **Sample-and-hold blanking works.** It is `controller.blanking_fill: hold`, which holds the last pre-pulse sample through a 20 ms window. The window covers the artifact (10 x 2 ms) and the M-wave (4 + 10 ms). It is the default, and it restores M2 performance.
- **Cost:** volitional EMG inside the window is lost. Under stim the envelope reads low by about window / period (20 / 40 ms), so intent is underestimated while stimulating.

### What M2 shows
- **Resting noise floor on a weak channel.** The near-paralysed triceps of `sci_c5` has an MVC envelope of 0.023 mV. Without rest-baseline subtraction, its resting floor (sensor noise, crosstalk, tone) normalizes to 0.15-0.2 of MVC. The controller then reads that as a steady extension intent and stimulates against every flexion. Baseline subtraction (below) fixes it.
- **The stable PD gain is limited by loop delay.** Compute latency, sensor latency, the 25 Hz pulse sample-and-hold, the 25 ms electromechanical delay and muscle activation add up to roughly 75 ms or more. For `sci_c5`, `kp >= 1.2` limit-cycles in `no_intent`. The default is kp 0.8, kd 0.03, which is stable for both patients.
- **Stim below the recruitment threshold does nothing.** With no integral term (by design), small PD errors leave stim under the 0.35 threshold. For `sci_c5`, closed loop improves holds (it gets closer to the 1.8 rad target) but not overall RMSE. The reference also overshoots targets, because the patient keeps pushing while the arm lags. That drift is what M5 is meant to study.

## Tests and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Progress

| Milestone | Status | Notes |
|---|---|---|
| M0: scaffold, config schemas, plant, passive drop | Done | |
| M1: patient model, open-loop movement, MVC trial | Done | |
| M2: clean EMG, controller, safety, stim; closed loop | Done | |
| M3: stim artifact, M-waves, blanking stage | Done | 113 tests passing |
| M4: fatigue, faults, perturbations, sweeps + SLURM | Next | stim `capacity` is the fatigue hook |
| M5: first sweep (noise x gain) | Not started | |

### Deviations from spec
- `timing.stim_hz` is 25, not 30: 2000/30 is not an integer, so the spec's own rate check rejects 30.
- The MJCF muscle order differs from the spec's list; the plant maps by name via config.
- Model is loaded from `simhive/myo_sim/elbow/`, not the `envs/myo/assets` copy (which adds a gym target site/tendon).
- Patient drive is intended velocity toward target, `v_int = clip((target - q)/tau, +-v_max)`, and excitation is `gain * (v_int - qd)`, split by muscle action. The spec gave velocity-proportional as an example. The `- qd` term (the patient's own proprioception) is needed to hold a posture against gravity. `onset_delay_ms` delays the intent only, not the feedback.
- The spec's `mvc_trial(muscle_group)` mode is `Intent(mvc_group=...)`, passed to `Patient.step`. The joint lock for isometric trials is kinematic (`Plant.lock`).
- The MVC protocol lives in `base.yaml` under `calibration`. The `mvc_calibration` scenario just runs it.
- **EMG is modulated by volitional excitation `u_vol`, not by MuJoCo activation.** sEMG reflects motor-unit firing and leads force. The plant's activation also contains the stim-evoked part, which should appear as M-waves (M3), not as volitional EMG. This avoids a second activation filter, which the spec forbids.
- **MVC normalization subtracts a resting baseline:** `(env - rest) / (mvc - rest)`. `rest` is measured in the calibration trial's rest periods (`MVCValues.rest`). See "What M2 shows" for why.
- **Intent has a deadband** (`controller.deadband`, default 0.05; 0 disables it). The rest is rescaled so the output still spans [-1, 1].
- **The reference is a damped double integrator:** `v' = gain*intent - damping*v`. `damping: 0` is the spec's pure double integration. With pure double integration, a reference that is moving keeps moving after intent stops.
- **The safety rate limit applies to rises only.** Decreases are immediate, so a cut is never slowed down.
- Controller and safety configs are plain frozen dataclasses that validate themselves, with no pydantic in the pure packages. The sim resolves YAML names into their indices (`runner.build_*_config`). Stim channel geometry (`stim.channels[].joint/sign`) is the single source for both controller allocation and the safety ROM guard.
- `ControllerPipeline.step` takes `q_meas` as a float or an array[N] (for N joints).
- Not yet included: fatigue (the stim model's `capacity` is fixed at 1 until M4), metrics for time to reach target and overshoot, and the optional full-rate (2 kHz) log.
- The blanking stage gets the stimulator's sync signal (`ControllerPipeline.on_stim_pulse`), as on hardware. Only pulses with nonzero intensity count as delivered: they alone produce artifacts and trigger blanking, so blanking costs nothing while stim is off.
- The blanking fill is sample-and-hold by default, not zero (see M3). `zero` is kept for comparison.
- The calibration requires `rest_s >= 2 * window_s`. The rest baseline is the last `window_s` of each rest, so the previous effort's envelope has time to decay first.
- Muscle-to-joint action comes from `plant.muscle_groups` in config. The plant checks each group's sign against the model's moment arms and fails loudly on a mismatch.
