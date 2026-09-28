# HEROES EMS Sim

A fast, headless, deterministic simulation of a person's arm in an **EMS-only elbow exoskeleton**. It exists so the EMG-driven stimulation controller can be developed and tested without hardware or a patient.

The exoskeleton has no motors. It reads the patient's muscle signals (surface EMG), works out what they are trying to do, and electrically stimulates their own muscles (EMS) to help. The **controller in this repo is the real one**: `heroes_control` and `heroes_safety` depend only on numpy/scipy, so the future ROS node on the device wraps the same code.

Architecture and requirements: [SIM_SPEC.md](SIM_SPEC.md). Status: **M0–M3 done** (closed loop with a realistic EMG model and stim-artifact blanking). Next: M4 (fatigue, faults, sweeps).

## The loop

```
 scenario target ──► patient model ──► volitional excitation u_vol ─────────────┐
                                             │                                   ▼
                                             ▼                          combine ─► MuJoCo arm (MyoSuite
                                   sEMG sensor model ◄── pulse artifact,     ▲     elbow, 6 muscles,
                                   (2 kHz, crosstalk,     M-waves            │     gravity)
                                    noise, mains)             ▲              │          │
                                             │                │         u_stim          │
                                             ▼                │              │          ▼
  ┌──────────── heroes_control (real controller) ────┐        │     stim model      angle sensor
  │ [blanking] → bandpass → rectify → envelope →     │        │     (quantize,      (latency, noise,
  │ MVC norm → intent → reference → PD → allocation  │◄───────┼──── recruitment,     bias, 100 Hz)
  └──────────────────────────┬───────────────────────┘ q_meas │     E matrix,           │
                             ▼                                │     EM delay)           │
                 heroes_safety supervisor ────────────────────┴──────► 25 Hz pulses     │
                 (always last before stim)                                              │
                             ▲──────────────────────────────────────────────────────────┘
```

Every stochastic component draws from one run seed, so the same config and seed give byte-identical logs.

## Quick start

```bash
uv sync                                                        # Python 3.11 env: mujoco, myosuite, numpy, scipy, ...
uv run python scripts/run.py configs/scenarios/step_targets.yaml          # closed loop, healthy patient
uv run python scripts/run.py configs/scenarios/no_intent.yaml \
    --patient configs/patients/sci_c5.yaml                                # key safety scenario, impaired patient
uv run python scripts/artifact_demo.py                                    # M3: stim-artifact feedback, blanking off/zero/hold
uv run python scripts/passive_drop.py                                     # M0: arm falls under gravity
uv run pytest                                                             # 113 tests, ~30 s
```

## Running scenarios

```bash
uv run python scripts/run.py <scenario.yaml> [--patient P.yaml] [--seed N] [--override key=value ...] [--out DIR] [--viewer]
```

| Scenario | What happens |
|---|---|
| `step_targets` | Patient tries to flex/extend to a series of angles (steps, a ramp, a sine). Closed loop. |
| `no_intent` | Patient intends nothing, so only resting tone and motor noise remain. **Any reference drift or stimulation here is unwanted.** |
| `mvc_calibration` | Isometric maximum-effort trials per muscle group, with the joint locked. Also runs automatically before every closed-loop episode, as on hardware. |

- **Patients** are parameter sets in `configs/patients/`. `healthy` is the default. `sci_c5` models a C5 spinal cord injury: flexors at 35–40% strength, triceps at 5%, slower and noisier.
- **Overrides** use dotted keys into the composed config. For example:
  - `--override controller.joints.r_elbow_flex.kp=0.5`
  - `--override controller.blanking_ms=0`
  - `--override emg.powerline.enabled=false`
  - `--override scenario.closed_loop=false` (volitional only, no controller or stim)
- **Skipping calibration:** pass `scenario.mvc` and `scenario.mvc_rest`, one value per EMG channel.
- `--viewer` opens the MuJoCo viewer in real time. It needs a display; everything else is headless.

### Outputs

Each run writes `runs/<scenario>_<patient>_s<seed>/`:

| File | Contents |
|---|---|
| `log.parquet` | One row per controller tick (100 Hz). |
| `events.parquet` | Safety rule firings: `t`, `rule`, `channel` (rising edges). |
| `meta.json` | Resolved config, seed, git hash (with `-dirty` if the tree was modified), package versions, MVC values and metrics. |

`log.parquet` columns, suffixed by joint, EMG channel, stim channel or muscle name:

| Group | Columns |
|---|---|
| Time and target | `t`, `target_*` |
| Plant | `q_*`, `qd_*`, `torque_*` |
| Sensors | `q_meas_*` (angle sensor), `env_*` (EMG envelope) |
| Controller | `norm_*`, `intent_*`, `ref_v_*`, `ref_*`, `error_*`, `pd_*`, `cmd_*` (intensity before safety) |
| Stim | `stim_*` (intensity after safety), `u_stim_*`, `u_vol_*`, `act_*` (muscle activation) |
| Safety | `safety_<rule>` (whether the rule fired this tick) |
| Calibration | `mvc_group` (which calibration trial is running) |

Metrics are printed and saved in `meta.json`:
- tracking RMSE, for the arm vs the target and for the reference vs the target;
- the reference's maximum excursion and drift rate (`no_intent`);
- stim dose per channel (intensity × s) and time at cap;
- safety event counts by rule.

## Repository layout

```
src/heroes_control/   the controller, numpy/scipy only
  filters.py            causal SOS bandpass / rectify / lowpass envelope, EMGFrontEnd (stage list)
  blanking.py           stim-artifact blanking stage (hold or zero fill), driven by stimulator sync
  normalization.py      MVC normalization with rest baseline; MVCValues
  intent.py             agonist − antagonist → intent in [-1, 1], deadband
  reference.py          gain + (damped) double integration → reference angle, ROM clamp
  pd.py                 PD, no integral term (clinical safety decision)
  allocation.py         signed PD output → per-channel intensity
  pipeline.py           ControllerConfig, ControllerPipeline.step(t, emg_chunk, q_meas), on_stim_pulse(t)
src/heroes_safety/    supervisor.py: watchdog, sensor sanity, ROM guard, cap, rate limit, dose; numpy only
src/heroes_sim/       everything that is not the controller
  plant.py              MuJoCo wrapper (MyoSuite MJCF loaded directly, no gym env), joint lock
  patient.py            volitional activation model, impairment parameters, MVC effort mode
  stim.py               quantization, recruitment sigmoid, electrode matrix, pulse hold, EM delay
  sensors/emg.py        synthetic sEMG: volitional, artifact, M-waves, noise, mains, wander, motion, saturation
  sensors/kinematics.py angle sensor
  scheduler.py          integer-ratio multi-rate clock
  scenario.py           target trajectories, MVC schedule
  runner.py             wires one episode; calibration; YAML names → controller/safety indices
  recorder.py, metrics.py, config.py (pydantic schemas)
configs/              base.yaml (all parameters), patients/, scenarios/
scripts/              run.py, artifact_demo.py, passive_drop.py
```

## Configuration

Every parameter lives in YAML and is validated at load. Names are cross-checked: every muscle, joint, EMG channel and stim channel referenced anywhere must exist. `configs/base.yaml` sections:

| Section | Holds |
|---|---|
| `timing` | Physics 2 kHz, EMG 2 kHz, controller 100 Hz, stim pulses 25 Hz, angle sensor 100 Hz. Every rate must divide the physics rate. |
| `plant` | Model path, joints, muscles, muscle groups (signs are checked against the model's moment arms), exo payload. |
| `calibration` | MVC protocol: lock angle, rest/effort durations, measurement window. |
| `emg` | Channels and crosstalk matrix `S`, plus every signal and noise component, each with its own `enabled` flag. |
| `angle_sensor` | Noise, bias random walk, latency, resolution. |
| `stim` | Pot steps, recruitment curve, EM delay, combination rule, and per-channel electrode matrix `E`, joint and direction. |
| `controller` | Filters, blanking, deadband, latency; per joint: agonist/antagonist, gain, damping, ROM, kp, kd. |
| `safety` | Caps, rise limit, joint limits and margin, dose window and limit, watchdog timeout, plausible ranges. |
| `metrics` | Settling window and step threshold for tracking RMSE. |

## Current results

Seed 0, default configs (full EMG noise model, sample-and-hold blanking):

| | healthy | sci_c5 |
|---|---|---|
| step_targets tracking RMSE, closed loop (open loop) | 0.094 (0.097) rad | 0.206 (0.196) rad |
| step_targets reference-vs-target RMSE | 0.21 rad | 0.28 rad |
| no_intent: arm excursion / reference drift | 0.03 / 0.00 rad | 0.15 / 0.00 rad |
| MVC flexion / extension torque at 90° | +51.5 / −36.4 N·m | +16.9 / −3.5 N·m |
| MVC envelope (rest), biceps / triceps | 0.583 (0.018) / 0.464 (0.017) mV | 0.205 (0.016) / 0.026 (0.014) mV |

A 16 s closed-loop episode plus its 12 s calibration runs at about 5–6× real time on one core.

## What the simulation has shown

These are the results most relevant to the hardware. Each is reproducible with the commands shown.

**1. Stim artifacts make the loop run away without blanking (M3).**
The mechanism: stim puts an artifact and an M-wave on the EMG → the envelope rises → the controller reads intent → it stimulates more. `scripts/artifact_demo.py`, sci_c5, seed 0:

| | clean EMG (M2) | no blanking | zero-fill blanking | **hold blanking** (default) |
|---|---|---|---|---|
| no_intent reference drift | 0.00 rad | **0.44 rad** (into the ROM clamp) | **0.44 rad** | 0.00 rad |
| step_targets tracking RMSE | 0.20 rad | **0.79 rad** | **0.72 rad** | 0.21 rad |
| step_targets triceps dose (intensity × s) | 0.29 | **4.65** | **4.18** | 0.47 |

- It runs away into extension: triceps stim → triceps artifact → "extend" → more triceps stim.
- **A healthy patient hides it.** Their reference also hits the extension clamp and the triceps dose reaches 8.5. Yet tracking RMSE only rises from 0.095 to 0.114 rad, because they overpower the stim with their own muscles. **On hardware, monitor the reference and the stim dose, not only tracking error.**

**2. Zero-fill blanking is not enough; sample-and-hold is.**
Zeroing a 20 ms window cuts a rectangular notch out of the slow baseline wander, and the bandpass turns each notch into a transient that reads as muscle activity. That is enough to drive a near-paralysed channel. Holding the last pre-pulse sample keeps the baseline continuous and restores M2 performance.
- The window is 20 ms. That covers the artifact (10 × 2 ms decay) and the M-wave (4 ms latency + 10 ms).
- The cost: volitional EMG inside the window is lost, so while stimulating the envelope reads about 50% low (20 of every 40 ms).

**3. Weak channels need rest-baseline subtraction (M2).**
The sci_c5 triceps has an MVC envelope of 0.026 mV and a resting floor of 0.014 mV (sensor noise, 50 Hz pickup, crosstalk, tone). Normalizing by MVC alone reads that floor as a steady extension effort, and the controller stimulated against every flexion. Normalizing as `(env − rest) / (MVC − rest)`, with `rest` measured during the calibration rests, fixes it.

**4. Loop delay limits the PD gain (M2).**
The delays add up to 75 ms or more:
- 10 ms compute latency and 10 ms sensor latency;
- up to 40 ms from the 25 Hz pulse sample-and-hold;
- 25 ms electromechanical delay;
- muscle activation.

With sci_c5, kp ≥ 1.2 oscillates by itself in `no_intent`. The default is kp 0.8, kd 0.03, which is stable for both patients.

**5. Stim below the recruitment threshold does nothing (M2).**
With no integral term (by design), small PD outputs stay under the 0.35 recruitment threshold. For sci_c5 the closed loop holds the arm closer to target but does not improve overall RMSE. The reference also overshoots, because the patient keeps pushing while the arm lags. That drift is what M5 will quantify.

## Open decisions

These choices are currently set to a default in the sim, and each changes controller behaviour. They are listed in the M3 pull request for sign-off.

| # | Decision | Current default |
|---|---|---|
| D1 | Should allocation offset small outputs to the recruitment threshold? (finding 5) | No offset (spec's plain split) |
| D2 | How does the hardware blank artifacts: zero, hold, or not at all? (findings 1–2) | Hold, 20 ms |
| D3 | Should the envelope be compensated for the blanked fraction under stim? | No compensation |
| D4 | Rest-baseline subtraction in normalization (finding 3) | On |
| D5 | Intent deadband | 0.05 |
| D6 | Reference damping (the spec says pure double integration) | 10 s⁻¹ |
| D7 | Safety rate limit on increases only, or both directions | Increases only |
| D8 | Stim pulse rate: 25 Hz, or change the physics rate to allow the spec's 30 Hz | 25 Hz |
| D9 | Is the sci_c5 resting drift (0.15 rad flexion from flexor tone) clinically plausible? | Kept as is |

## Deviations from the spec

Behavioural (each is also an open decision above):
- **MVC normalization subtracts a resting baseline** (D4).
- **Intent has a deadband** (D5). Values inside it are removed and the rest rescaled, so output still spans [-1, 1]. Setting it to 0 disables it.
- **The reference is a damped double integrator,** `v' = gain·intent − damping·v` (D6). Setting `damping: 0` gives pure double integration, where a moving reference keeps moving after intent stops.
- **The safety rate limit applies only to increases** (D7), so a cut is never slowed.
- **Blanking uses sample-and-hold by default** (D2). The blanking stage gets the stimulator's sync signal (`ControllerPipeline.on_stim_pulse`), as on hardware. Only pulses with nonzero intensity count as delivered, so blanking costs nothing while stim is off.
- **`timing.stim_hz` is 25, not 30** (D8). 2000/30 is not an integer, so the spec's own rate check rejects 30.

Modelling:
- **EMG is modulated by volitional excitation, not MuJoCo activation.** sEMG reflects motor-unit firing and leads force. The plant's activation also contains the stim-evoked part, which shows up as M-waves, not volitional EMG. This also avoids a second activation filter, which the spec forbids.
- **Patient drive is intended velocity toward the target minus actual velocity:** `gain·(clip((target − q)/τ, ±v_max) − qd)`, split by muscle action. The spec gave velocity-proportional as an example; the `− qd` feedback term is needed to hold a posture against gravity. `onset_delay_ms` delays the intent, not the feedback.
- The spec's `mvc_trial(muscle_group)` is `Intent(mvc_group=...)`. The isometric joint lock is kinematic (`Plant.lock`).
- The calibration requires `rest_s ≥ 2 × window_s`, so the rest baseline is measured after the previous effort decays.

Plumbing:
- The model is loaded from `simhive/myo_sim/elbow/`, not the `envs/myo/assets` copy (which adds a gym target site and tendon). Muscles are mapped by name, because the MJCF order differs from the spec's list.
- The MVC protocol lives in `base.yaml` under `calibration`; the `mvc_calibration` scenario just runs it.
- Controller and safety configs are self-validating frozen dataclasses, with no pydantic in the pure packages. The sim resolves YAML names into their indices (`runner.build_*_config`). `stim.channels[].joint/sign` is the single source for both allocation and the ROM guard.
- `ControllerPipeline.step` takes `q_meas` as a float or an array[N].
- Not yet built: fatigue (the stim `capacity` is fixed at 1 until M4), time-to-target and overshoot metrics, and the optional 2 kHz full-rate log.

## How the spec's non-negotiables are enforced

| Rule | Enforcement |
|---|---|
| Controller is standalone (numpy/scipy only) | Allowlist import scan over `heroes_control` / `heroes_safety` (`tests/test_import_boundary.py`) |
| Causal, streaming | Causality test (future samples altered → past outputs identical); chunk sizes 1/7/20 give identical output |
| No integral term | Constant error → constant PD output, tested |
| Deterministic | Same seed → byte-identical Parquet, tested; per-component RNG streams keyed by name |
| Headless by default | Viewer only with `--viewer`; tests and runs need no display |
| Config-driven | Pydantic schemas with `extra="forbid"`, cross-section name checks |
| Safety supervisor is last and always on | The runner always routes controller output through it; each rule has a triggering test; NaN → zero stim |

## Development

```bash
uv run pytest                                      # all tests
uv run ruff check . && uv run ruff format .        # lint / format
```

When the EMG model or calibration protocol changes, re-run `mvc_calibration` for both patients and update the pinned MVC values in `tests/test_runner.py`.

## Progress

| Milestone | Status |
|---|---|
| M0: scaffold, config schemas, plant, passive drop | Done |
| M1: patient model, open-loop movement, MVC trial | Done |
| M2: clean EMG, controller, safety supervisor, stim model; closed loop | Done |
| M3: stim artifact, M-waves, full noise model, blanking stage | Done |
| M4: fatigue, faults, perturbations, sweeps + SLURM | Next (stim `capacity` is the fatigue hook) |
| M5: first sweep, tracking error and reference drift vs EMG noise × gain | Not started |
