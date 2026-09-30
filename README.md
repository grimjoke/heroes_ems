# HEROES EMS Sim

A fast, headless, deterministic simulation of a person's arm in an **EMS-only elbow exoskeleton**. It exists so the EMG-driven stimulation controller can be developed and tested without hardware or a patient.

The exoskeleton has no motors. It reads the patient's muscle signals (surface EMG), works out what they are trying to do, and electrically stimulates their own muscles (EMS) to help. The **controller in this repo is the real one**: `heroes_control` and `heroes_safety` depend only on numpy/scipy, so the future ROS node on the device wraps the same code.

Architecture and requirements: [SIM_SPEC.md](SIM_SPEC.md). Status: **M0–M3 done** (closed loop with a realistic EMG model and stim-artifact blanking; design decisions D1–D9 resolved). Next: M4 (fatigue, faults, sweeps).

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
uv run python scripts/artifact_demo.py                                    # M3: stim-artifact feedback, blanking variants
uv run python scripts/passive_drop.py                                     # M0: arm falls under gravity
uv run pytest                                                             # 131 tests, ~40 s
```

## Running scenarios

```bash
uv run python scripts/run.py <scenario.yaml> [--patient P.yaml] [--seed N] [--override key=value ...] [--out DIR] [--viewer]
```

| Scenario | What happens |
|---|---|
| `step_targets` | Patient tries to flex/extend to a series of angles (steps, a ramp, a sine). Closed loop. |
| `no_intent` | Patient intends nothing, so only resting tone and motor noise remain. **Any reference drift or stimulation here is unwanted.** Also runs a stim-off baseline with the same seed, so the patient's own drift is separated from the controller's. |
| `mvc_calibration` | Isometric maximum-effort trials per muscle group, with the joint locked. Also runs automatically before every closed-loop episode, as on hardware. |

- **Patients** are parameter sets in `configs/patients/`. `healthy` is the default. `sci_c5` models a C5 spinal cord injury: flexors at 35–40% strength, triceps at 5%, slower and noisier.
- **Overrides** use dotted keys into the composed config. For example:
  - `--override controller.joints.r_elbow_flex.kp=0.5`
  - `--override controller.blanking_ms=0`
  - `--override emg.powerline.enabled=false`
  - `--override scenario.closed_loop=false` (volitional only, no controller or stim)
- **Skipping calibration:** pass `scenario.calibrated` (`envelope` and `rest` per EMG channel, `deadband` per joint). `meta.json` of any run prints it in exactly that shape.
- `--viewer` opens the MuJoCo viewer in real time. It needs a display; everything else is headless.

### Outputs

Each run writes `runs/<scenario>_<patient>_s<seed>/`:

| File | Contents |
|---|---|
| `log.parquet` | One row per controller tick (100 Hz). |
| `events.parquet` | Safety rule firings: `t`, `rule`, `channel` (rising edges). |
| `meta.json` | Resolved config, seed, git hash (with `-dirty` if the tree was modified), package versions, calibration values and metrics. |

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
- `no_intent`: the reference's maximum excursion and drift rate, the arm's excursion with stim on and off, and `controller_excursion` (the largest gap between the stim-on and stim-off runs, i.e. arm motion caused by the controller);
- stim dose per channel (intensity × s) and time at cap;
- safety event counts by rule.

## Repository layout

```
src/heroes_control/   the controller, numpy/scipy only
  filters.py            causal SOS bandpass / rectify / lowpass envelope, EMGFrontEnd (stage list)
  blanking.py           stim-artifact blanking stage (hold or zero fill), driven by stimulator sync; zero-fill correction
  normalization.py      MVC normalization with rest baseline; MVCValues
  intent.py             agonist − antagonist → intent in [-1, 1], deadband; deadband calibration
  reference.py          gain + double integration → reference angle, ROM clamp (optional damping)
  pd.py                 PD, no integral term (clinical safety decision)
  allocation.py         signed PD output → per-channel intensity (optional gated threshold offset)
  pipeline.py           ControllerConfig, ControllerPipeline.step(t, emg_chunk, q_meas), on_stim_pulse(t)
src/heroes_safety/    supervisor.py: watchdog, sensor sanity, ROM guard, cap, rate limit, dose; numpy only
src/heroes_sim/       everything that is not the controller
  plant.py              MuJoCo wrapper (MyoSuite MJCF loaded directly, no gym env), joint lock
  patient.py            volitional activation model, impairment parameters, MVC effort mode
  stim.py               quantization, recruitment sigmoid, electrode matrix, pulse hold, EM delay
  sensors/emg.py        synthetic sEMG: volitional, artifact, M-waves, noise, mains, wander, motion, saturation
  sensors/kinematics.py angle sensor
  scheduler.py          multi-rate clock: sampled clocks at integer ratios, stim pulses as events
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
| `timing` | Physics 2 kHz, EMG 2 kHz, controller 100 Hz, angle sensor 100 Hz, stim pulses 25 Hz. Sampled clocks must divide the physics rate; stim pulses are events at any rate, fired at the nearest physics step. |
| `plant` | Model path, joints, muscles, muscle groups (signs are checked against the model's moment arms), exo payload. |
| `calibration` | MVC protocol: lock angle, rest/effort durations, measurement window. |
| `emg` | Channels and crosstalk matrix `S`, plus every signal and noise component, each with its own `enabled` flag. |
| `angle_sensor` | Noise, bias random walk, latency, resolution. |
| `stim` | Pot steps, recruitment curve, EM delay, combination rule, and per-channel electrode matrix `E`, joint and direction. |
| `controller` | Filters, blanking, deadband floor and k, allocation offset, latency; per joint: agonist/antagonist, gain, damping, ROM, kp, kd. |
| `safety` | Caps, rise limit, joint limits and margin, dose window and limit, watchdog timeout, plausible ranges. |
| `metrics` | Settling window and step threshold for tracking RMSE. |

## Current results

Seed 0, default configs (full EMG noise model, 15 ms hold blanking, pure double integration):

| | healthy | sci_c5 |
|---|---|---|
| step_targets tracking RMSE, closed loop (open loop) | 0.105 (0.097) rad | **0.504** (0.196) rad |
| step_targets reference-vs-target RMSE | 0.63 rad | 0.54 rad |
| no_intent reference drift | 0.00 rad | 0.00 rad |
| no_intent arm excursion, stim on / stim off / caused by controller | 0.03 / 0.04 / 0.02 rad | 0.15 / 0.20 / 0.17 rad |
| MVC flexion / extension torque at 90° | +51.5 / −36.4 N·m | +16.9 / −3.5 N·m |
| MVC envelope (rest), biceps / triceps | 0.583 (0.018) / 0.464 (0.017) mV | 0.205 (0.016) / 0.026 (0.014) mV |
| Calibrated intent deadband (3σ of resting intent) | 0.05 (floor; 3σ = 0.005) | 0.05 (floor; 3σ = 0.043) |

**The closed loop currently makes sci_c5 tracking worse than no stim at all** (finding 6). A 16 s closed-loop episode plus its 12 s calibration runs at about 5× real time on one core.

## What the simulation has shown

These are the results most relevant to the hardware. Each is reproducible with the commands shown.

**1. Stim artifacts make the loop run away without blanking (M3).**
The mechanism: stim puts an artifact and an M-wave on the EMG → the envelope rises → the controller reads intent → it stimulates more. `scripts/artifact_demo.py`, seed 0:

| | clean EMG | no blanking | zero-fill + correction | **hold blanking** (default) |
|---|---|---|---|---|
| sci_c5 no_intent reference drift | 0.00 rad | **1.56 rad** (into the flexion clamp) | **0.44 rad** | 0.00 rad |
| healthy no_intent reference drift | 0.00 rad | 0.00 rad | **0.44 rad** | 0.00 rad |
| sci_c5 step_targets triceps dose (intensity × s) | 2.67 | **4.64** | 3.69 | 1.51 |
| healthy step_targets reference RMSE | 0.64 rad | **1.11 rad**, triceps dose 8.5 | 0.60 rad | 0.63 rad |

- **Pure double integration lets a runaway go all the way to the limit.** Nothing bleeds off reference velocity, so once artifact-driven intent starts it moving, the reference crosses the whole range (1.56 rad). With damping, the same case stopped at 0.44 rad.
- **A healthy patient hides it.** Without blanking their reference RMSE nearly doubles and the triceps dose reaches 8.5, yet tracking RMSE only rises from 0.105 to 0.114 rad, because they overpower the stim with their own muscles. **On hardware, monitor the reference and the stim dose, not only tracking error.**

**2. Hold beats zero, even with correction.**
With a 15 ms window at 25 Hz (37.5% of samples blanked), measured on steady EMG:

| fill | envelope vs unblanked |
|---|---|
| hold | 0.94 |
| zero | 0.70 |
| zero + kept-fraction correction | **1.11** |

- Zero fill biases the stimulated side's envelope down and notches the baseline wander into fake EMG.
- Dividing by the kept fraction overshoots, because the bandpass smears the gaps, so less signal is lost than the fraction blanked. An inflated envelope on the stimulated side is itself positive feedback: zero + correction runs away in `no_intent` even for the healthy patient.
- Hold keeps the baseline continuous, needs no correction, and stays clean. The software stage is required, because the EMG amplifier has no blanking circuit (D2).
- Under stim with hold blanking, normalized envelopes read *lower* than the same ticks with stim off (sci_c5 triceps 0.000 vs 0.013). So the residual artifact floor stays below the stim-off calibrated baseline (the D4 caveat), and intent under stim is slightly underestimated.

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
With no integral term (by design) and no allocation offset (D1), small PD outputs stay under the 0.35 recruitment threshold, a dead zone the PD cannot close. The gated offset exists as a config flag (`controller.allocation`), off by default, for the M5 sweep.

**6. Pure double integration, as deployed, degrades tracking (D6).**
With `damping: 0` (the hardware controller), reference velocity persists after intent stops, so the reference overshoots every target.

| step_targets, seed 0 | damping 0 (default, as deployed) | damping 10 s⁻¹ (candidate fix) |
|---|---|---|
| healthy reference RMSE / tracking RMSE | 0.63 / 0.105 rad | 0.22 / 0.094 rad |
| sci_c5 tracking RMSE (open loop 0.196) | **0.50 rad** | 0.21 rad |
| sci_c5 hold at the 1.8 rad target (open loop 1.61) | **1.56 rad** (worse than no stim) | 1.70 rad |

- No gain fixes it: across gains 10–100, reference RMSE stays at 0.36–0.73 rad.
- `no_intent` is unaffected, because the calibrated deadband keeps resting intent at exactly 0 and the velocity never starts.
- Damping stays available as the candidate fix for the M5 sweep. If it wins, change the hardware controller deliberately.

**7. The sci_c5 resting drift is flexion, from asymmetric tone (D9).**
With stim off and no intent, the arm drifts +0.157 rad (9°) toward flexion, in 5 of 5 seeds and never toward extension.

| sci_c5, stim off, 5 seeds | net drift |
|---|---|
| default | +0.157 rad |
| no cocontraction | +0.078 |
| no motor noise | +0.083 |
| neither | −0.003 (hangs at rest) |

- Resting cocontraction and motor noise contribute about equally. Motor noise acts as tone because excitation is clipped at zero, so noise has a positive mean. Both are scaled by per-muscle strength (flexors 0.35–0.4, triceps 0.05).
- Preserved biceps, paralysed triceps, drift toward flexion: plausible for C5 SCI.
- `no_intent` now subtracts a stim-off baseline run with the same seed, so this drift is not counted as controller drift.

## Design decisions

Resolved in [PR #3](https://github.com/grimjoke/heroes_ems/pull/3).

| # | Decision | Resolution |
|---|---|---|
| D1 | Allocation offset at the recruitment threshold | **Off** by default: an offset makes tiny noisy outputs jump to threshold-level stim. It is a config flag (`controller.allocation.offset`), gated by `epsilon` (enforced), to sweep at M5. |
| D2 | Artifact blanking | **The EMG amp has no blanking circuit** (to confirm against the datasheet), so the sim applies none in hardware; blanking is a software stage in the controller. Default window **15 ms** (the M-wave ends by 14 ms), to sweep at M5. **Hold** fill. |
| D3 | Envelope correction for blanking | **None**, because the fill is hold. Zero fill *requires* correction, and correction is only allowed with zero fill (`ControllerConfig` raises otherwise). |
| D4 | Rest-baseline subtraction | **Yes**, clamped at zero after subtracting. Under stim with hold blanking, the residual floor stays below the calibrated baseline (finding 2). |
| D5 | Intent deadband | **Calibrated:** `max(floor 0.05, k·σ_rest)` with k = 3, from resting intent in the calibration trial. |
| D6 | Reference damping | **0 (pure double integration), as on hardware.** Damping is kept as an option and candidate fix for the M5 sweep (finding 6). Velocity is zeroed at the ROM clamp (tested). |
| D7 | Rate-limit direction | **Increases only.** Fast shutoff is the safe direction. Watchdog and sensor-sanity paths bypass it entirely (tested). |
| D8 | Stim pulse rate | **Integer ratios apply to sampled clocks only.** Stim pulses are events at the nearest physics step (jitter ≤ 0.25 ms). Physics stays at 2 kHz with 1:1 EMG. The rate is 25 Hz until set to the stimulator's real rate. |
| D9 | sci_c5 resting drift | **Plausible** (flexion, asymmetric tone; finding 7). `no_intent` subtracts a stim-off baseline run. |

## Deviations from the spec

Behavioural:
- **MVC normalization subtracts a resting baseline,** clamped at zero (D4).
- **Intent has a deadband calibrated per patient** (D5). Values inside it are removed and the rest rescaled, so output still spans [-1, 1].
- **The safety rate limit applies only to increases** (D7); fault paths bypass it.
- **Blanking is a software stage with sample-and-hold fill** (D2, D3). It gets the stimulator's sync signal (`ControllerPipeline.on_stim_pulse`), as on hardware. Only pulses with nonzero intensity count as delivered, so blanking costs nothing while stim is off.
- **Stim pulses are an event clock** (D8). The spec's integer-ratio check applies to sampled clocks only.
- Available but off by default, for M5: reference damping (D6) and the allocation threshold offset (D1). With both off, the reference and allocation are exactly as the spec describes.

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

When the EMG model or calibration protocol changes, re-run `mvc_calibration` for both patients and paste each `meta.json` `"calibrated"` block into the pinned values in `tests/test_runner.py`.

## Progress

| Milestone | Status |
|---|---|
| M0: scaffold, config schemas, plant, passive drop | Done |
| M1: patient model, open-loop movement, MVC trial | Done |
| M2: clean EMG, controller, safety supervisor, stim model; closed loop | Done |
| M3: stim artifact, M-waves, full noise model, blanking stage | Done |
| M4: fatigue, faults, perturbations, sweeps + SLURM | Next (stim `capacity` is the fatigue hook) |
| M5: first sweep, tracking error and reference drift vs EMG noise × gain | Not started. Also sweep: reference damping (D6), allocation offset (D1), blanking window (D2). |
