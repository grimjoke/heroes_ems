# HEROES EMS Sim

A fast, headless, deterministic simulation of a person's arm in an **EMS-only elbow exoskeleton**. It exists so the EMG-driven stimulation controller can be developed and tested without hardware or a patient.

The exoskeleton has no motors. It reads the patient's muscle signals (surface EMG), works out what they are trying to do, and electrically stimulates their own muscles (EMS) to help. The **controller in this repo is the real one**: `heroes_control` and `heroes_safety` depend only on numpy/scipy, so the future ROS node on the device wraps the same code.

Architecture and requirements: [SIM_SPEC.md](SIM_SPEC.md). Status: **M0–M4 done**: closed loop with a realistic EMG model, stim-artifact blanking, muscle fatigue, fault injection and cluster sweeps. Design decisions D1–D9 are resolved and D10–D13 are open (safety gaps found in M4). Next: M5, the first sweep.

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
uv run python scripts/run.py configs/scenarios/fatigue_hold.yaml          # 2-minute hold, sci_c5: fatigue
uv run python scripts/artifact_demo.py                                    # M3: stim-artifact feedback, blanking variants
uv run python scripts/fault_demo.py                                       # M4: each fault mid-hold, what safety catches
uv run python scripts/make_sweep.py configs/sweeps/example.yaml           # M4: sweep -> SLURM or local (see Sweeps)
uv run python scripts/passive_drop.py                                     # M0: arm falls under gravity
uv run pytest                                                             # 156 tests, ~1.5 min
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
| `fatigue_hold` | sci_c5 raises the arm to 90° and holds for 2 minutes. Stim does much of the work, so muscle fatigue builds up. |

Any movement scenario can also carry **faults** and **perturbations**, each active from `t_start` until `t_end` (or the end of the run):

| Fault `kind` | Effect |
|---|---|
| `electrode_detach` (`channel`: stim) | The electrode delivers nothing: no recruitment, no artifact. The controller does not know. |
| `emg_dropout` (`channel`: EMG) | The EMG channel reads a flat 0 (lead off). |
| `emg_saturation` (`channel`: EMG) | The EMG channel is stuck at the amplifier rail (`emg.saturation_mv`). |
| `angle_freeze` (`joint`) | The angle sensor repeats its last reading. |
| `artifact_increase` (`factor`) | Stim artifact amplitude multiplied, e.g. as electrode gel dries. |

`perturbations: [{t_start, t_end, torque: {joint: N·m}}]` applies an external torque (+ = flexion). For example:

```yaml
faults:
  - {kind: electrode_detach, channel: biceps_stim, t_start: 4.0}
perturbations:
  - {t_start: 4.0, t_end: 4.5, torque: {r_elbow_flex: -5.0}}
```

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
| `meta.json` | Resolved config, seed, git hash (with `-dirty` if the tree was modified), package versions, calibration values. |
| `metrics.json` | Flat metrics (one number per key), plus `seed`, `sim_s` and `wall_s`. This is what sweeps aggregate. |

`log.parquet` columns, suffixed by joint, EMG channel, stim channel or muscle name:

| Group | Columns |
|---|---|
| Time and target | `t`, `target_*` |
| Plant | `q_*`, `qd_*`, `torque_*` |
| Sensors | `q_meas_*` (angle sensor), `env_*` (EMG envelope) |
| Controller | `norm_*`, `intent_*`, `ref_v_*`, `ref_*`, `error_*`, `pd_*`, `cmd_*` (intensity before safety) |
| Stim | `stim_*` (intensity after safety), `u_stim_*`, `u_vol_*`, `act_*` (muscle activation), `cap_*` (fatigue capacity) |
| Safety | `safety_<rule>` (whether the rule fired this tick) |
| Faults | `fault_<i>_<kind>` (whether fault i is active), `perturb_*` (external torque) |
| Calibration | `mvc_group` (which calibration trial is running) |

Metrics are printed and saved in `metrics.json`:
- tracking RMSE, for the arm vs the target and for the reference vs the target;
- per target step: fraction of steps reached, mean time to come within `metrics.tolerance_rad`, mean and max overshoot;
- `no_intent`: the reference's maximum excursion and drift rate, the arm's excursion with stim on and off, and `controller_excursion` (the largest gap between the stim-on and stim-off runs, i.e. arm motion caused by the controller);
- stim dose per channel (intensity × s) and time at cap;
- final fatigue capacity per muscle and the minimum across muscles;
- safety event counts by rule (`safety_<rule>`).

## Sweeps

A sweep spec names a scenario, fixed overrides, a grid of dotted keys and a list of seeds. `configs/sweeps/example.yaml` is a small smoke test:

```yaml
name: example
scenario: configs/scenarios/step_targets.yaml
patient: configs/patients/sci_c5.yaml
overrides: {scenario.duration_s: 4.0}
grid:
  controller.joints.r_elbow_flex.gain: [25.0, 50.0]
  emg.white_noise.std_mv: [0.005, 0.02]
seeds: [0, 1]
```

```bash
uv run python scripts/make_sweep.py configs/sweeps/example.yaml   # -> sweeps/example/0000.yaml ... + index.csv
# on a SLURM cluster (make_sweep prints this line with N filled in):
sbatch --array=0-7 --output=sweeps/example/slurm/%a.out scripts/slurm/array.sbatch sweeps/example
# or locally, resumable (finished runs are skipped):
uv run python scripts/run_sweep.py sweeps/example --jobs 4
uv run python scripts/aggregate.py sweeps/example   # -> sweeps/example/summary.csv, mean ± std per grid point
```

- **Every point is validated when the sweep is made**, so a bad value fails on your machine, not on the cluster.
- **Each run file carries the fully resolved config**, so a sweep reproduces exactly even after `base.yaml` or a patient file changes. `scripts/run.py --run-file sweeps/<name>/0003.yaml` reruns one point.
- **Results** go to `sweeps/<name>/results/NNNN/`; SLURM logs to `sweeps/<name>/slurm/`.
- `array.sbatch` requests 1 CPU, 2 GB and 1 hour per task, and pins BLAS threads to 1. It expects `uv` on the nodes and a synced environment on a shared filesystem.

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
  stim.py               quantization, recruitment sigmoid, electrode matrix, pulse hold, EM delay, fatigue
  sensors/emg.py        synthetic sEMG: volitional, artifact, M-waves, noise, mains, wander, motion, saturation
  sensors/kinematics.py angle sensor (with freeze fault)
  scheduler.py          multi-rate clock: sampled clocks at integer ratios, stim pulses as events
  scenario.py           target trajectories, MVC schedule
  runner.py             wires one episode; calibration; fault injection; YAML names → controller/safety indices
  sweep.py              sweep spec → run files; aggregation
  recorder.py, metrics.py (incl. summarize), config.py (pydantic schemas)
configs/              base.yaml (all parameters), patients/, scenarios/, sweeps/
scripts/              run.py, make_sweep.py, run_sweep.py, aggregate.py, slurm/array.sbatch,
                      artifact_demo.py, fault_demo.py, passive_drop.py
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
| `stim` | Pot steps, recruitment curve, EM delay, combination rule, fatigue model and rates, and per-channel electrode matrix `E`, joint and direction. |
| `controller` | Filters, blanking, deadband floor and k, allocation offset, latency; per joint: agonist/antagonist, gain, damping, ROM, kp, kd. |
| `safety` | Caps, rise limit, joint limits and margin, dose window and limit, watchdog timeout, plausible ranges. |
| `metrics` | Settling window and step threshold for tracking RMSE; tolerance for time-to-target. |

## Current results

Seed 0, default configs (full EMG noise model, 15 ms hold blanking, pure double integration, fatigue on):

| | healthy | sci_c5 |
|---|---|---|
| step_targets tracking RMSE, closed loop (open loop) | 0.107 (0.097) rad | **0.427** (0.196) rad |
| step_targets reference-vs-target RMSE | 0.63 rad | 0.48 rad |
| no_intent reference drift | 0.00 rad | 0.00 rad |
| no_intent arm excursion, stim on / stim off / caused by controller | 0.03 / 0.04 / 0.02 rad | 0.15 / 0.20 / 0.17 rad |
| MVC flexion / extension torque at 90° | +51.5 / −36.4 N·m | +16.9 / −3.5 N·m |
| MVC envelope (rest), biceps / triceps | 0.583 (0.018) / 0.464 (0.017) mV | 0.205 (0.016) / 0.026 (0.014) mV |
| Calibrated intent deadband (3σ of resting intent) | 0.05 (floor; 3σ = 0.005) | 0.05 (floor; 3σ = 0.043) |

**The closed loop currently makes sci_c5 tracking worse than no stim at all** (finding 6). A 16 s closed-loop episode plus its 12 s calibration runs at about 5× real time on a typical core; on slower cloud containers, about 2×.

## What the simulation has shown

These are the results most relevant to the hardware. Each is reproducible with the commands shown.

**1. Stim artifacts make the loop run away without blanking (M3).**
The mechanism: stim puts an artifact and an M-wave on the EMG → the envelope rises → the controller reads intent → it stimulates more. `scripts/artifact_demo.py`, seed 0:

| | clean EMG | no blanking | zero-fill + correction | **hold blanking** (default) |
|---|---|---|---|---|
| sci_c5 no_intent reference drift | 0.00 rad | **1.56 rad** (into the flexion clamp) | **0.44 rad** | 0.00 rad |
| healthy no_intent reference drift | 0.00 rad | 0.00 rad | **0.44 rad** | 0.00 rad |
| sci_c5 step_targets triceps dose (intensity × s) | 3.70 | **4.81** | 3.89 | 1.27 |
| healthy step_targets reference RMSE | 0.63 rad | **1.12 rad**, triceps dose 8.5 | 0.59 rad | 0.63 rad |

- **Pure double integration lets a runaway go all the way to the limit.** Nothing bleeds off reference velocity, so once artifact-driven intent starts it moving, the reference crosses the whole range (1.56 rad). With damping, the same case stopped at 0.44 rad.
- **A healthy patient hides it.** Without blanking their reference RMSE nearly doubles and the triceps dose reaches 8.5, yet tracking RMSE only rises from 0.107 to 0.110 rad, because they overpower the stim with their own muscles. **On hardware, monitor the reference and the stim dose, not only tracking error.**

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

| step_targets, seed 0 (measured before fatigue was added; with fatigue the default sci_c5 RMSE is 0.43) | damping 0 (default, as deployed) | damping 10 s⁻¹ (candidate fix) |
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

**8. Fatigue destabilizes the hold under pure double integration (M4).**
`fatigue_hold`, sci_c5, seed 0: over 2 minutes biceps capacity falls to 0.69 (BRA 0.88, crosstalk only), and the patient's own drive roughly triples to compensate.

| arm spread (std of q) per 20 s window | 0–20 s | 20–40 | 40–60 | 60–80 | 80–100 | 100–120 |
|---|---|---|---|---|---|---|
| pure double integration (default) | 0.11 | 0.07 | 0.09 | 0.10 | **0.27** | **0.24** rad |
| same, fatigue off | 0.09 | 0.13 | 0.06 | 0.20 | 0.03 | 0.05 |
| damping 10 s⁻¹, fatigue on | 0.07 | 0.003 | 0.004 | 0.003 | 0.002 | 0.002 |

- Without fatigue, pure double integration already oscillates in bursts. With fatigue, the oscillation grows and persists: in the last 40 s the reference swings between 0.6 and 2.1 rad (the clamp), at up to ±7 rad/s.
- With damping the hold is steady under the same fatigue (tracking RMSE 0.06 vs 0.18 rad). This is more evidence for the M5 damping sweep (D6).

**9. The safety supervisor misses two fault classes (M4).**
`scripts/fault_demo.py`: sci_c5 holds 1.2 rad and each fault starts at 4 s. Upper joint limit 2.27 rad, ROM guard at 2.17.

| fault | pure double integration (default): arm range after fault | with damping 10 s⁻¹ | safety rules that acted |
|---|---|---|---|
| none | 1.09–1.54 rad | 1.11–1.15 | – |
| **biceps EMG saturation** | **0.12–2.29** | **1.11–2.27** (to the limit) | rate limit only |
| **triceps EMG saturation** | **0.40–2.33** | **0.51–1.14** | rate limit, cap |
| **biceps electrode detach** | 1.10–1.43, stim **at cap 1.5 s** | 1.10–1.11, stim **at cap 0.8 s** | cap |
| angle sensor freeze | **−0.04–2.33** | 1.11–1.16 | rate limit, cap |
| artifact ×5 | 0.99–2.30 | 1.11–1.15 | rate limit |
| push −5 N·m for 0.5 s | 0.32–2.33 | 1.05–1.37 | rate limit, cap |
| biceps / triceps EMG dropout | 0.69–1.43 / 1.34–1.90 | ≈ no effect | – |

- **`sensor_sanity` never fired.** A saturated EMG channel sits at the 10 mV amplifier rail, which is under the 20 mV plausibility limit. A frozen angle reading is a plausible value. The supervisor cannot see either (D10, D11).
- **A detached electrode gets stimulated at cap.** The patient keeps intending, the reference runs to its clamp, and the PD error saturates. That is integral-like wind-up through the double integrator, even with no integral term, and only the cap and dose limit bound it (D12).
- **The arm passed the 2.27 rad joint limit** (up to 2.33) in five cases under pure double integration, and reached it with damping (biceps saturation), despite the ROM guard's 0.1 rad margin. The guard acts on position only, and the loop delay plus momentum carry the arm through (D13).
- With damping, only saturation and the detached electrode remain harmful. Under pure double integration a modest push, a larger artifact or a frozen sensor also drive the arm across its whole range.
- The healthy patient's table is dominated by a no-fault pathology: under pure double integration their reference swings rail to rail (0.1–2.1 rad) and stim sits at cap on both channels while they hold the arm steady themselves. `scripts/fault_demo.py --patient configs/patients/healthy.yaml` shows it.

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

### Open decisions (M4)

The supervisor gaps from finding 9 are proposals only. Safety behaviour does not change without sign-off.

| # | Decision | Proposal |
|---|---|---|
| D10 | Detect EMG saturation | Treat a chunk with more than a few samples at the amplifier rail as `sensor_sanity` (zero stim), or set `emg_plausible_abs_mv` just below the rail. Artifact spikes stay under the rail at normal gain, so neither would trip in normal use. |
| D11 | Detect a frozen angle sensor | With 3 mrad noise and 1 mrad resolution, N identical consecutive readings (e.g. 5 ticks = 50 ms) cannot happen by chance; treat that as `sensor_sanity`. |
| D12 | Detect a dead electrode | Prefer the stimulator's own impedance or lead-off check (hardware). If it has none, a supervisor rule could cut a channel that sits at high intensity with no joint response for T seconds. |
| D13 | ROM guard vs momentum | Make the guard velocity-aware (cut when `q + qd·τ` enters the margin, τ ≈ loop delay), or widen the margin. |

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
- **Fatigue is the simpler exponential model** the spec allows (`stim.fatigue.model: exponential`); the interface takes a 3-compartment model later. Fatigue is driven by stim recruitment only and scales `u_stim` and the M-wave; volitional drive does not fatigue.
- **Fault semantics:** a dropout reads 0, and saturation sticks at the positive rail. A detached electrode also produces no artifact, because no current flows.
- Not yet built: a 3-compartment fatigue model, and the optional 2 kHz full-rate log.

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
| M4: fatigue, faults, perturbations, sweeps + SLURM | Done |
| M5: first sweep, tracking error and reference drift vs EMG noise × gain | Next. Also sweep: reference damping (D6), allocation offset (D1), blanking window (D2). |
