# HEROES EMS Sim

A fast, headless, deterministic simulation of a person's arm in an **EMS-only elbow exoskeleton**. It exists so the EMG-driven stimulation controller can be developed and tested without hardware or a patient.

The exoskeleton has no motors. It reads the patient's muscle signals (surface EMG), works out what they are trying to do, and electrically stimulates their own muscles (EMS) to help. The **controller in this repo is the real one**: `heroes_control` and `heroes_safety` depend only on numpy/scipy, so the future ROS node on the device wraps the same code.

Architecture and requirements: [SIM_SPEC.md](SIM_SPEC.md). Status: **M0–M4 done, M5 underway.** The closed loop has a realistic EMG model, stim-artifact blanking, muscle fatigue, fault injection, cluster sweeps and a safety supervisor that detects every simulated sensor and electrode fault. D6 and D14–D16 are resolved and implemented (reference damping, stim-on deadband calibration, interpolating blank with a notch after it, lead-off latch). Open decisions: mains under the blank (D17), the inner PD gain (D18), and the calibration gate on a silent muscle (D19).

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
  │ blank → notch → bandpass → rectify → envelope →  │        │     (quantize,      (latency, noise,
  │ MVC norm → intent → reference → PD → allocation  │◄───────┼──── recruitment,     bias, 100 Hz)
  └──────────────────────────┬───────────────────────┘ q_meas │     E matrix,           │
                             ▼                                │     EM delay)           │
                 heroes_safety supervisor ────────────────────┴──────► 30 Hz pulses     │
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
uv run pytest                                                             # 177 tests, ~1.5 min
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
| `timing` | Physics 2 kHz, EMG 2 kHz, controller 100 Hz, angle sensor 100 Hz, stim pulses 30 Hz. Sampled clocks must divide the physics rate; stim pulses are events at any rate, fired at the nearest physics step. |
| `plant` | Model path, joints, muscles, muscle groups (signs are checked against the model's moment arms), exo payload. |
| `calibration` | MVC protocol (lock angle, rest/effort durations, measurement window), the stim-on rest for the deadband (D14), and the noise-floor quality gate. |
| `emg` | Channels and crosstalk matrix `S`, every signal and noise component (each with its own `enabled` flag), and the amplifier (supply, gain → rail; overload recovery). |
| `angle_sensor` | Noise, bias random walk, latency, resolution. |
| `stim` | Pot steps, recruitment curve, EM delay, combination rule, fatigue model and rates, and per-channel electrode matrix `E`, joint and direction. |
| `controller` | Filters, blanking (window, fill), mains notch, optional mains canceller (D17), deadband floor and percentile, allocation offset, latency; per joint: agonist/antagonist, gain, damping, ROM, kp, kd. |
| `safety` | Caps, rise limit, joint limits and margin, dose window and limit, watchdog timeout, plausible ranges; D10–D12 detector thresholds; lead-off latch and reset (D16). |
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
- Hold keeps the baseline continuous, needs no correction, and stays clean. The software stage is required, because the EMG amplifier has no blanking circuit (D2, H3).
- **Since D15 the default fill is interpolation**, which also removes the step at the end of the hold (finding 11). It still leaves a mains residual under each blank (finding 13).
- Under stim with hold blanking, normalized envelopes on seed 0 read *lower* than the same ticks with stim off (sci_c5 triceps 0.000 vs 0.013). **That seed-0 reading was misleading:** across 20 seeds, hold blanking itself leaks on the weak triceps channel (finding 11).

**3. Weak channels need rest-baseline subtraction (M2).**
The sci_c5 triceps has an MVC envelope of 0.026 mV and a resting floor of 0.014 mV (sensor noise, 50 Hz pickup, crosstalk, tone). Normalizing by MVC alone reads that floor as a steady extension effort, and the controller stimulated against every flexion. Normalizing as `(env − rest) / (MVC − rest)`, with `rest` measured during the calibration rests, fixes it.

**4. Loop delay limits the PD gain (M2).**
The delays add up to 75 ms or more:
- 10 ms compute latency and 10 ms sensor latency;
- up to 40 ms from the 25 Hz pulse sample-and-hold;
- 25 ms electromechanical delay;
- muscle activation.

With sci_c5, kp ≥ 1.2 oscillates by itself in `no_intent`. kp 0.8 looked stable on seed 0 but is marginal across seeds (finding 12); the default is now kp 0.6, kd 0.03 (D18).

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

**9. Supervisor fault coverage: every simulated sensor and electrode fault is now caught (M4, D10–D12).**
`scripts/fault_demo.py`: sci_c5 holds 1.2 rad and each fault starts at 4 s. Pure double integration (the default); joint limit 2.27 rad.

| fault | arm range, M4 (before the detectors) | arm range now | rule that caught it |
|---|---|---|---|
| none | 1.09–1.54 rad | 1.09–1.54 | – |
| biceps / triceps EMG saturation | **0.12–2.29 / 0.40–2.33** | 1.10–1.43 | `emg_rail`, first tick |
| biceps / triceps EMG dropout (lead off) | 0.69–1.43 / 1.34–1.90 | 1.10–1.43 | `emg_dead`, within 10 ticks |
| angle driver stale | – (new) | 1.10–1.43 | `angle_stale`, 50 ms |
| angle frozen value (fresh messages) | **−0.04–2.33** | 1.10–1.43 | `angle_frozen`, 10 ticks |
| biceps electrode detached / 30% contact | stim at cap into a dead electrode | 1.10–1.43 | `impedance`, first pulse |
| artifact ×5 | 0.99–2.30 | 0.99–2.30 | – (controller law, not a sensor fault) |
| push −5 N·m for 0.5 s | 0.32–2.33 | 0.32–2.33 | – (controller law) |

- With damping 10 s⁻¹ the last two rows stay within 1.05–1.39 rad (finding 10).
- A channel stuck at the rail is also flat, so `emg_dead` fires after `emg_rail` on it; that is harmless.
- **Thresholds come from 395 fault-free sweep runs**, not one seed:
  - 0 samples ever reached the rail;
  - a live channel's chunk std never stayed below 0.5 × its resting std for more than 3 ticks, while a dead lead sits at 0.27 (biceps) to 0.39 (triceps);
  - identical angle readings never exceeded 5 in a row.
- **Verified:** 0 false trips over 40 more fault-free runs with all detectors live (both scenarios, both patients, 10 seeds); each fault class trips its rule within 0.5 s of onset (tested).

**10. M5 damping sweep: damping 5–10 s⁻¹ removes the joint-limit breaches and the fatigue oscillation (M5).**
`configs/sweeps/m5_damping_*.yaml`, `scripts/analyze_m5_damping.py`. 10 seeds for `step_targets`, 5 for `fatigue_hold`.

| sci_c5, gain 50 | step_targets tracking RMSE | runs past the joint limit | fatigue_hold arm spread after 80 s |
|---|---|---|---|
| **damping 0 (hardware)** | **0.435 ± 0.065 rad** | **10/10**, up to 0.057 rad | **0.175 ± 0.086 rad**, 4/5 past the limit |
| damping 2 | 0.278 ± 0.073 | 3/10 | – |
| damping 5 | 0.208 ± 0.024 | 0/10 | 0.006 ± 0.005 |
| damping 10 | 0.206 ± 0.012 | 0/10 | 0.004 ± 0.003 |
| damping 20 | 0.227 ± 0.012 | 0/10, but only half the steps reached | – |

- **What matters is gain/damping**, the reference speed per unit of intent. A ratio of 5–10 works. At 2.5 the reference is too sluggish to reach steps (damping 20 at gain 50, damping 10 at gain 25). At 25 or more it overshoots and breaches the limit (damping 2; gain 100 at almost any damping).
- **Healthy patient:** tracking is about 0.095–0.11 rad throughout, because they do the work. But damping 10 at gain 50 cuts stim dose from about 3.0 / 3.3 to 1.4 / 1.1 (biceps / triceps) and reference RMSE from 0.63 to 0.21 rad: much less stim fighting them.
- The breaches are small in angle (≤ 0.06 rad past 2.27) but systematic, and they happen **with no fault**.

**11. `no_intent` is not clean across seeds: resting noise leaks past the deadband and pure double integration accumulates it (M5).**
Seed 0 showed zero reference drift; 20 seeds do not.

| sci_c5 no_intent, 20 seeds | runs with reference drift > 0.01 rad | worst reference drift | worst arm motion caused by the controller | worst past the joint limit |
|---|---|---|---|---|
| damping 0 | 8/20 | 0.44 rad (to the clamp) | 0.88 rad | 0.12 rad |
| damping 10 | 7/20 | 0.36 rad | 0.92 rad | 0.12 rad |
| healthy, either | 0/20 | 0 | 0.03 rad | 0 |

Two mechanisms, isolated on seeds 1 and 11. Artifact and M-wave were ruled out: turning them off changes nothing.
- **The deadband is too narrow for the weak channel's noise tail.** Resting raw intent reaches 0.07–0.13 at the 99.9th percentile in 14/20 seeds, against a deadband of 0.05. The deadband comes from 3σ over two 1 s rest windows of a 4 Hz envelope, only about 16 independent samples. The sci_c5 triceps normalization span (MVC − rest ≈ 0.012 mV) amplifies noise, and the zero clamp makes it one-sided (extension). **With stim never delivered, 13/20 seeds still drift**, so this is not a stim effect (D14).
- **Hold-blanking edges.** Each pulse starts a 15 ms hold. The held sample differs from the live signal by the mains (20 µV at 50 Hz; 15 ms is ¾ of a cycle) plus wander, and the step when the hold ends rings through the bandpass as fake EMG. On seed 1 the drift is gone with stim off and shrinks with mains and wander off (D15).
- Damping bounds how fast a leak turns into drift but does not remove the leak, so D14 and D15 matter whichever reference law is chosen.
- **Resolved by D14/D15** (finding 12): the reference never reaches the joint limit over 20 seeds; the residual drift is ≤ 0.06 rad.

## Design decisions

Resolved in [PR #3](https://github.com/grimjoke/heroes_ems/pull/3).

| # | Decision | Resolution |
|---|---|---|
| D1 | Allocation offset at the recruitment threshold | **Off** by default: an offset makes tiny noisy outputs jump to threshold-level stim. It is a config flag (`controller.allocation.offset`), gated by `epsilon` (enforced), to sweep at M5. |
| D2 | Artifact blanking | **The EMG amp has no blanking circuit** (to confirm against the datasheet), so the sim applies none in hardware; blanking is a software stage in the controller. Default window **15 ms** (the M-wave ends by 14 ms). Fill was **hold**; now **interpolation** (D15). |
| D3 | Envelope correction for blanking | **None** with hold or interp fill. Zero fill *requires* correction, and correction is only allowed with zero fill (`ControllerConfig` raises otherwise). |
| D4 | Rest-baseline subtraction | **Yes**, clamped at zero after subtracting. Under stim with hold blanking, the residual floor stays below the calibrated baseline (finding 2). |
| D5 | Intent deadband | Was `max(0.05, 3σ_rest)` from 2 s of stim-off rest; **superseded by D14.** |
| D6 | Reference damping | Was 0 (pure double integration, as on hardware); **revisited after M5: damping 10** (below). Velocity is zeroed at the ROM clamp (tested). |
| D7 | Rate-limit direction | **Increases only.** Fast shutoff is the safe direction. Watchdog and sensor-sanity paths bypass it entirely (tested). |
| D8 | Stim pulse rate | **Integer ratios apply to sampled clocks only.** Stim pulses are events at the nearest physics step (jitter ≤ 0.25 ms). Physics stays at 2 kHz with 1:1 EMG. The rate is now **30 Hz** (D15); the real rate is the device programme's (H10). |
| D9 | sci_c5 resting drift | **Plausible** (flexion, asymmetric tone; finding 7). `no_intent` subtracts a stim-off baseline run. |

Resolved after M4 (PR #4):

| # | Decision | Resolution |
|---|---|---|
| D10 | EMG saturation and dead leads | **`emg_rail`**: ≥ 10 of 20 samples at \|x\| ≥ 0.98 × rail, **either sign**. **`emg_dead`**: chunk std < 0.5 × calibrated resting std for 10 ticks, cleared after 10 live ticks (hysteresis stops a noisy dead lead flickering stim back on). A dead antagonist turns agonist noise into intent, so both matter. The dropout fault now reads the amplifier noise floor, not an exact 0. |
| D11 | Frozen angle sensor | **Primary: `angle_stale`**, message age > 50 ms (messages carry their acquisition stamp). **Fallback: `angle_frozen`**, 10 identical readings, for fresh messages carrying a frozen value. 10 is 2× the worst fault-free run (5, over 395 runs). |
| D12 | Dead or partly detached electrode | **Stimulator impedance reporting** (option A): `impedance` cuts a channel above `impedance_max_ohm`. Partial contact keeps the current but raises impedance and current density as 1/contact, which is a **skin-burn risk**, so this is a safety requirement. If the real stimulator cannot report impedance, that goes on the hardware risk list. The artifact-presence check stays research; the movement-based rule is rejected. |
| D13 | Joint-limit guard | **Deferred to the damping decision.** Damping 5–10 removes the breaches (finding 10). If a residual problem remains, evaluate **active braking** (stimulate the antagonist near the limit) before velocity-aware cutting, with care: co-contraction, and a wrong-direction pulse if the angle is wrong. |

Resolved after M5 (PR #5):

| # | Decision | Resolution |
|---|---|---|
| D6 | Reference damping | **Adopted: damping 10 s⁻¹ at gain 50**, a velocity mapping of G/b = 5 rad/s per unit intent with τ = 100 ms (the low end of 5–10). Discretised exactly (zero-order hold, `a = exp(−b·dt)`), not forward Euler. Confirmed over 20 seeds (finding 12). Damping does not stop a constant leak from ramping the reference; the deadband must. |
| D14 | Deadband calibration | **`max(0.05, 99th percentile of |intent|)` over 30 s of rest with stim on** at a typical level (0.2 on both channels), arm supported, after a 2 s settle. The stim-off rest stays the D4 baseline. A **noise-floor quality gate** (resting std ≥ 1.5 × the amplifier floor on every channel) rejects a bad calibration; it does not set the deadband. |
| D15 | Blanking edges | **Interpolate across the blank** (a line between the samples either side; output delayed by window + 1 sample = 15.5 ms), **50 Hz notch after blanking**, **stim at 30 Hz** (off 25 Hz = mains/2, where every blank lands at the same mains phase). A notch before blanking (rings on every artifact) and a one-mains-cycle 20 ms window are rejected. |
| D16 | Lead-off: hysteresis or latch | **Latch until an operator reset.** The reset is accepted only after every tripped channel has been healthy (no dead run) for `emg_dead_reset_s` (3 s); otherwise it is refused and logged. Auto-clear hysteresis stays a config option (`emg_dead_latch: false`) for sim scenarios. |

### Open decisions

| # | Decision | Evidence | Proposal |
|---|---|---|---|
| **D17** | Mains under the blank | Finding 13: interpolating across 15 ms of 50 Hz leaves a residual at 50 ± k·30 Hz that the notch after blanking cannot remove. On the sci_c5 triceps (MVC span 0.02 mV) it reads as 0.17–0.20 intent, so D14 sets the deadband to 0.16–0.20 and the patient can no longer drive the reference (half the steps never reached; the damped-hold test fails, marked `xfail`). | **An adaptive mains canceller ahead of blanking** (`controller.mains_canceller`, off by default): LMS fit of 50/100/150 Hz sinusoids on unblanked samples only, frozen during blanks, so it cannot ring the way a notch before blanking would. It brings the stim-on tail back to the noise floor and the deadband back to 0.05 (finding 13). Alternatives: per-channel deadbands, or measure the real mains pickup first (it may be far below the 20 µV modelled). |
| **D18** | Inner PD gain | Finding 12: at kp 0.8 the sci_c5 `no_intent` loop limit-cycles on 12/20 seeds (arm swings up to 1.0 rad, past the joint limit). It is marginal even at 25 Hz stim; the D15 changes (30 Hz, 15 ms interp delay) tipped it. | **Changed to kp 0.6** (kd 0.03): 0/20 limit cycles, same tracking. Flagged because it is a gain change on the hardware controller; revert if you disagree. |
| **D19** | Calibration gate vs a silent muscle | Finding 13 side effect: with no mains pickup, the sci_c5 triceps resting std (5.2 µV) is below 1.5 × the 5 µV floor, so the D14 quality gate rejects a correctly attached lead on a paralysed muscle. The D16 reset check is relative to the calibrated resting std, not the floor, so it is unaffected once calibration passes. | Gate on the impedance / lead-off signal where the hardware has one (it does not: H7); otherwise lower the gate to just above 1 × floor on channels flagged as paralysed, or skip it for them and rely on the operator. Needs the real noise floor (H4). |

## Hardware facts

Answers to the H1–H11 list, as supplied. Each sets a config value or a risk.

| # | Fact | Answer | In the sim |
|---|---|---|---|
| H1 | EMG amp rail | rail = supply / 2 / gain; RAW gain ~200 → ±12.5 mV at 5 V, ±8.25 mV at 3.3 V | `emg.amplifier` {`supply_v` 3.3, `raw_gain` 200} → `saturation_mv` derived; both rails tested. **Confirm the supply voltage.** |
| H2 | Overload recovery | Not specified; **every stim pulse saturates this amp**. To measure (notes, step 2). | Modelled: `overload_recovery_ms` 5, exponential return after leaving the rail. |
| H3 | Hardware blanking | None | Software blanking stage (D2, D15) |
| H4 | Noise floor | Not specified; measure 60 s at rest | `emg.white_noise.std_mv` stays 5 µV |
| H5 | IMU | Unknown; model or a 60 s still recording to come | `angle_sensor.*` unchanged |
| H6 | IMU stamps | Check the ROS message header stamp / seq | Assumed stamped (D11 primary) |
| H7 | Stimulator impedance reporting | **None** (consumer unit) | `safety.impedance_max_ohm: null`; D12 has no data source (notes, item 3) |
| H8 | Current or voltage control | 200 mA into 500 Ω, probably current-controlled; the series resistor may change what the patient gets (notes, item 1) | Current-controlled |
| H9 | Electrodes, limits | Sanitas ~45 × 45 mm (~20 cm²); IEC 60601-2-10 attention above 2 mA rms/cm². 200 mA, 450 µs, 25 Hz ≈ 30 mA rms = 1.5 mA/cm² full pad, **3 mA/cm² with half the pad lifted** | Current density logged per channel |
| H10 | Pulse rate | Set by the device programme (1–150 Hz), **not by the controller** (notes, item 4) | `timing.stim_hz` 30 (D15) |
| H11 | Controller gain | 50, to confirm in the ROS node config | `gain` 50 |

**Hardware risk list:**
- **No impedance or lead-off reporting (H7).** A partly lifted pad keeps the same current over less area: at half contact the H9 example is 3 mA/cm², above the 2 mA/cm² attention level. Nothing in software can see it (tested: `test_partial_contact_undetected_without_impedance_reporting`). Needs a hardware or procedural mitigation.
- **The controller cannot time the pulses (H10).** The sim's blanking (and the D17 canceller's freeze) assume a stimulator sync signal, as the spec does. With the device running its own programme, blanking needs either a sync output tapped from the stimulator or artifact detection on the EMG (threshold on the rail, H2). Until then, D15 results are an upper bound on what the hardware can do.
- **Every pulse saturates the EMG amp (H2).** The sim's 8 mV artifact sits just under the 8.25 mV rail; real artifacts are larger. Recovery time decides whether 15 ms of blanking is enough; measure it.
- **The series resistor (H8)** may change the delivered current; measure what reaches the pads.
- **Pure double integration (D6)** is what the hardware runs today; the sim now uses damping 10. Change the hardware deliberately.

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
- **Fault semantics:** a dropout reads only the amplifier noise floor; saturation sticks at the positive rail; a stale angle driver repeats the last message with its old stamp; a frozen angle sends fresh stamps with the old value. A fully detached electrode delivers no current (no recruitment, no artifact); a partly detached one delivers the same current at higher impedance and current density.
- **The supervisor has rules beyond the spec's list** (D10–D12): `emg_rail`, `emg_dead`, `angle_stale`, `angle_frozen` (all zero every channel) and `impedance` (zeroes one channel). Each detector's parameter can switch it off, but the sim config always sets them.
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

When the EMG model or calibration protocol changes, re-run `mvc_calibration` for both patients and paste each `meta.json` `"calibrated"` block (including `emg_rest_std`) into the pinned values in `tests/test_runner.py`. Pinned calibrations are validated: with the dead-lead detector on, `emg_rest_std` is required.

## Progress

| Milestone | Status |
|---|---|
| M0: scaffold, config schemas, plant, passive drop | Done |
| M1: patient model, open-loop movement, MVC trial | Done |
| M2: clean EMG, controller, safety supervisor, stim model; closed loop | Done |
| M3: stim artifact, M-waves, full noise model, blanking stage | Done |
| M4: fatigue, faults, perturbations, sweeps + SLURM | Done |
| M5: first sweep, tracking error and reference drift vs EMG noise × gain | Underway: damping × gain sweep done (findings 10–11). Next: noise × gain, plus allocation offset (D1), blanking window and hold edges (D2, D15), deadband calibration (D14). |
