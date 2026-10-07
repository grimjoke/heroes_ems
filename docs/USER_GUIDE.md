# HEROES EMS Sim: User Guide

This guide explains what the simulator is, how each part works, how to run it, and how to read what it produces. It is written for anyone who uses or reviews the sim, including those who will not change its code.

- Design decisions, findings and open questions are in the [README](../README.md).
- The original requirements are in [SIM_SPEC.md](../SIM_SPEC.md).

> **This is a simulator, not a medical device.** Its results tell you where the controller is fragile and what to measure on the bench. They are not evidence that anything is safe for a person. Every safety-relevant number has to be re-checked on hardware.

---

## 1. What this is

HEROES is an **elbow exoskeleton with no motors**. It moves the arm by electrically stimulating the wearer's own muscles (EMS, electrical muscle stimulation). It decides when and how hard to stimulate by reading the wearer's muscle signals (surface EMG). The intended users are people with partial paralysis, for example after a cervical spinal cord injury. Their muscles still work when stimulated, but their own drive is weak.

The loop on the device is:

1. Electrodes on the biceps and triceps pick up the patient's EMG.
2. The **controller** works out what movement the patient is trying to make.
3. It turns that into a target elbow angle, compares it with the measured angle, and decides how strongly to stimulate each muscle.
4. A **safety supervisor** checks every command and can cut stimulation.
5. The stimulator fires current pulses through skin electrodes, and the muscles contract.

**This repository simulates everything around the controller**: the arm, the patient, the sensors, the stimulator and its side effects. Developers can then run the **real controller code** thousands of times without a patient or hardware.

- The controller (`heroes_control`) and safety supervisor (`heroes_safety`) depend only on numpy and scipy, so the ROS node on the device can wrap exactly the same code.
- Everything else (`heroes_sim`) is the simulated world.

### What it is good for

- Finding failure modes before a person is involved, for example:
  - the controller reading stimulation artifacts as intent and running away;
  - oscillation;
  - drift when the patient is at rest;
  - a fault that the supervisor does not catch.
- Comparing design options under identical, reproducible conditions (same seed gives byte-identical results).
- Running parameter sweeps over many seeds on a laptop or a SLURM cluster.

### What it is not

- It is not validated against human data. Muscle, EMG and stimulation models are plausible textbook models with placeholder numbers, and several hardware facts are still unknown (README, "Hardware facts").
- It does not model skin injury, pain, spasticity, or electrode chemistry beyond impedance and current density.
- It covers one joint: the right elbow, flexion and extension.

---

## 2. Install and first run

You need Python 3.11 and [uv](https://docs.astral.sh/uv/). MuJoCo and MyoSuite are installed as Python packages; no separate install is needed.

```bash
uv sync                                                    # create the environment
uv run pytest                                              # ~200 tests, about 2 minutes
uv run python scripts/run.py configs/scenarios/step_targets.yaml
```

The last command:
1. calibrates a healthy patient (about 40 s of simulated time);
2. runs a 16 s closed-loop episode where they try to reach a series of elbow angles;
3. prints the metrics and writes everything to `runs/step_targets_healthy_s0/`.

Runs are **headless**. Add `--viewer` to watch the arm in the MuJoCo viewer in real time (needs a display).

Try the impaired patient and the key safety scenario:

```bash
uv run python scripts/run.py configs/scenarios/no_intent.yaml --patient configs/patients/sci_c5.yaml
```

---

## 3. The big picture

```
 scenario (what the patient tries to do)
        │
        ▼
 patient model ──► volitional muscle drive ────────────────┐
        │                                                   ▼
        ▼                                          MuJoCo arm + 6 muscles
 EMG sensor model ◄── stim artifacts, M-waves            (gravity, joint limits)
 (noise, mains, crosstalk)                                  │         ▲
        │                                                   ▼         │
        ▼                                            angle sensor     │
 ┌──── heroes_control (the real controller) ────┐           │        │
 │ EMG → envelope → intent → reference angle →  │◄──────────┘        │
 │ PD on angle error → stim intensity per muscle│                    │
 └─────────────────────┬────────────────────────┘                    │
                       ▼                                             │
          heroes_safety supervisor (always last) ──► stimulator ─────┘
                                                     (pulses, recruitment,
                                                      fatigue, electrodes)
```

Clocks:

| Clock | Rate |
|---|---|
| Physics | 2 kHz |
| EMG sampling | 2 kHz |
| Controller and angle sensor | 100 Hz (one "tick" = 10 ms) |
| Stim pulses | 30 Hz (events placed at the nearest physics step) |

Every random component draws from its own stream derived from one run seed. The same config and seed always give the same output, bit for bit.

---

## 4. How each part works

### 4.1 The arm

The arm is MyoSuite's elbow model in MuJoCo:
- one joint, `r_elbow_flex`: 0 is straight, about 2.27 rad (130°) is fully bent;
- three flexors: `BIClong`, `BICshort`, `BRA`;
- three triceps heads: `TRIlong`, `TRIlat`, `TRImed`;
- gravity.

Each muscle's activation is the combination of the patient's own (volitional) drive and stimulation. The exoskeleton frame can add a payload (`plant.exo_payload`, 0 kg by default).

### 4.2 The patient

A patient profile (`configs/patients/*.yaml`) sets:

| Parameter | Meaning |
|---|---|
| `strength` | Per-muscle volitional strength, 0–1. |
| `onset_delay_ms` | How long the patient takes to react to a new target. |
| `cocontraction` | Resting co-activation of both muscle groups. |
| `noise_std` | Motor noise. It also acts as tone, because drive cannot go negative. |
| `drive` | How they move toward a target: approach time constant, maximum speed, gain. The drive is "intended velocity toward the target minus actual velocity", so the patient can hold a posture against gravity. |

Two profiles ship:

| Profile | Description |
|---|---|
| `healthy` | Full strength. Overpowers the stimulation; useful as a sanity baseline. |
| `sci_c5` | C5 incomplete spinal cord injury: flexors at 35–40%, triceps at 5% (near-paralysed), 300 ms onset delay, noisier. At rest the arm drifts slowly toward flexion from asymmetric tone, which is plausible for this injury. **This is the patient the device is for, and the one that exposes most problems.** |

### 4.3 Sensors

**Surface EMG** (2 channels, `biceps` and `triceps`, 2 kHz) is the sum of:

| Component | Model |
|---|---|
| Volitional EMG | Band-limited noise scaled by each muscle's volitional drive (not by activation; stim-evoked activity shows up as M-waves instead). |
| Crosstalk | Each channel also picks up the neighbouring muscles (`emg.channels[].weights`). |
| Stim artifact | A large spike after every pulse, decaying in ~2 ms; much larger on the stimulated muscle's own channel. |
| M-wave | The muscle's electrical response to a stim pulse, 4–14 ms after it. |
| Noise and pickup | White amplifier noise (5 µV), 50 Hz mains and harmonics, slow baseline wander, optional motion artifact. |
| Amplifier | Clips at its rail (supply / 2 / gain = ±8.25 mV at 3.3 V and gain 200) and takes a few ms to recover after clipping. |

Each component has its own `enabled` flag in `configs/base.yaml`, so you can switch any of them off to see what it does.

**Angle sensor** (an IMU on the device, 100 Hz): noise, slowly wandering bias, 10 ms latency, quantization. Each message carries the time it was measured, so the supervisor can tell a stale reading from a fresh one.

### 4.4 Calibration

Calibration runs automatically before every closed-loop episode, as it would on the device. With the arm locked at 90°:

1. **MVC trials.** Rest, then maximum effort with the flexors, rest, maximum extensor effort. The EMG envelope at maximum effort (MVC) is each channel's "100%".
2. **Stim-off rest baseline.** The envelope during the rests is each channel's resting floor: noise, mains pickup, tone. It is subtracted before normalizing (README D4); without that, a weak muscle's noise floor reads as a steady effort.
3. **Quality gate.** If a channel's resting signal is barely above the amplifier's own noise floor, the lead is probably off. Calibration then stops with `CalibrationError` rather than produce a bad calibration.
4. **Stim-on rest (deadband).** Stimulation runs for 30 s at a typical level (0.2) while the patient rests. The controller watches its own intent signal. The 99th percentile of that resting "intent", or 0.05 if larger, becomes the **deadband**: intent smaller than this is ignored. This captures every way the stimulator leaks into the intent signal.

The calibrated values are printed and stored in `meta.json` under `"calibrated"`. Pass them back as `scenario.calibrated` to skip calibration (the tests do this).

### 4.5 The controller

Each 10 ms tick the controller receives the last 20 EMG samples per channel and the latest angle reading. It also receives a sync signal each time the stimulator fires. It then runs these stages, in order:

| # | Stage | What it does | Why |
|---|---|---|---|
| 0 | **Mains canceller** (optional, off by default) | Fits and subtracts the 50/100/150 Hz mains pickup, learning only from samples outside the blanks. | Interpolating across a blank leaves a mains residual the notch can't remove (README D17). |
| 1 | **Blanking** | For 15 ms after each pulse, replaces the EMG with a straight line between the samples either side. This delays the EMG by 15.5 ms. | Removes the artifact and M-wave before any filter can smear them. The EMG amp has no hardware blanking. |
| 2 | **Notch** | Removes 50 Hz mains. | Placed after blanking: before it, every artifact spike would make the notch ring. |
| 3 | **Bandpass** | 20–450 Hz. | Keeps the EMG band; removes wander and high-frequency noise. |
| 4 | **Rectify + envelope** | Absolute value, then a 4 Hz lowpass. | Gives muscle activity as a slowly varying amplitude. |
| 5 | **Normalize** | `(envelope − rest) / (MVC − rest)`, clamped at 0. | 0 = at rest, 1 = maximum effort, per channel. |
| 6 | **Intent** | Biceps minus triceps, with the deadband removed and the rest rescaled to [−1, 1]. | Positive = wants to flex, negative = wants to extend. |
| 7 | **Reference** | Intent sets a reference velocity: steady intent u settles at 5·u rad/s with a 100 ms time constant. The angle is integrated from it and kept inside 0.1–2.1 rad. | Intent moves a target angle rather than driving stim directly. With no intent, the reference stays put. |
| 8 | **PD** | `kp·(reference − measured angle) + kd·d/dt`. No integral term (by design). | Closes the loop on the measured angle. |
| 9 | **Allocation** | Positive PD output goes to the biceps electrode, negative to the triceps. | One electrode per direction. |

The output is a stimulation intensity per channel, 0–1, before safety.

Things to know:
- **The reference is the patient's command.** If it drifts while the patient is at rest, the controller will move the arm on its own. The `no_intent` scenarios exist to catch exactly that.
- **The PD gain is deliberately low** (kp 0.3, README D18). Higher gains oscillate when stimulation has to hold the arm against gravity on its own. The cost is that the arm sags below the reference in that situation.

### 4.6 Stimulation

| Element | Model |
|---|---|
| Intensity | Quantized to the device's 7-bit potentiometer (128 steps). |
| Recruitment | Sigmoid: nothing below about 0.35 intensity, saturating near 0.9. Below the threshold, stimulation does nothing. |
| Electromechanical delay | 25 ms from pulse to force. |
| Electrodes | Each electrode recruits its target muscles, plus some crosstalk to neighbours (`stim.channels[].muscles`). |
| Fatigue | Each muscle's capacity falls while it is being stimulated and recovers slowly (~100 s) at rest. Volitional drive does not fatigue. |
| Contact | A partly lifted electrode delivers the same current through a smaller area. Recruitment is unchanged, but impedance and **current density** rise as 1/contact. Both are logged per channel. |

### 4.7 The safety supervisor

The supervisor sits between the controller and the stimulator and always has the last word. Every tick it applies these rules:

| Rule | Trips when | Effect |
|---|---|---|
| `watchdog` | No controller output for 50 ms | All stim off |
| `sensor_sanity` | NaN, or an implausible angle or EMG value | All stim off |
| `emg_rail` | An EMG channel sits at the amplifier rail (≥ 10 of 20 samples, either sign) | All stim off |
| `emg_dead` | An EMG channel goes flat (std < 0.5 × its calibrated resting std for 10 ticks): lead off | All stim off, **latched until an operator reset** |
| `angle_stale` | The newest angle message is more than 50 ms old | All stim off |
| `angle_frozen` | 10 identical angle readings in a row | All stim off |
| `impedance` | Electrode impedance above `impedance_max_ohm` | That channel off. **Off by default: the real stimulator does not report impedance.** |
| `rom_guard` | Arm within 0.1 rad of a joint limit | Stim pushing further into the limit is cut |
| `cap` | Intensity above the per-channel cap (0.8) | Clipped to the cap |
| `rate_limit` | Intensity rising more than 0.05 per tick | Rise limited; decreases are never limited |
| `dose` | Mean intensity over 10 s above 0.5 | Channel limited |

**Lead-off latch and operator reset:** after `emg_dead` trips, stimulation stays off even if the lead comes back. An operator reset is accepted only once every affected channel has looked healthy for 3 s; otherwise it is refused and logged. In scenarios, `operator_resets: [t1, t2, ...]` requests resets at those times. Set `safety.emg_dead_latch: false` for automatic recovery instead.

The detector thresholds were set from the distribution over hundreds of fault-free runs (not tuned on one seed). In 680 further fault-free runs, none tripped falsely.

---

## 5. Scenarios

| Scenario | What happens | What to look at |
|---|---|---|
| `step_targets` | The patient tries to hold, step to 1.2 and 1.8 rad, ramp and follow a sine. | `tracking_rmse` (arm vs target), `ref_rmse`, `steps_reached`, stim dose |
| `no_intent` | The patient intends nothing; the arm starts at 0.54 rad. **Any reference drift or controller-caused motion is unwanted.** Also runs a stim-off copy with the same seed, so the patient's own drift isn't blamed on the controller. | `ref_max_excursion`, `controller_excursion`, `beyond_limit_max` |
| `no_intent_flexed` | Same, but starting at 1.2 rad, so stim has to hold the arm against gravity. This is the hardest case for loop stability. | Same as `no_intent`, and whether the arm oscillates |
| `fatigue_hold` | sci_c5 raises the arm to 90° and holds for 2 minutes. | Arm spread over time, `final_capacity_*` |
| `mvc_calibration` | Only the calibration. | Printed MVC values; paste `meta.json` "calibrated" into tests |

Scenario files set the initial angle, duration, the target segments (`hold`, `step`, `ramp`, `sine`), and optionally faults, perturbations and operator resets. `closed_loop: false` runs the patient alone, with no controller or stim; use it as the open-loop baseline.

---

## 6. Running things

### One run

```bash
uv run python scripts/run.py <scenario.yaml> [--patient P.yaml] [--seed N] \
    [--override key=value ...] [--out DIR] [--viewer]
```

Overrides use dotted keys into the composed config, and values are parsed as YAML:

```bash
--override controller.joints.r_elbow_flex.kp=0.5
--override emg.powerline.enabled=false
--override scenario.closed_loop=false
--override 'controller.mains_canceller={freqs_hz: [50, 100, 150], mu: 0.002, mu_bias: 0.02}'
--override safety.impedance_max_ohm=2000          # pretend the stimulator reports impedance
```

Every config is validated when it loads. A typo in a key or a muscle name fails immediately with a clear message.

### Demos

| Command | Shows |
|---|---|
| `scripts/passive_drop.py` | The arm falling under gravity (plant sanity check) |
| `scripts/artifact_demo.py [--patient P]` | Stim-artifact feedback with no blanking, zero fill, hold, interpolation, and interpolation plus the mains canceller |
| `scripts/fault_demo.py [--patient P] [--override k=v]` | Every fault injected mid-hold, and which safety rule caught it |
| `scripts/analyze_m5_damping.py` | Analysis of the M5 damping sweeps and the fault-free detector statistics |

---

## 7. Reading the results

Each run writes `runs/<scenario>_<patient>_s<seed>/` (or the `--out` directory):

| File | Contents |
|---|---|
| `log.parquet` | One row per controller tick (100 Hz), every signal in the loop |
| `events.parquet` | Each safety rule firing: time, rule, channel |
| `meta.json` | Fully resolved config, seed, git commit (`-dirty` if the tree had changes), package versions, calibration values |
| `metrics.json` | One number per metric; what sweeps aggregate |

Load the log with pandas: `pd.read_parquet("runs/.../log.parquet")`. Column names end in the joint, channel or muscle name:

| Column | Meaning |
|---|---|
| `t` | Time, s |
| `target_r_elbow_flex` | What the patient is trying to do |
| `q_*`, `qd_*` | True arm angle and velocity |
| `q_meas_*` | What the angle sensor reported |
| `env_*`, `norm_*` | EMG envelope (mV) and normalized (0–1) |
| `intent_*` | Decoded intent, −1..1 |
| `ref_*`, `ref_v_*` | Reference angle and velocity |
| `pd_*`, `cmd_*` | PD output; requested intensity before safety |
| `stim_*` | Intensity actually delivered after safety |
| `act_*`, `u_vol_*`, `u_stim_*` | Muscle activation, its volitional part, its stim part |
| `cap_*` | Fatigue capacity (1 = fresh) |
| `impedance_*`, `current_density_*` | Electrode impedance; delivered intensity per unit contact area (equals `stim_*` at full contact) |
| `safety_<rule>` | Whether that rule was active this tick |
| `fault_<i>_<kind>`, `perturb_*` | Injected faults and external torques |

### Key metrics

| Metric | Meaning | Good |
|---|---|---|
| `tracking_rmse_*` | Arm vs target, after each step settles | Below the open-loop run (`scenario.closed_loop=false`) |
| `ref_rmse_*` | Reference vs target | Small; large means intent isn't reaching the reference |
| `steps_reached_*`, `time_to_target_mean_*`, `overshoot_*` | Per step | 1.0, short, small |
| `ref_max_excursion_*` (no_intent) | How far the reference moved with no intent | ≈ 0 |
| `controller_excursion_*` (no_intent) | Largest gap between the stim-on and stim-off runs: motion caused by the controller | Small (< 0.1 rad) |
| `beyond_limit_max_*`, `time_beyond_limit_s_*` | How far and how long the arm went past the joint limits | 0 |
| `stim_dose_*`, `time_at_cap_*` | Intensity × seconds per channel; time at the cap | Low; a high antagonist dose means stim is fighting the patient |
| `deadband_*` | Calibrated intent deadband | 0.05 (the floor); much higher means stim leaks into intent |
| `final_capacity_*` | Fatigue left at the end | Close to 1 |
| `safety_<rule>` | Number of firings | 0 for the fault rules in a fault-free run |

**Look at the reference and the stim dose, not only tracking error.** A healthy patient overpowers a misbehaving controller, so their tracking can look fine while the controller stimulates against them.

---

## 8. Faults and perturbations

Add these to a scenario (or with `--override 'scenario.faults=[...]'`). Each is active from `t_start` until `t_end`, or to the end of the run if `t_end` is omitted:

| `kind` | Parameters | Simulates |
|---|---|---|
| `electrode_detach` | `channel` (stim), `contact` (0 = fully off) | Stim electrode off or partly lifted. Higher current density, and the controller does not know. |
| `emg_dropout` | `channel` (EMG) | Recording lead off: only amplifier noise remains |
| `emg_saturation` | `channel` (EMG) | Channel stuck at the amplifier rail |
| `angle_freeze` | `joint` | Fresh angle messages carrying a frozen value |
| `angle_stale` | – | Angle driver stuck: old message, old timestamp |
| `artifact_increase` | `factor` | Larger stim artifacts (e.g. drying gel) |

```yaml
faults:
  - {kind: electrode_detach, channel: biceps_stim, contact: 0.3, t_start: 4.0}
perturbations:
  - {t_start: 4.0, t_end: 4.5, torque: {r_elbow_flex: -5.0}}   # + = flexion, N·m
operator_resets: [6.0]
```

---

## 9. Sweeps

A sweep runs a grid of parameter values over many seeds. Spec files live in `configs/sweeps/`:

```yaml
name: my_sweep
grid:
  scenario: [configs/scenarios/step_targets.yaml, configs/scenarios/no_intent.yaml]
  patient: [configs/patients/healthy.yaml, configs/patients/sci_c5.yaml]
  controller.joints.r_elbow_flex.kp: [0.3, 0.4]
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```bash
uv run python scripts/make_sweep.py configs/sweeps/my_sweep.yaml   # writes sweeps/my_sweep/NNNN.yaml
uv run python scripts/run_sweep.py sweeps/my_sweep --jobs 4        # local, resumable
#   or on SLURM: the sbatch line make_sweep prints
uv run python scripts/aggregate.py sweeps/my_sweep                 # summary.csv: mean ± std per point
uv run python scripts/run.py --run-file sweeps/my_sweep/0003.yaml  # rerun one point
```

- Every run file holds the **fully resolved config**, so a sweep reproduces exactly even after `base.yaml` changes.
- Each run imports the package fresh, so **don't edit `src/` while a sweep is running**.
- A typical run takes 15–25 s of wall time.
- **Use 10–20 seeds for any conclusion.** Several problems in this project showed up on 1 seed in 20, or only off seed 0.

---

## 10. Configuration reference

All parameters live in `configs/base.yaml`. Each sweep, scenario or patient file, or `--override`, layers on top of it.

| Section | Holds | Most useful knobs |
|---|---|---|
| `timing` | Clock rates | `stim_hz` |
| `plant` | Model, joints, muscles, payload | `exo_payload.mass_kg` |
| `calibration` | MVC protocol, stim-on rest, quality gate | `stim_rest.intensity`, `quality.min_rest_over_floor` |
| `emg` | Channels, crosstalk, every signal and noise component, amplifier | `*.enabled`, `white_noise.std_mv`, `powerline.amplitude_mv`, `amplifier.supply_v` |
| `angle_sensor` | Noise, bias walk, latency, resolution | `latency_ms` |
| `stim` | Pot steps, recruitment curve, EM delay, fatigue, electrodes | `recruitment`, `fatigue.model` |
| `controller` | Every stage of section 4.5 | `blanking_ms`, `blanking_fill`, `mains_canceller`, `deadband_*`, `joints.*.{gain,damping,kp,kd}` |
| `safety` | Every rule of section 4.7 | `cap`, `emg_dead_latch`, `impedance_max_ohm` |
| `metrics` | Settling window, tolerance | `tolerance_rad` |

Placeholders waiting for hardware data are marked `H1`–`H11` in comments.

---

## 11. Current status and known limitations

The status as of PR #5 is below; the README tracks the details.

- **The impaired patient (sci_c5) currently tracks no better than without stimulation**, unless the mains canceller is switched on. The stim-on deadband absorbs a mains residual on the near-paralysed triceps and the patient loses control of the reference. Switching the canceller on fixes it in simulation. The default awaits a decision (D17).
- **Holding a bent arm on stimulation alone either sags or oscillates.** The interim gain kp 0.3 sags; kp 0.4 and above oscillate past the joint limit (D18).
- **The calibration quality gate can reject a correctly attached lead on a paralysed muscle** when there is no mains pickup (D19).
- **After a fault, the reference keeps moving while stim is off.** A reset would resume toward it (D20).

Hardware risks found so far:
- **The stimulator does not report impedance.** A half-lifted pad gives about 3 mA/cm², above the IEC 60601-2-10 attention level, and software cannot see it.
- **The controller cannot time the pulses**, so the sim's sync-based blanking needs a sync tap or artifact detection on hardware.
- Every pulse saturates the EMG amplifier.
- A series resistor may change the delivered current.

Each item is tracked as a GitHub issue.

### Moving to hardware

The ROS node should wrap `ControllerPipeline` and `SafetySupervisor` unchanged:
- feed them raw EMG chunks, the stamped angle, and the stimulator's pulse sync;
- send `stim_*` to the stimulator.

Things that differ on the device:
- pulse timing is set by the stimulator's programme;
- there is no impedance signal;
- the real noise floor, mains pickup, rail and gain must be measured and entered in the config.

Hardware facts still outstanding are listed in the README.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CalibrationError: resting EMG std below 1.5 x the amplifier noise floor` | A channel looked dead during calibration. Check the lead. In the sim this also happens for sci_c5 with mains switched off (D19). |
| `ValidationError ... Extra inputs are not permitted` | Misspelled config key (e.g. `blanking_s` instead of `blanking_ms`). |
| `blanking_correction must be on with zero fill` | Zero-fill blanking needs `blanking_correction: true`; other fills need it false. |
| Run is slower than expected | Calibration adds about 40 s of simulated time to every closed-loop run; pin `scenario.calibrated` to skip it. The mains canceller adds about 5%. |
| Results differ between machines | Check the `git` and package versions in `meta.json`. Same code, config and seed gives identical logs. |
| A sweep gives odd results after editing code | Code was edited while the sweep ran. Rerun the affected points (`run_sweep.py --rerun`). |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **EMS / FES** | Electrical (functional) muscle stimulation: current pulses through skin electrodes make a muscle contract |
| **sEMG** | Surface electromyography: the muscle's electrical activity measured on the skin |
| **Envelope** | Smoothed amplitude of the EMG: a proxy for how hard a muscle is working |
| **MVC** | Maximum voluntary contraction: the envelope at maximum effort, used as 100% |
| **Intent** | The controller's estimate of what the patient wants: + flex, − extend |
| **Deadband** | Intent below this is treated as zero, so noise does not move the arm |
| **Reference** | The angle the controller is trying to reach, moved by intent |
| **Stim artifact** | The large spike the stim pulse itself puts on the EMG |
| **M-wave** | The stimulated muscle's own electrical response, a few ms after the pulse |
| **Blanking** | Ignoring and refilling the EMG for a short window after each pulse |
| **Recruitment** | The fraction of a muscle activated by a given stim intensity |
| **Crosstalk** | An electrode picking up, or stimulating, a neighbouring muscle |
| **Rail / saturation** | The largest signal the amplifier can output; a signal stuck there is clipped |
| **Lead off** | A recording electrode no longer touching the skin: the channel goes flat |
| **Seed** | Number that fixes all randomness in a run; same seed, same result |
| **SCI C5** | Spinal cord injury at the 5th cervical level: elbow flexors partly preserved, triceps weak |
