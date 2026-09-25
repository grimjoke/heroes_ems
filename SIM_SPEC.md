# HEROES EMS Sim: Architecture Spec (v1)

Brief for Claude Code. Read fully before generating code. When this spec and your instincts disagree, follow the spec and flag the disagreement.

## 1. Purpose

A fast, headless, deterministic closed-loop simulation for developing and testing the EMG-driven EMS controller without hardware.

Loop: simulated patient intent → muscle activation → synthetic sEMG (with stimulation artifacts) → **real controller code** → safety supervisor → EMS stimulation model → muscle excitation → MuJoCo musculoskeletal plant → joint angle sensor → controller.

Constraints:
- The exoskeleton is **EMS-only**. No motors. The exo frame is passive; the only actuators are the patient's muscles (volitional + stimulated).
- No ROS in the loop. The controller runs in-process. A ROS bridge is out of scope for v1.
- Plant models come from **MyoSuite**. No OpenSim dependency.

## 2. Non-negotiables

1. **Controller is a standalone package** (`heroes_control`) importing only `numpy` and `scipy`. No `mujoco`, `myosuite`, `rospy`, or anything from `heroes_sim`. Enforce with a test (see Section 9). The future ROS node wraps this same package.
2. **Controller is causal and streaming.** Stateful, sample-by-sample or small-chunk processing. `scipy.signal.sosfilt` with carried `zi` state. **Never `filtfilt`** or any non-causal operation.
3. **No integral action in the PD controller.** Clinical safety decision: avoids escalating stimulation when a patient physically cannot reach a target. Do not add an I term, anti-windup schemes, or "helpful" bias correction.
4. **Deterministic.** Every stochastic component takes a `numpy.random.Generator` derived from one run seed. Same config + seed = bit-identical logs.
5. **Headless by default.** Viewer is opt-in via a flag. Must run on a SLURM cluster node with no display.
6. **Config-driven.** Every parameter lives in YAML, validated by pydantic models. No magic numbers in module code.
7. **Safety supervisor is separate from the controller** and is always the last stage before stimulation. It cannot be disabled by config (it can be parameterized).

## 3. Repo layout

```
heroes-sim/
  pyproject.toml
  CLAUDE.md                 # short pointer to this spec + dev commands
  SIM_SPEC.md               # this file
  src/
    heroes_control/         # pure controller, numpy/scipy only
      filters.py            # causal bandpass, rectify, lowpass envelope
      normalization.py      # MVC normalization
      intent.py             # agonist/antagonist split -> [-1, 1]
      reference.py          # gain + double integration -> reference angle
      pd.py                 # PD, no integral
      allocation.py         # signed PD output -> per-channel stim intensity
      pipeline.py           # composes the above; single step(t, emg, angle) API
    heroes_safety/
      supervisor.py         # limits, watchdog, fault handling
    heroes_sim/
      plant.py              # MuJoCo model wrapper
      stim.py               # EMS stimulation model
      patient.py            # volitional activation model
      sensors/
        emg.py              # synthetic sEMG + stim artifact + M-waves
        kinematics.py       # joint angle sensor model
      scheduler.py          # multi-rate clock
      scenario.py           # trajectories, profiles, faults
      runner.py             # wires everything, runs one episode
      recorder.py           # logging to Parquet
      metrics.py
      config.py             # pydantic schemas
  configs/
    base.yaml
    patients/{sci_c5.yaml, healthy.yaml, ...}
    scenarios/{step_targets.yaml, no_intent.yaml, ...}
  scripts/
    run.py                  # one config -> one run
    make_sweep.py           # grid -> numbered config files
    slurm/array.sbatch      # SLURM_ARRAY_TASK_ID -> config file
  tests/
```

## 4. v1 scope

- **Joint:** elbow flexion/extension, 1 DoF.
- **Model:** MyoSuite `myoelbow_1dof6muscles.xml` (muscles: BIClong, BICshort, BRA, TRIlong, TRIlat, TRImed). Resolve the path from the installed MyoSuite package; verify the filename and muscle names against the installed version and fail loudly if they differ.
- **Do not use MyoSuite's gym environments.** Load the MJCF directly with the `mujoco` Python bindings. The gym wrappers impose frame-skip and reward structure we don't want.
- **EMS channels:** 2 (biceps electrode, triceps electrode).
- **EMG channels:** 2 (biceps, triceps).
- **Gravity on.** Forearm hangs at rest.

Everything must generalize to N joints, M muscles, K electrodes, J EMG channels via the mapping matrices below. No elbow-specific logic outside config.

## 5. Timing (scheduler.py)

Integer-ratio multi-rate loop driven by physics steps. Defaults:

| Clock | Rate | Notes |
|---|---|---|
| Physics | 2 kHz (dt = 0.5 ms) | Set `model.opt.timestep` explicitly |
| EMG sampling | 2 kHz | 1 sample per physics step |
| Controller | 100 Hz | Receives the EMG buffer since last tick |
| EMS pulse train | 30 Hz | Stim pulses are discrete events |
| Angle sensor | 100 Hz | Configurable latency |

Validate at startup that all rates divide the physics rate. Controller output takes effect with a configurable compute latency (default 1 controller tick = 10 ms).

## 6. Modules and interfaces

Use frozen dataclasses for data passed between modules. Arrays are `float64`.

### 6.1 Plant (`plant.py`)
- `reset(qpos, qvel)`, `step(excitation: ndarray[M])`, `joint_state() -> (q, qd)`, `muscle_state() -> activation, force, length`.
- Excitation in [0, 1] goes to `data.ctrl`. MuJoCo's built-in muscle activation dynamics are the **only** activation dynamics in the system. No other module adds first-order activation filtering.
- Optional exo payload: extra passive mass/inertia on the forearm body from config.

### 6.2 Patient (`patient.py`)
- Input: intended movement (target trajectory from scenario) and current joint state.
- Output: volitional excitation `u_vol[M]`.
- v1 model: simple intent-to-excitation mapping (e.g. proportional to intended velocity toward target, split agonist/antagonist) scaled by impairment parameters:
  - `strength` per muscle (0 = paralysed, 1 = healthy)
  - `onset_delay_ms`
  - `cocontraction` baseline
  - `noise` (motor noise on excitation)
- SCI and stroke profiles are parameter sets, not code branches.
- MVC calibration: patient model must support an `mvc_trial(muscle_group)` mode (max voluntary effort, isometric via locked joint) used by the runner to compute the controller's MVC values, exactly as on hardware.

### 6.3 Stimulation model (`stim.py`)
Input: per-channel intensity `s[K]` in [0, 1] from the safety supervisor. Output: stimulated excitation `u_stim[M]`.

Per channel:
1. **Quantization** to digital-pot steps (`n_steps` in config).
2. **Recruitment curve:** sigmoid from intensity to recruitment fraction (threshold, slope, saturation params).
3. **Electrode-to-muscle matrix** `E[M, K]`: primary targets plus crosstalk (e.g. biceps electrode also recruits some BRA).
4. **Pulse structure:** recruitment is applied as twitches at pulse frequency, not a continuous signal. v1 may approximate as continuous excitation held between pulses; keep the event interface so twitch modeling can be added.
5. **Electromechanical delay:** fixed delay line.
6. **Fatigue:** per-muscle capacity state in [0, 1], decreasing with stim-induced recruitment, recovering at rest (3-compartment or simpler exponential model, pluggable). Effective `u_stim = recruitment * capacity`.

**Combining with volitional:** `u_total = 1 - (1 - u_vol) * (1 - u_stim)`, clipped to [0, 1]. Make the combination rule a pluggable function.

### 6.4 EMG sensor (`sensors/emg.py`)
Output: raw sEMG `emg[J]` at 2 kHz.

- **Muscle-to-sensor matrix** `S[J, M]` with crosstalk.
- **Volitional component:** activation-modulated bandlimited noise (roughly 20 to 450 Hz), amplitude scaled per muscle. Interface must allow swapping in a Fuglevand motor-unit model later without changing callers.
- **Stimulation artifact:** large short spike on every stim pulse, amplitude scaling with stim intensity and electrode proximity, with exponential decay.
- **M-waves:** evoked compound response time-locked to each pulse, few ms latency, amplitude scaling with recruitment.
- **Noise:** sensor white noise, powerline (50 Hz, Greece) with harmonics, baseline wander, optional motion artifact coupled to joint velocity.
- All components individually toggleable in config for ablations.

Rationale: stim artifact and M-waves contaminating the EMG the controller reads is a primary failure mode (possible positive feedback: stim -> M-wave -> envelope up -> more stim). The sim must expose it, not hide it.

### 6.5 Angle sensor (`sensors/kinematics.py`)
IMU-like joint angle: white noise, bias random walk, latency, quantization. Output at 100 Hz.

### 6.6 Controller (`heroes_control/pipeline.py`)
Implements the existing hardware chain:

1. Causal bandpass (Butterworth SOS)
2. Full-wave rectification
3. Causal lowpass envelope
4. MVC normalization
5. Agonist/antagonist split -> intent in [-1, 1]
6. Gain
7. Double integration -> reference joint angle (clamped to ROM)
8. PD on (reference - measured angle), **no integral**
9. Allocation: positive output -> agonist channel, negative -> antagonist, each in [0, 1]

API:
```python
class ControllerPipeline:
    def __init__(self, cfg: ControllerConfig, mvc: MVCValues): ...
    def reset(self, q0: float) -> None: ...
    def step(self, t: float, emg_chunk: ndarray, q_meas: float) -> ControllerOutput: ...
```
`ControllerOutput` exposes every intermediate (envelope, normalized, intent, reference velocity, reference angle, error, PD output, channel intensities) for logging. Stage implementations must be swappable (e.g. insert a stim-artifact blanking stage or a denoiser before integration) via composition, not subclassing.

### 6.7 Safety supervisor (`heroes_safety/supervisor.py`)
Sits between controller output and stim model. Pure numpy.
- Per-channel intensity cap
- Rate limit (max intensity change per tick)
- ROM guard: cut stim that drives further toward a joint limit when within a margin
- Dose limit: rolling stim duty/energy cap per channel
- Watchdog: controller output stale or NaN -> all channels to zero
- Sensor sanity: angle or EMG out of plausible range -> zero stim
- Emits events (which rule fired, when) to the recorder.

### 6.8 Scenario (`scenario.py`)
YAML-defined:
- Target trajectory: step sequence, ramps, sinusoid, hold periods
- Patient profile reference
- Initial state
- Faults with onset times: electrode detach (channel gain -> 0), EMG channel dropout/saturation, angle sensor freeze, sudden artifact increase
- External perturbation torques

Required built-in scenarios:
- `no_intent`: patient intends nothing, only noise. Measures reference drift and unwanted stim. Key safety scenario.
- `step_targets`: flex/extend to a series of angles.
- `mvc_calibration`: runs before every episode unless MVC values are provided.
- `fatigue_hold`: long isometric hold to exercise fatigue.

### 6.9 Recorder and metrics
- Parquet, one file per run, plus a JSON with resolved config, seed, git hash, package versions.
- Log at controller rate by default; optional full-rate (2 kHz) log of raw EMG and plant state.
- Metrics (computed post-hoc from logs, `metrics.py`):
  - Tracking RMSE (actual vs target), and reference vs target separately
  - Reference drift rate in `no_intent`
  - Total stim dose per channel, time-at-cap
  - Final fatigue capacity
  - Safety events count by type
  - Time to reach within tolerance of target, overshoot

## 7. Running

- `python scripts/run.py configs/scenarios/step_targets.yaml --seed 0 [--viewer] [--override key=value ...]`
- `python scripts/make_sweep.py sweep.yaml -> sweeps/<name>/0000.yaml ...`
- `scripts/slurm/array.sbatch` maps `SLURM_ARRAY_TASK_ID` to a config file, writes results to a per-sweep output dir. No Hydra; keep it transparent.
- Aggregation script collects metric JSONs from a sweep into one table.

Target: one 30 s elbow episode well under real time on one CPU core.

## 8. Milestones

- **M0:** Repo scaffold, config schemas, plant loads MyoElbow, passive drop under gravity renders in viewer and runs headless.
- **M1:** Patient model + open-loop volitional movement. MVC calibration trial.
- **M2:** Clean EMG model (no stim artifact) + controller pipeline + safety supervisor + stim model. Closed loop runs. `step_targets` and `no_intent` pass sanity checks.
- **M3:** Stim artifact, M-waves, full noise model. Demonstrate the failure mode, then add a blanking stage as a swappable controller stage.
- **M4:** Fatigue, faults, perturbations. Sweep infrastructure + SLURM script.
- **M5:** First sweep: tracking error and reference drift vs EMG noise level x controller gain. Figure-ready output.

## 9. Tests (pytest)

- **Import boundary:** `heroes_control` and `heroes_safety` import nothing from `mujoco`, `myosuite`, `heroes_sim`. (AST scan or import-linter.)
- **Causality:** controller output at time t is unchanged when future samples are altered.
- **Streaming equivalence:** processing a signal in chunks of size 1, 7, and 20 gives identical output.
- **No integral:** constant error yields constant PD output (for zero error derivative).
- **Determinism:** two runs with same seed produce identical Parquet.
- **Safety:** each supervisor rule has a test that triggers it; NaN input -> zero stim.
- **Stim model:** zero intensity -> zero stim excitation; recruitment monotonic; quantization respected.
- **Plant sanity:** no excitation + gravity -> forearm settles at hanging posture; full biceps excitation -> flexion.
- **Scheduler:** rates that don't divide physics rate raise at startup.

## 10. Out of scope for v1

ROS bridge, multi-joint models, lower limb, Fuglevand motor-unit EMG, learned denoisers, RL, GUI beyond the MuJoCo viewer. Keep interfaces ready for them; do not build them.

## 11. Stack

Python 3.11, `mujoco`, `myosuite` (model assets only), `numpy`, `scipy`, `pydantic`, `pyyaml`, `pyarrow`/`pandas`, `pytest`. Package manager: `uv`. Lint/format: `ruff`.
