"""M4 demo: inject each fault mid-hold and see what the safety supervisor catches.

    uv run python scripts/fault_demo.py                                  # sci_c5
    uv run python scripts/fault_demo.py --patient configs/patients/healthy.yaml
    uv run python scripts/fault_demo.py --override controller.joints.r_elbow_flex.damping=10

The patient holds 1.2 rad (step_targets' first hold, extended); each fault starts at
FAULT_T and lasts to the end. Calibrates once (no stim, no faults), then runs every case
with the same seed, so the no-fault row is the reference for all the others.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from heroes_sim.config import load_run_config
from heroes_sim.runner import calibrate, pinned, run

FAULT_T = 4.0
DURATION = 10.0
HOLD = [{"kind": "hold", "duration_s": 1.0}, {"kind": "step", "duration_s": 9.0, "q": [1.2]}]
CASES: dict[str, dict] = {
    "no fault": {},
    "biceps electrode detach": {"faults": [{"kind": "electrode_detach", "channel": "biceps_stim"}]},
    "biceps electrode 30% contact": {
        "faults": [{"kind": "electrode_detach", "channel": "biceps_stim", "contact": 0.3}]
    },
    "triceps electrode detach": {
        "faults": [{"kind": "electrode_detach", "channel": "triceps_stim"}]
    },
    "biceps EMG dropout": {"faults": [{"kind": "emg_dropout", "channel": "biceps"}]},
    "triceps EMG dropout": {"faults": [{"kind": "emg_dropout", "channel": "triceps"}]},
    "biceps EMG saturation": {"faults": [{"kind": "emg_saturation", "channel": "biceps"}]},
    "triceps EMG saturation": {"faults": [{"kind": "emg_saturation", "channel": "triceps"}]},
    "angle frozen value": {"faults": [{"kind": "angle_freeze", "joint": "r_elbow_flex"}]},
    "angle driver stale": {"faults": [{"kind": "angle_stale"}]},
    "artifact x5": {"faults": [{"kind": "artifact_increase", "factor": 5.0}]},
    "push +5 N m, 0.5 s": {
        "perturbations": [{"t_end": FAULT_T + 0.5, "torque": {"r_elbow_flex": 5.0}}]
    },
    "push -5 N m, 0.5 s": {
        "perturbations": [{"t_end": FAULT_T + 0.5, "torque": {"r_elbow_flex": -5.0}}]
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=Path, default=Path("configs/patients/sci_c5.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    scenario = Path("configs/scenarios/step_targets.yaml")
    base = {"scenario.duration_s": DURATION, "scenario.target": HOLD}
    for item in args.override:
        key, _, value = item.partition("=")
        base[key] = yaml.safe_load(value)
    cfg = load_run_config(scenario, overrides=base, patient_path=args.patient)
    base["scenario.calibrated"] = pinned(cfg, calibrate(cfg, args.seed, "calibration/", None))
    j = cfg.plant.joints[0]
    lo, hi = cfg.safety.joint_limits[j]
    print(f"{cfg.patient.name} / seed {args.seed}: hold 1.2 rad, fault from t = {FAULT_T} s")
    print(f"joint limits [{lo}, {hi}] rad, stim cap {max(cfg.safety.cap.values())}\n")
    print(
        f"  {'case':<30} {'q range after':>14} {'ref range':>12} {'max stim b/t':>13} "
        f"{'max J rel':>9}  safety rules fired after the fault (ticks)"
    )
    for label, extra in CASES.items():
        ov = dict(base)
        for key in ("faults", "perturbations"):
            if key in extra:
                ov[f"scenario.{key}"] = [{"t_start": FAULT_T, **f} for f in extra[key]]
        cfg = load_run_config(scenario, overrides=ov, patient_path=args.patient)
        log = run(cfg, args.seed).log
        after = log[log["t"] >= FAULT_T]
        q, ref = after[f"q_{j}"], after[f"ref_{j}"]
        sb, st = after["stim_biceps_stim"], after["stim_triceps_stim"]
        fired = {
            c.removeprefix("safety_"): int(after[c].sum())
            for c in after.columns
            if c.startswith("safety_") and after[c].any()
        }
        dens = after[[c for c in after.columns if c.startswith("current_density_")]].max().max()
        print(
            f"  {label:<30} {q.min():>6.2f}-{q.max():<6.2f} {ref.min():>5.2f}-{ref.max():<5.2f} "
            f"{sb.max():>6.2f}/{st.max():<5.2f} {dens:>9.2f}  {fired or '-'}"
        )
    print("\nmax J rel: peak current density vs full electrode contact (1 = nominal).")
    print("q above the upper limit - margin, or stim at cap while q is pinned there, means the")
    print("supervisor did not stop a fault from driving the arm into its range limit.")


if __name__ == "__main__":
    main()
