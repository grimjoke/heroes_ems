"""One scenario config -> one run. Headless by default.

    uv run python scripts/run.py configs/scenarios/step_targets.yaml --seed 0
    uv run python scripts/run.py configs/scenarios/step_targets.yaml \\
        --patient configs/patients/sci_c5.yaml --override controller.deadband_k=4
    uv run python scripts/run.py --run-file sweeps/<name>/0003.yaml   # one sweep point
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from heroes_sim import metrics
from heroes_sim.config import load_run_config, load_run_file
from heroes_sim.recorder import write_run
from heroes_sim.runner import run


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path, nargs="?")
    ap.add_argument("--run-file", type=Path, help="sweep run file (instead of a scenario)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--patient", type=Path, help="patient profile (default: the scenario's)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, help="default: runs/<scenario>_<patient>_s<seed>")
    ap.add_argument("--viewer", action="store_true", help="opt-in real-time viewer")
    args = ap.parse_args()

    if (args.scenario is None) == (args.run_file is None):
        ap.error("give either a scenario or --run-file")
    params: dict = {}
    if args.run_file is not None:
        if args.override or args.patient:
            ap.error("--run-file is fully resolved; --override/--patient do not apply")
        cfg, args.seed, params = load_run_file(args.run_file)
        default_out = args.run_file.parent / "results" / args.run_file.stem
    else:
        overrides = parse_overrides(args.override)
        cfg = load_run_config(args.scenario, args.base, overrides, args.patient)
        default_out = Path("runs") / f"{cfg.scenario.name}_{cfg.patient.name}_s{args.seed}"
    out = args.out or default_out

    wall0 = time.perf_counter()
    if args.viewer:
        import mujoco.viewer

        viewer = None

        def on_step(plant) -> None:
            nonlocal viewer
            if viewer is None:
                viewer = mujoco.viewer.launch_passive(plant.model, plant.data)
            viewer.sync()
            time.sleep(plant.dt)

        result = run(cfg, args.seed, on_step)
        if viewer is not None:
            viewer.close()
    else:
        result = run(cfg, args.seed)
    wall = time.perf_counter() - wall0

    log = result.log
    sim_s = float(log["t"].iloc[-1])
    if result.calibration is not None and result.calibration.log is not log:
        sim_s += float(result.calibration.log["t"].iloc[-1])
    print(
        f"{cfg.scenario.name} / {cfg.patient.name} / seed {args.seed}: "
        f"{sim_s:.1f} s sim in {wall:.2f} s ({sim_s / wall:.1f}x real time)"
    )
    summary: dict = {}
    if result.calibration is not None:
        summary["mvc_trials"] = {}
        for r in result.calibration.trials:
            summary["mvc_trials"][r.group] = {
                "activation": dict(zip(cfg.plant.muscles, r.activation.round(6).tolist())),
                "torque": dict(zip(cfg.plant.joints, r.torque.round(6).tolist())),
            }
            torques = ", ".join(f"{t:+.1f}" for t in r.torque)
            print(f"  MVC {r.group}: torque [{torques}] N m")
    if result.mvc is not None:
        names = [ch.name for ch in cfg.emg.channels]
        mvc = result.mvc
        # Same shape as scenario.calibrated, so it can be pasted back to skip calibration.
        summary["calibrated"] = {
            "envelope": dict(zip(names, mvc.envelope)),
            "rest": dict(zip(names, mvc.baseline.tolist())),
            "deadband": dict(zip(cfg.plant.joints, mvc.deadband or ())),
        }
        cal = summary["calibrated"]
        print(
            "  MVC envelope (rest): "
            + ", ".join(f"{n}={v:.4f} ({cal['rest'][n]:.4f})" for n, v in cal["envelope"].items())
        )
        if cal["deadband"]:
            print("  deadband: " + ", ".join(f"{j}={v:.4f}" for j, v in cal["deadband"].items()))

    events = pd.DataFrame([asdict(e) for e in result.events], columns=["t", "rule", "channel"])
    m = metrics.summarize(cfg, result, events)
    for key, v in m.items():
        if not key.startswith(("final_capacity_", "safety_")) or v:
            print(f"  {key}: {v:.4f}")
    flat = {"seed": args.seed, **params, **m, "sim_s": sim_s, "wall_s": wall}
    written = write_run(
        out,
        log,
        cfg,
        args.seed,
        summary,
        events if cfg.scenario.kind == "movement" else None,
        flat if cfg.scenario.kind == "movement" else None,
    )
    print(f"  wrote {written}")


if __name__ == "__main__":
    main()
