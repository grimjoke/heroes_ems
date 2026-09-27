"""One scenario config -> one run. Headless by default.

    uv run python scripts/run.py configs/scenarios/step_targets.yaml --seed 0
    uv run python scripts/run.py configs/scenarios/mvc_calibration.yaml \\
        --patient configs/patients/sci_c5.yaml --override patient.onset_delay_ms=0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from heroes_sim import metrics
from heroes_sim.config import MovementScenario, load_run_config
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
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--patient", type=Path, help="patient profile (default: the scenario's)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, help="default: runs/<scenario>_<patient>_s<seed>")
    ap.add_argument("--viewer", action="store_true", help="opt-in real-time viewer")
    args = ap.parse_args()

    cfg = load_run_config(args.scenario, args.base, parse_overrides(args.override), args.patient)
    out = args.out or Path("runs") / f"{cfg.scenario.name}_{cfg.patient.name}_s{args.seed}"

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
    extra = {}
    sim_s = float(log["t"].iloc[-1])
    print(
        f"{cfg.scenario.name} / {cfg.patient.name} / seed {args.seed}: "
        f"{sim_s:.1f} s sim in {wall:.2f} s ({sim_s / wall:.1f}x real time)"
    )
    if isinstance(cfg.scenario, MovementScenario):
        for j in cfg.plant.joints:
            if cfg.scenario.target is None:
                print(f"  {j}: max excursion {metrics.max_excursion(log, j):.4f} rad")
            else:
                print(f"  {j}: tracking RMSE {metrics.tracking_rmse(log, j, cfg.metrics):.4f} rad")
    else:
        extra["mvc"] = {}
        for r in result.mvc:
            extra["mvc"][r.group] = {
                "activation": dict(zip(cfg.plant.muscles, r.activation.round(6).tolist())),
                "torque": dict(zip(cfg.plant.joints, r.torque.round(6).tolist())),
            }
            acts = " ".join(f"{m}={a:.2f}" for m, a in zip(cfg.plant.muscles, r.activation))
            torques = ", ".join(f"{t:+.1f}" for t in r.torque)
            print(f"  {r.group}: torque [{torques}] N m; activation {acts}")
    print(f"  wrote {write_run(out, log, cfg, args.seed, extra)}")


if __name__ == "__main__":
    main()
