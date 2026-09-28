"""One scenario config -> one run. Headless by default.

    uv run python scripts/run.py configs/scenarios/step_targets.yaml --seed 0
    uv run python scripts/run.py configs/scenarios/step_targets.yaml \\
        --patient configs/patients/sci_c5.yaml --override controller.deadband=0.1
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
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
        summary["mvc_envelope"] = dict(zip(names, result.mvc.envelope))
        summary["mvc_rest"] = dict(zip(names, result.mvc.baseline.tolist()))
        print(
            "  MVC envelope (rest): "
            + ", ".join(
                f"{n}={v:.4f} ({summary['mvc_rest'][n]:.4f})"
                for n, v in summary["mvc_envelope"].items()
            )
        )

    events = pd.DataFrame([asdict(e) for e in result.events], columns=["t", "rule", "channel"])
    sc = cfg.scenario
    if isinstance(sc, MovementScenario):
        m: dict = {}
        for j in cfg.plant.joints:
            if sc.target is None:
                m[f"max_excursion_{j}"] = metrics.max_excursion(log, j)
            else:
                m[f"tracking_rmse_{j}"] = metrics.tracking_rmse(log, j, cfg.metrics)
            if sc.closed_loop:
                if sc.target is None:
                    m[f"ref_max_excursion_{j}"] = metrics.max_excursion(log, j, of="ref")
                    m[f"ref_drift_rate_{j}"] = metrics.reference_drift_rate(log, j)
                else:
                    m[f"ref_rmse_{j}"] = metrics.tracking_rmse(log, j, cfg.metrics, of="ref")
        if sc.closed_loop:
            for ch in cfg.stim.channels:
                m[f"stim_dose_{ch.name}"] = metrics.stim_dose(log, ch.name)
                m[f"time_at_cap_{ch.name}"] = metrics.time_at_cap(
                    log, ch.name, cfg.safety.cap[ch.name]
                )
            m["safety_events"] = metrics.safety_event_counts(events)
        summary["metrics"] = m
        for key, v in m.items():
            print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")
    written = write_run(
        out, log, cfg, args.seed, summary, events if sc.kind == "movement" else None
    )
    print(f"  wrote {written}")


if __name__ == "__main__":
    main()
