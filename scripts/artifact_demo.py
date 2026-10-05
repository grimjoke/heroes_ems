"""M3 demo: stim artifact / M-wave feedback with and without blanking.

    uv run python scripts/artifact_demo.py                       # sci_c5, both scenarios
    uv run python scripts/artifact_demo.py --patient configs/patients/healthy.yaml

Runs each scenario with: clean EMG (M2 model), no blanking, zero-fill blanking (with the
envelope correction it requires), sample-and-hold blanking, interpolating blanking (the
default, D15), and interpolation plus the adaptive mains canceller (D17). Each variant is calibrated first under its own EMG model (the
clean model has a lower noise floor), as it would be on hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from heroes_sim import metrics
from heroes_sim.config import load_run_config
from heroes_sim.runner import calibrate, pinned, run

CLEAN = {
    f"emg.{c}.enabled": False for c in ("stim_artifact", "m_wave", "powerline", "baseline_wander")
}
VARIANTS = {
    "clean EMG (M2)": {**CLEAN, "controller.blanking_ms": 0},
    "no blanking": {"controller.blanking_ms": 0},
    "zero-fill + correction": {
        "controller.blanking_fill": "zero",
        "controller.blanking_correction": True,
    },
    "hold blanking": {"controller.blanking_fill": "hold"},
    "interp blanking": {},
    "interp + mains canceller": {
        "controller.mains_canceller": {
            "freqs_hz": [50.0, 100.0, 150.0],
            "mu": 0.002,
            "mu_bias": 0.02,
        }
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=Path, default=Path("configs/patients/sci_c5.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenarios = Path("configs/scenarios")
    for name in ("no_intent", "step_targets"):
        base = load_run_config(scenarios / f"{name}.yaml", patient_path=args.patient)
        j = base.plant.joints[0]
        print(f"\n{name} / {base.patient.name} / seed {args.seed}")
        print(
            f"  {'variant':<26} {'deadband':>8} {'ref drift':>9} {'track RMSE':>10} {'ref RMSE':>9} "
            f"{'dose bic':>8} {'dose tri':>8}  safety events"
        )
        for label, ov in VARIANTS.items():
            cfg = load_run_config(
                scenarios / f"{name}.yaml", overrides=ov, patient_path=args.patient
            )
            cal = calibrate(cfg, args.seed, "calibration/", None)
            ov = {**ov, "scenario.calibrated": pinned(cfg, cal)}
            cfg = load_run_config(
                scenarios / f"{name}.yaml", overrides=ov, patient_path=args.patient
            )
            res = run(cfg, args.seed)
            log = res.log
            if cfg.scenario.target is None:
                drift = f"{metrics.max_excursion(log, j, of='ref'):.3f}"
                track = ref = "-"
            else:
                drift = "-"
                track = f"{metrics.tracking_rmse(log, j, cfg.metrics):.3f}"
                ref = f"{metrics.tracking_rmse(log, j, cfg.metrics, of='ref'):.3f}"
            doses = [metrics.stim_dose(log, ch.name) for ch in cfg.stim.channels]
            counts: dict[str, int] = {}
            for e in res.events:
                counts[e.rule] = counts.get(e.rule, 0) + 1
            print(
                f"  {label:<26} {cal.mvc.deadband[0]:>8.3f} {drift:>9} {track:>10} {ref:>9} "
                f"{doses[0]:>8.2f} {doses[1]:>8.2f}  {counts or ''}"
            )


if __name__ == "__main__":
    main()
