"""M0 demo: zero excitation, gravity on, forearm released from q0. Headless by default."""

from __future__ import annotations

import argparse
import time

import numpy as np

from heroes_sim.config import load_config
from heroes_sim.plant import Plant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/base.yaml")
    ap.add_argument("--viewer", action="store_true", help="opt-in interactive viewer")
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert cfg.passive_drop is not None, "config has no passive_drop section"
    plant = Plant(cfg.plant, cfg.timing)
    plant.reset(np.array([cfg.passive_drop.q0_rad]), np.zeros(1))
    zero = np.zeros(plant.n_muscles)
    n = int(cfg.passive_drop.duration_s / plant.dt)

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(plant.model, plant.data) as v:
            while v.is_running():
                t0 = time.perf_counter()
                plant.step(zero)
                v.sync()
                time.sleep(max(0.0, plant.dt - (time.perf_counter() - t0)))
        return

    t0 = time.perf_counter()
    for i in range(n):
        plant.step(zero)
        if i % (n // 6) == 0:
            print(f"t={i * plant.dt:5.2f}s  q={plant.joint_state()[0][0]:.4f} rad")
    wall = time.perf_counter() - t0
    q, qd = plant.joint_state()
    print(
        f"final q={q[0]:.4f} rad, qd={qd[0]:.2e}; {cfg.passive_drop.duration_s / wall:.0f}x real time"
    )


if __name__ == "__main__":
    main()
