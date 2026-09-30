"""Grid spec -> sweeps/<name>/0000.yaml ... (self-contained run files) + index.csv.

uv run python scripts/make_sweep.py configs/sweeps/example.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from heroes_sim.sweep import load_spec, make_sweep, run_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--root", type=Path, default=Path("sweeps"))
    ap.add_argument("--force", action="store_true", help="overwrite existing run files")
    args = ap.parse_args()

    out = make_sweep(load_spec(args.spec), args.root, args.force)
    n = len(run_files(out))
    print(f"wrote {n} run files to {out}/")
    print("run on SLURM:")
    print(
        f"  sbatch --array=0-{n - 1} --output={out}/slurm/%a.out scripts/slurm/array.sbatch {out}"
    )
    print("or locally:")
    print(f"  uv run python scripts/run_sweep.py {out} --jobs 4")
    print("then:")
    print(f"  uv run python scripts/aggregate.py {out}")


if __name__ == "__main__":
    main()
