"""Run a sweep locally (no SLURM), N runs in parallel. Resumable: finished runs are skipped.

uv run python scripts/run_sweep.py sweeps/<name> --jobs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from heroes_sim.sweep import run_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--rerun", action="store_true", help="also rerun finished runs")
    args = ap.parse_args()

    todo = [
        f
        for f in run_files(args.sweep_dir)
        if args.rerun or not (args.sweep_dir / "results" / f.stem / "metrics.json").exists()
    ]
    print(f"{len(todo)} runs to do, {args.jobs} at a time")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def one(f: Path) -> tuple[Path, int]:
        log = args.sweep_dir / "slurm" / f"{f.stem}.local.out"
        with open(log, "w") as out:
            cmd = [sys.executable, "scripts/run.py", "--run-file", str(f)]
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, check=False)
            return f, proc.returncode

    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for f, code in pool.map(one, todo):
            print(f"  {f.stem}: {'ok' if code == 0 else f'FAILED ({code})'}")
            if code:
                failed.append(f.stem)
    if failed:
        sys.exit(f"{len(failed)} runs failed: {failed} (logs in {args.sweep_dir}/slurm/)")


if __name__ == "__main__":
    main()
