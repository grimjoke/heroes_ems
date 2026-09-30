#!/bin/bash
# One sweep point per SLURM array task: SLURM_ARRAY_TASK_ID -> sweeps/<name>/NNNN.yaml.
#
#   uv run python scripts/make_sweep.py configs/sweeps/<spec>.yaml     # prints the line below
#   sbatch --array=0-<N-1> --output=sweeps/<name>/slurm/%a.out scripts/slurm/array.sbatch sweeps/<name>
#
# Results: sweeps/<name>/results/NNNN/ ; then scripts/aggregate.py sweeps/<name>.
# Submit from the repo root. Needs `uv` on the node and the environment synced once
# (`uv sync --frozen`) on a shared filesystem. Headless: no display or GPU needed.
#SBATCH --job-name=heroes-sweep
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

set -euo pipefail
SWEEP_DIR=${1:?usage: sbatch --array=0-N scripts/slurm/array.sbatch sweeps/<name>}
: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
RUN_FILE=$(printf "%s/%04d.yaml" "$SWEEP_DIR" "$SLURM_ARRAY_TASK_ID")
[[ -f "$RUN_FILE" ]] || { echo "no run file $RUN_FILE" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_FILE"
uv run --frozen python scripts/run.py --run-file "$RUN_FILE"
