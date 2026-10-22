#!/bin/bash -l
#SBATCH --job-name=olr-gp-fit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fit_%A_%a.out
# One GP fit per array task: tasks 0-2 are the full-data fits, the rest one kernel on one CV fold
# (python src/run_cv.py --list-tasks). Finished fits are skipped, so the array can simply be resubmitted.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source euler/modules.sh
source "$VENV/bin/activate"
python src/run_cv.py --config "$CFG" --task-index "$SLURM_ARRAY_TASK_ID" --verbose
