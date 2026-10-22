#!/bin/bash -l
#SBATCH --job-name=olr-linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/linear_%j.out
# Regression-only baselines (no GP kernel), with and without ERA5: every CV fold, then the full-data fits.
# Seconds per fit, so one job runs them all (src/baselines.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source euler/modules.sh
source "$VENV/bin/activate"
python src/run_cv.py --config "$CFG" --model linear linear_noera5
python src/run_cv.py --config "$CFG" --model linear linear_noera5 --full
