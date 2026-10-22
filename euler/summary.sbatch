#!/bin/bash -l
#SBATCH --job-name=olr-gp-summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/summary_%j.out
# Collect the per-fold predictions into CSVs, compute metrics and the kernel comparison, list missing fits.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source euler/modules.sh
source "$VENV/bin/activate"
python src/run_cv.py --config "$CFG" --summarise
