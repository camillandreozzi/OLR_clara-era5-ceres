#!/bin/bash -l
#SBATCH --job-name=olr-gp-prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep_%j.out
# k-means inducing inputs of every fold and the full fit, computed once so the array tasks never race.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source euler/modules.sh
source "$VENV/bin/activate"
python src/run_cv.py --config "$CFG" --prepare-inducing
