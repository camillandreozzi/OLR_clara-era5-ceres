#!/bin/bash -l
#SBATCH --job-name=olr-gp-bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/bench_%A_%a.out
# Sparse-GP checks (src/benchmark_gp.py), one kernel per array task: VFE vs the exact GP on a 3,000-row subset,
# then M sensitivity on the median-sized spatial fold. M = 1000 (the CV setting) gets the 3 starts of the CV
# fits, the other M one start; M = 2000 tests whether 1000 was enough. All three kernels, so that the ranking
# can be compared across M. Writes results/model/benchmark/{vfe_vs_exact,m_sensitivity}_<kernel>.csv.
#   sbatch euler/benchmark.sbatch                  (sbatch --array=0 euler/benchmark.sbatch: exponential only)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source euler/modules.sh
source "$VENV/bin/activate"
KERNELS=(exponential gaussian matern32)
python src/benchmark_gp.py --config "$CFG" --kernels "${KERNELS[$SLURM_ARRAY_TASK_ID]}" --m-values 250 500 1000 2000
