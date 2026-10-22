#!/bin/bash -l
# Submit the baselines of the 2 x 2 ablation (src/baselines.py), scored on the same folds as the kernels:
#   regression only (linear, linear_noera5)        one small job
#   GP without ERA5 ([baselines] gp_noera5_kernels) inducing inputs (cached) -> one array task per fit
#   then the summary (model_comparison.csv, model_pairwise.csv)
#   bash euler/submit_baselines.sh                 (MAX_PARALLEL=40 bash euler/submit_baselines.sh to cap tasks)
# Safe while the kernel array runs: the GPs without ERA5 have their own task list (--group baselines) and
# every baseline writes to its own results/model/<model>/ folder. Finished fits are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/modules.sh
source "$VENV/bin/activate"
mkdir -p logs
LIN=$(sbatch --parsable euler/linear.sbatch)
echo "regression-only baselines: job $LIN"
DEPS=$LIN
N=$(python src/run_cv.py --config "$CFG" --list-tasks --group baselines | tail -n 1)
if [ "$N" -gt 0 ]; then
  PREP=$(sbatch --parsable euler/prepare.sbatch)
  FIT=$(sbatch --parsable --dependency=afterok:"$PREP" --array=0-$((N - 1))%"${MAX_PARALLEL:-150}" \
        --export=ALL,GROUP=baselines euler/fit_array.sbatch)
  DEPS=$DEPS:$FIT
  echo "GP without ERA5: inducing inputs $PREP -> $N fits (array $FIT)"
else
  echo "GP without ERA5: not submitted, [baselines] gp_noera5_kernels is empty in $CFG"
  if [ -f results/model/kernel_comparison.csv ]; then
    echo "  current winner: $(sed -n 2p results/model/kernel_comparison.csv | cut -d, -f1); set" \
         "gp_noera5_kernels = [\"<winner>\"] and run this script again"
  fi
fi
SUM=$(sbatch --parsable --dependency=afterany:"$DEPS" euler/summary.sbatch)
echo "summary $SUM after $DEPS    watch: squeue -u \$USER    logs: logs/"
