#!/bin/bash -l
# Submit the period-specific ERA5 slope sensitivity (src/sensitivity_gapslope.py) into results/model/, next to the
# paper's fits, so the variants reuse every fold's cached inducing inputs and pair exactly with the current models:
#   regression only with the slope (linear_gapslope)        one small job
#   <winner>_gapslope ([baselines] gp_noera5_kernels)       inducing inputs (cached) -> one array task per fit
#   then the summary (results/model/gapslope_*.csv)
#   bash euler/submit_sensitivity.sh                 (MAX_PARALLEL=40 bash euler/submit_sensitivity.sh to cap tasks)
# Finished fits are skipped, so the script can simply be run again.
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/modules.sh
source "$VENV/bin/activate"
# Validate inputs before submitting even the regression job.
python - <<'PY'
import os
import sys

sys.path.insert(0, "src")
import config

cfg = config.load(os.environ["CFG"])
data = config.PROCESSED_DIR / f"harmonised_{cfg['grid']['res_deg']}deg_monthly.parquet"
if not data.is_file():
    raise SystemExit(f"Missing input: {data}. Copy the existing data and folds before submitting.")
PY
N=$(python src/run_cv.py --config "$CFG" --list-tasks --group sensitivity | tail -n 1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -eq 0 ]; then
  echo "No sensitivity tasks found; no jobs submitted." >&2
  exit 1
fi
mkdir -p logs
LIN=$(sbatch --parsable --job-name=olr-gapslope-linear --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G \
      --time=01:00:00 --output=logs/gapslope_linear_%j.out \
      --wrap="bash -lc 'source euler/modules.sh && source \"\$VENV/bin/activate\" && python src/sensitivity_gapslope.py --config \"\$CFG\" --fit linear_gapslope'")
PREP=$(sbatch --parsable euler/prepare.sbatch)
FIT=$(sbatch --parsable --dependency=afterok:"$PREP" --array=0-$((N - 1))%"${MAX_PARALLEL:-150}" \
      --export=ALL,GROUP=sensitivity euler/fit_array.sbatch)
SUM=$(sbatch --parsable --dependency=afterany:"$LIN:$FIT" --job-name=olr-gapslope-summary --ntasks=1 \
      --cpus-per-task=8 --mem-per-cpu=4G --time=02:00:00 --output=logs/gapslope_summary_%j.out \
      --wrap="bash -lc 'source euler/modules.sh && source \"\$VENV/bin/activate\" && python src/sensitivity_gapslope.py --config \"\$CFG\" --summarise'")
echo "linear_gapslope $LIN; inducing inputs $PREP -> $N fits (array $FIT) -> summary $SUM"
echo "watch: squeue -u \$USER    logs: logs/    then copy back results/model/gapslope_*.csv (and the fits if wanted)"
