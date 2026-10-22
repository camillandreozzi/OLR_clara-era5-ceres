#!/bin/bash -l
# Submit the whole run from an Euler login node:
#   inducing inputs  ->  one array task per fit  ->  summary
#   bash euler/submit.sh                 (MAX_PARALLEL=40 bash euler/submit.sh to cap running tasks)
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/modules.sh
source "$VENV/bin/activate"
mkdir -p logs
N=$(python src/run_cv.py --config "$CFG" --list-tasks | tail -n 1)
PREP=$(sbatch --parsable euler/prepare.sbatch)
FIT=$(sbatch --parsable --dependency=afterok:"$PREP" --array=0-$((N - 1))%"${MAX_PARALLEL:-150}" euler/fit_array.sbatch)
SUM=$(sbatch --parsable --dependency=afterany:"$FIT" euler/summary.sbatch)
echo "submitted: inducing inputs $PREP -> $N fits (array $FIT) -> summary $SUM"
echo "watch: squeue -u \$USER    logs: logs/    afterwards: sacct -j $FIT --format=JobID,State,Elapsed,MaxRSS"
