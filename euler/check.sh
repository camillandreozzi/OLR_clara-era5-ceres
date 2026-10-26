#!/bin/bash -l
# Has the Euler run finished? Read-only; run on a login node:
#   bash euler/check.sh                 (DAYS=14 bash euler/check.sh to look further back in sacct)
#   1. jobs of this pipeline still queued or running
#   2. Slurm states of the last DAYS days, with the log of every job that did not complete
#   3. fits whose output file is missing, per task group, with the sbatch line that reruns exactly those
#      tasks (the output files are the truth: a task that failed and was rerun shows up in 2 but not here)
#   4. summary tables, and whether they are older than the newest fit
set -uo pipefail
cd "$(dirname "$0")/.."
source euler/modules.sh
source "$VENV/bin/activate"
JOBS=olr-gp-prep,olr-gp-fit,olr-gp-summary,olr-linear

echo "== 1. queue"
QUEUED=$(squeue -h -u "$USER" --name="$JOBS" | wc -l)
if [ "$QUEUED" -gt 0 ]; then
  squeue -u "$USER" --name="$JOBS" --format="%.20i %.16j %.10T %.10M %.25R"
else
  echo "nothing queued or running"
fi

echo
echo "== 2. sacct, last ${DAYS:-7} days (one line per array task; a pending range of tasks is one line)"
SACCT=$(sacct -u "$USER" -S "now-${DAYS:-7}days" --name="$JOBS" -X -n -P --format=JobID,JobName,State,Elapsed)
echo "$SACCT" | awk -F'|' 'NF {split($3, s, " "); n[$2 " " s[1]]++} END {for (k in n) printf "%6d  %s\n", n[k], k}' | sort -k2
BAD=$(echo "$SACCT" | awk -F'|' 'NF && $3 !~ /^(COMPLETED|RUNNING|PENDING)/')
if [ -n "$BAD" ]; then
  echo "did not complete (job, state, elapsed, log, first error line):"
  echo "$BAD" | while IFS='|' read -r id name state elapsed; do
    case $name in
      olr-gp-fit)     log=logs/fit_$id.out ;;
      olr-gp-prep)    log=logs/prep_$id.out ;;
      olr-gp-summary) log=logs/summary_$id.out ;;
      olr-linear)     log=logs/linear_$id.out ;;
    esac
    err=$(grep -m1 -E "Error|error:|Killed|oom|TIME LIMIT|CANCELLED" "$log" 2>/dev/null)
    printf "  %-14s %-16s %-10s %s\n      %s\n" "$id" "${state%% *}" "$elapsed" "$log" "${err:-(no error line)}"
  done
fi

echo
echo "== 3. outputs"
QUEUED=$QUEUED python - <<'EOF'
import os
import sys

sys.path.insert(0, "src")
import baselines
import config
import run_cv

cfg = config.load(os.environ["CFG"])
config.MODEL_DIR = config.RESULTS_DIR / cfg.get("output", {}).get("model_dir", "model")
folds = run_cv.check_folds(cfg)


def ranges(ids):
    """[0, 1, 2, 5, 7, 8] -> '0-2,5,7-8' (an sbatch --array list)."""
    out, start = [], ids[0]
    for a, b in zip(ids, ids[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}-{a}" if a > start else f"{a}")
            start = b
    return ",".join(out)


n_missing = 0
for group in run_cv.GROUPS:
    tasks = run_cv.task_list(cfg, folds, group)
    if not tasks:
        print(f"{group}: no tasks ([baselines] gp_noera5_kernels is empty)")
        continue
    missing = run_cv.completeness(cfg, list(dict.fromkeys(k for k, _, _ in tasks)))
    gone = set(missing.itertuples(index=False, name=None))
    ids = [i for i, t in enumerate(tasks) if t in gone]
    n_missing += len(ids)
    print(f"{group}: {len(tasks) - len(ids)} / {len(tasks)} fits done")
    if ids:
        print(missing.groupby(["kernel", "cv_type"]).size().to_string())
        env = "" if group == "kernels" else f"GROUP={group} "
        print(f"  rerun: {env}sbatch --array={ranges(ids)} euler/fit_array.sbatch")
missing = run_cv.completeness(cfg, list(baselines.LINEAR))
n_missing += len(missing)
print(f"regression-only baselines: {len(missing)} fits missing" + ("  rerun: sbatch euler/linear.sbatch" if len(missing) else ""))

print("\n== 4. summary")
fits = [p for p in config.MODEL_DIR.glob("*/full_model.pt")] + list(config.MODEL_DIR.glob("*/folds/*_[0-9][0-9].parquet"))
newest = max((p.stat().st_mtime for p in fits), default=0)
stale = False
for name in ("kernel_comparison.csv", "model_comparison.csv"):
    p = config.MODEL_DIR / name
    if not p.exists():
        print(f"{name}: missing")
        stale = True
    elif p.stat().st_mtime < newest:
        print(f"{name}: older than the newest fit")
        stale = True
    else:
        print(f"{name}: up to date")
comp = config.MODEL_DIR / "kernel_comparison.csv"
if comp.exists():
    print(f"winner (first row of kernel_comparison.csv): {comp.read_text().splitlines()[1].split(',')[0]}")

print()
if int(os.environ["QUEUED"]):
    print("NOT FINISHED: jobs still queued or running (missing fits above may just be waiting)")
elif n_missing:
    print(f"NOT FINISHED: {n_missing} fits missing and nothing queued; rerun them (lines above), then the summary")
elif stale:
    print("FITS DONE, summary out of date: sbatch euler/summary.sbatch")
else:
    print("ALL DONE: every fit finished and the summary is up to date")
EOF
