# Software environment of the GP pipeline on Euler; sourced by every script and job.
# Python >= 3.11 is required (tomllib). If these module names do not exist, list the available ones with
#   module spider python
# and set EULER_MODULES before running, e.g.  export EULER_MODULES="stack/2024-06 python/3.11.6"
EULER_MODULES=${EULER_MODULES:-"stack/2024-06 gcc/12.2.0 python/3.11.6"}
module load $EULER_MODULES
export VENV=${VENV:-$HOME/venvs/olr-gp}
export CFG=${CFG:-config/pipeline_euler.toml}
# one BLAS / OpenMP thread pool per allocated core
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
