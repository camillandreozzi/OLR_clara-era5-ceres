#!/bin/bash -l
# One-off setup on an Euler login node: Python environment, unit tests, task count.
#   bash euler/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source euler/modules.sh
python3 -c 'import sys; assert sys.version_info >= (3, 11), f"Python >= 3.11 needed, got {sys.version}"'
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r euler/requirements-euler.txt
pip freeze > euler/pip_freeze_euler.txt
python -m pytest tests -q
echo "array tasks: $(python src/run_cv.py --config "$CFG" --list-tasks | tail -n 1)"
