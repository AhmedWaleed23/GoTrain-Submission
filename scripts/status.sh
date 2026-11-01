#!/usr/bin/env bash
# Show whether a job is running, the lock holder, GPU use and the latest experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${GOTRAIN_PYTHON:-$(command -v python)}"
HOME_DIR="${GOTRAIN_HOME:-$PWD/gotrain_runs}"
echo "== lock"; cat "$HOME_DIR/locks/training.lock" 2>/dev/null || echo "no training lock (no job running)"
echo "== gotrain processes"; pgrep -af "python -m gotrain" || echo "none"
if command -v nvidia-smi >/dev/null 2>&1; then echo "== GPU"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; fi
echo "== experiments"; "$PY" -m gotrain list | tail -n 25
