#!/usr/bin/env bash
# Launch a GoTrain command detached from the browser session (survives JupyterLab disconnects).
# Usage: scripts/run_background.sh <name> <gotrain args...>
#   e.g. scripts/run_background.sh baseline baseline --task all
#        scripts/run_background.sh agent agent --task all --session-minutes 90
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${GOTRAIN_PYTHON:-$(command -v python)}"
HOME_DIR="${GOTRAIN_HOME:-$PWD/gotrain_runs}"
NAME="${1:?name}"; shift
mkdir -p "$HOME_DIR/logs"
LOCK="$HOME_DIR/locks/training.lock"
if [ -f "$LOCK" ]; then
  PID=$(sed -n 's/.*"pid": *\([0-9]*\).*/\1/p' "$LOCK" | head -1)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "A training job is already running (pid $PID, lock $LOCK). Not starting a duplicate." >&2
    exit 1
  fi
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$HOME_DIR/logs/${NAME}_${STAMP}.log"
nohup "$PY" -m gotrain "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$HOME_DIR/logs/${NAME}_${STAMP}.pid"
echo "started pid $PID: python -m gotrain $*"
echo "log: $LOG"
echo "follow with: tail -f $LOG"
