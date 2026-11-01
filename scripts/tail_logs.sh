#!/usr/bin/env bash
# Follow the most recent background log. Usage: scripts/tail_logs.sh [name]
set -euo pipefail
cd "$(dirname "$0")/.."
HOME_DIR="${GOTRAIN_HOME:-$PWD/gotrain_runs}"
PATTERN="${1:-}"
LOG=$(ls -t "$HOME_DIR"/logs/${PATTERN}*.log 2>/dev/null | head -1 || true)
if [ -z "$LOG" ]; then echo "no logs under $HOME_DIR/logs"; exit 1; fi
echo "== $LOG"
tail -n 50 -f "$LOG"
