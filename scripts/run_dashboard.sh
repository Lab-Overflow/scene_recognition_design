#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STATE_FILE="${STATE_FILE:-$ROOT/logs/latest_state.json}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8787}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

CMD=("$PYTHON_BIN" scripts/scene_dashboard.py --state-file "$STATE_FILE" --host "$DASHBOARD_HOST" --port "$DASHBOARD_PORT")
printf 'Running: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
