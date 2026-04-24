#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROVIDER="${1:-none}"  # none | openai | anthropic | stub
MODEL="${2:-}"
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

mkdir -p "$(dirname "$STATE_FILE")"

# Start dashboard in background.
"$PYTHON_BIN" scripts/scene_dashboard.py --state-file "$STATE_FILE" --host "$DASHBOARD_HOST" --port "$DASHBOARD_PORT" >/tmp/scene_dashboard.log 2>&1 &
DASH_PID=$!

cleanup() {
  if kill -0 "$DASH_PID" >/dev/null 2>&1; then
    kill "$DASH_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Dashboard: http://$DASHBOARD_HOST:$DASHBOARD_PORT"
echo "(dashboard logs: /tmp/scene_dashboard.log)"

export STATE_FILE

if [[ -n "${FRAME_URL:-}" ]]; then
  export PROVIDER
  export MODEL
  bash scripts/run_live_demo_url.sh
else
  if [[ -n "$MODEL" ]]; then
    bash scripts/run_live_demo.sh "$PROVIDER" "$MODEL"
  else
    bash scripts/run_live_demo.sh "$PROVIDER"
  fi
fi
