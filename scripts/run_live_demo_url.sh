#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Auto-load local env files for one-command startup.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

if [[ -z "${FRAME_URL:-}" ]]; then
  echo "FRAME_URL is required, example: FRAME_URL=http://127.0.0.1:8080/frame.jpg"
  exit 1
fi

PROVIDER="${PROVIDER:-none}"       # none | openai | anthropic | stub
MODEL="${MODEL:-}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1.0}"
DETECTOR="${DETECTOR:-yolo}"       # yolo | none
STATE_FILE="${STATE_FILE:-$ROOT/logs/latest_state.json}"
LOG_JSONL="${LOG_JSONL:-}"
MAX_FRAMES="${MAX_FRAMES:-0}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

CMD=("$PYTHON_BIN" scripts/live_scene_demo.py
  --frame-url "$FRAME_URL"
  --detector "$DETECTOR"
  --provider "$PROVIDER"
  --sample-interval "$SAMPLE_INTERVAL"
  --state-file "$STATE_FILE"
  --max-frames "$MAX_FRAMES"
)

if [[ -n "$MODEL" ]]; then
  CMD+=(--model "$MODEL")
fi
if [[ -n "$LOG_JSONL" ]]; then
  CMD+=(--log-jsonl "$LOG_JSONL")
fi

printf 'Running: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
