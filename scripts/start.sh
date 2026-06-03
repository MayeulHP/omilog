#!/usr/bin/env bash
# Run omilog locally with --reload. Any extra args go straight to uvicorn.
#
# Examples:
#   ./scripts/start.sh                     # 127.0.0.1:8000, reload on
#   ./scripts/start.sh --port 9000         # different port
#   ./scripts/start.sh --host 0.0.0.0      # bind all interfaces (only do this on tailnet)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo ".venv missing — run ./scripts/setup.sh first" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo ".env missing — run ./scripts/setup.sh first" >&2
  exit 1
fi

# Defaults; user can override by passing flags after `start.sh`.
exec .venv/bin/uvicorn omilog.main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
