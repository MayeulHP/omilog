#!/usr/bin/env bash
# Run omilog locally with --reload. Host/port come from .env (OMILOG_HOST,
# OMILOG_PORT); any extra args pass through to uvicorn.
#
# Why we don't `source .env`: a bcrypt hash like $2b$12$… gets mangled by the
# shell (positional params $2 / $12 eaten). So we pluck just the two values
# we need with grep, never letting the rest near a shell parser.
#
# Examples:
#   ./scripts/start.sh                     # uses OMILOG_HOST / OMILOG_PORT
#   ./scripts/start.sh --port 9000         # override port
#   ./scripts/start.sh --no-reload         # not a real flag — drop --reload by setting RELOAD=0
#
# Environment overrides (take precedence over .env):
#   RELOAD=0  ./scripts/start.sh           # disable hot reload
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

# Pluck HOST/PORT from .env without sourcing it.
env_get() {
  local key="$1" default="$2"
  local v
  v=$(grep -E "^${key}=" .env | head -1 | cut -d= -f2- || true)
  printf '%s' "${v:-$default}"
}

HOST=$(env_get OMILOG_HOST 127.0.0.1)
PORT=$(env_get OMILOG_PORT 8000)
RELOAD="${RELOAD:-1}"

args=(--host "$HOST" --port "$PORT")
[[ "$RELOAD" == "1" ]] && args+=(--reload)

echo "▸ omilog uvicorn on $HOST:$PORT (reload=$RELOAD)"
exec .venv/bin/uvicorn omilog.main:app "${args[@]}" "$@"
