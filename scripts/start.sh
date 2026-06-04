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

# First-run bootstrap: if either .venv or .env is missing, hand off to setup.sh.
# This makes start.sh the only entry point the user ever needs.
if [[ ! -d .venv ]] || [[ ! -f .env ]]; then
  echo "▸ first run detected — running ./scripts/setup.sh"
  ./scripts/setup.sh
fi

# Keep deps fresh on every launch — picks up pyproject.toml changes after a
# git pull without a separate manual step. Idempotent, fast with uv.
#
# --inexact: don't remove packages that aren't in the default deps. Without
# this, optional extras (diarization, etc.) that the user installed manually
# via `uv sync --extra <name>` would get uninstalled on every start, which
# is surprising and annoying. Trade-off: the venv may keep packages from
# removed pyproject deps, but that's strictly safer than the alternative.
if command -v uv >/dev/null 2>&1; then
  uv sync --inexact --quiet 2>/dev/null || uv sync --inexact
else
  echo "▸ uv not installed; skipping dep sync (run setup.sh after pyproject changes)" >&2
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
