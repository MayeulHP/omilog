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
# this, optional extras the user installed manually would get uninstalled
# every launch.
#
# Auto-detect which extras to enable based on .env opt-ins, so the user
# doesn't have to remember `uv sync --extra <name>` after every git pull.
sync_extras=()
if grep -qiE '^OMILOG_DIARIZATION_ENABLED=(true|1|yes|on|t)$' .env 2>/dev/null; then
  sync_extras+=(--extra diarization)
fi

if command -v uv >/dev/null 2>&1; then
  if [[ ${#sync_extras[@]} -gt 0 ]]; then
    echo "▸ syncing with extras: ${sync_extras[*]}"
    uv sync --inexact "${sync_extras[@]}" --quiet 2>/dev/null \
      || uv sync --inexact "${sync_extras[@]}"
  else
    uv sync --inexact --quiet 2>/dev/null || uv sync --inexact
  fi
else
  echo "▸ uv not installed; skipping dep sync (run setup.sh after pyproject changes)" >&2
fi

# sherpa-onnx's aarch64 wheel dlopens libonnxruntime.so by bare name. The
# wheel bundles an ABI-matched libonnxruntime.so.<ver> under sherpa_onnx/lib/
# but doesn't put that dir on the dynamic linker's search path. We
# (a) create the bare-name symlink so dlopen resolves it and (b) add the
# dir to LD_LIBRARY_PATH. Crucially: we point at sherpa-onnx's bundled
# copy, not the pip `onnxruntime` package, because sherpa-onnx's C extension
# is built against a specific onnxruntime version and a newer pip wheel
# will fail with "version `VERS_1.X` not found." On platforms where the
# wheel already works this block is a silent no-op (find returns nothing).
if [[ -d .venv ]]; then
  SHERPA_LIB=$(find .venv -path '*/sherpa_onnx/lib' -type d 2>/dev/null | head -1)
  if [[ -n "$SHERPA_LIB" ]]; then
    if [[ ! -e "$SHERPA_LIB/libonnxruntime.so" ]]; then
      versioned=$(find "$SHERPA_LIB" -maxdepth 1 -name 'libonnxruntime.so.*' \
                    -type f 2>/dev/null | head -1)
      if [[ -n "$versioned" ]]; then
        ln -sf "$(basename "$versioned")" "$SHERPA_LIB/libonnxruntime.so"
        echo "▸ linked $SHERPA_LIB/libonnxruntime.so → $(basename "$versioned")"
      fi
    fi
    export LD_LIBRARY_PATH="${SHERPA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
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
