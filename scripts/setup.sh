#!/usr/bin/env bash
# One-shot bootstrap: venv + deps + .env. Idempotent — safe to re-run.
#
# Usage:
#   ./scripts/setup.sh
set -euo pipefail

# Always run relative to the repo root.
cd "$(dirname "$0")/.."

step() { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }

# 1) Python venv + deps. First-time install only. After this initial pass,
#    ./scripts/start.sh runs uv sync each launch to pick up pyproject changes
#    after a `git pull`.
if [[ -d .venv ]]; then
  ok ".venv already exists (start.sh will keep deps fresh)"
else
  if command -v uv >/dev/null 2>&1; then
    step "uv detected → uv sync --extra dev"
    uv sync --extra dev
  else
    step "uv not found → creating .venv via python3 -m venv"
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e ".[dev]"
  fi
  ok "venv ready"
fi

# 2) .env
if [[ -f .env ]]; then
  ok ".env already exists (leaving alone — delete it to regenerate)"
else
  step "bootstrapping .env"
  .venv/bin/python scripts/_bootstrap_env.py
fi

# 3) Sanity: import the app
step "import check"
.venv/bin/python -c "from omilog.main import app; print('import ok')"

echo
ok "Setup done. Start the server with:  ./scripts/start.sh"
