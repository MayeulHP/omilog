#!/usr/bin/env bash
# Run omilog under tmux so it survives SSH disconnects without needing
# systemd. Re-running this script while a session already exists just
# re-attaches; it never starts a second copy.
#
# Usage:
#   ./scripts/start_tmux.sh                # start (if needed) + attach
#   ./scripts/start_tmux.sh --detach       # start (if needed), don't attach
#   ./scripts/start_tmux.sh --kill         # stop the omilog session
#   ./scripts/start_tmux.sh --status       # report whether it's running
#
# Anything else after the first arg is passed through to start.sh, e.g.
#   ./scripts/start_tmux.sh -- --port 9000
#
# Inside the session: C-b d to detach (omilog keeps running), C-b c for
# a new shell window, C-c inside the omilog window to stop the server.
set -euo pipefail

SESSION="${OMILOG_TMUX_SESSION:-omilog}"
cd "$(dirname "$0")/.."

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed — install it (apt install tmux / brew install tmux)" >&2
  exit 1
fi

case "${1:-}" in
  --kill)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux kill-session -t "$SESSION"
      echo "▸ killed tmux session '$SESSION'"
    else
      echo "▸ no tmux session '$SESSION' running"
    fi
    exit 0
    ;;
  --status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "▸ tmux session '$SESSION' is running"
      tmux list-windows -t "$SESSION"
      exit 0
    else
      echo "▸ tmux session '$SESSION' is NOT running"
      exit 1
    fi
    ;;
esac

detach=0
if [[ "${1:-}" == "--detach" ]]; then
  detach=1
  shift
fi
# Swallow an optional `--` so `start_tmux.sh -- --port 9000` works and
# anything else just falls through directly.
[[ "${1:-}" == "--" ]] && shift

# Already running? Don't double-start; just attach (or print + exit if
# the user asked for detach).
if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ $detach -eq 1 ]]; then
    echo "▸ tmux session '$SESSION' already running (use --kill to stop it)"
    exit 0
  fi
  echo "▸ attaching to existing tmux session '$SESSION' (C-b d to detach)"
  exec tmux attach-session -t "$SESSION"
fi

# Build the start.sh command with any extra args quoted-safe for tmux.
start_cmd="./scripts/start.sh"
for arg in "$@"; do
  start_cmd+=" $(printf '%q' "$arg")"
done

# Trailing `; read` keeps the pane open if start.sh exits (crash, port
# in use, etc.) so the user sees the error instead of a vanished session.
tmux new-session -d -s "$SESSION" -n omilog \
  "bash -lc '$start_cmd; echo; echo \"▸ start.sh exited — press enter to close pane\"; read'"
echo "▸ started tmux session '$SESSION'"

if [[ $detach -eq 1 ]]; then
  echo "  to attach later:  $0"
  echo "  to stop:          $0 --kill"
  exit 0
fi

exec tmux attach-session -t "$SESSION"
