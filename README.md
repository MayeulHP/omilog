# omilog

Self-hosted personal conversation capture. Omi necklace → phone (Chronicle APK) → Pi (this backend) → GPU cluster (whisper.cpp + llama.cpp) → SQLite-backed transcripts, summaries, ICS events.

This repo is **Phase 0**: just the backend surface needed for the Chronicle APK to log in, pass its connection check, and dump Opus audio frames into `storage/`. No transcription, LLM, or web UI yet — those land in Phases 1–2.

The full architecture and phasing live in the spec notes (not in-repo); this README only covers what's needed to run what exists.

## Quick start (macOS dev)

```bash
./scripts/setup.sh    # creates .venv + .env (prompts for password)
./scripts/start.sh    # uvicorn with --reload on 127.0.0.1:8000
```

`setup.sh` picks `uv` if it's installed, otherwise falls back to `python3 -m venv`. `.env` is written from inside Python so the bcrypt hash never goes near a shell (a `$2b$12$…` hash gets silently mangled if shell-sourced).

Manual route if you don't want the scripts: copy `.env.template` to `.env`, fill `OMILOG_PASSWORD_HASH` (run `.venv/bin/python scripts/hash_password.py`) and `OMILOG_JWT_SECRET` (`python -c 'import secrets; print(secrets.token_urlsafe(64))'`), then `.venv/bin/uvicorn omilog.main:app --reload`.

Verify:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -X POST http://127.0.0.1:8000/auth/jwt/login \
  -d 'username=mayeul' -d 'password=YOURPASSWORD'
# {"access_token":"...","token_type":"bearer"}
```

Run the tests:

```bash
uv run pytest
```

## What's implemented

| Surface | Status |
| --- | --- |
| `POST /auth/jwt/login` (form-encoded) | ✅ |
| `GET /health`, `GET /readiness` | ✅ |
| `WS /ws?codec=opus` (JWT via header or `?token=`) | ✅ — writes raw audio payload to `storage/{session_id}.opus`, logs an `AudioSession` row |
| `GET /api/conversations` | ✅ stub (returns `[]`) |
| `GET/POST /api/clients` and friends | ✅ permissive stubs (log + return empty/ok) |
| Transcription / extraction pipeline | ⛔ Phase 1+ |
| Web UI | ⛔ Phase 1+ |

## WebSocket framing — known unknown

Chronicle uses the Wyoming protocol over WebSocket, but the exact mapping (text-frame-then-binary vs. fully framed binary) was reconstructed from Chronicle's CLAUDE.md and **not yet verified against a live capture**. `api/audio_ws.py` is intentionally permissive: it accepts JSON text frames with `audio-start` / `audio-chunk` / `audio-stop` events and appends any binary frame contents to the session file.

Before locking the parser down, run mitmproxy on the phone against a throwaway backend for one boot cycle and capture the actual shapes. Budget: ~30 min.

## Pi deploy (later)

`deploy/omilog.service` and `caddy/Caddyfile` are sketches for the eventual Pi deploy. They're not exercised in Phase 0 — leave them alone unless you're actually deploying.

## Layout

```
src/omilog/
  main.py        FastAPI app + lifespan
  config.py      pydantic-settings (.env)
  auth.py        bcrypt + JWT
  db.py          SQLite engine, WAL pragma
  models.py      AudioSession (only Phase 0 table)
  api/
    auth.py            /auth/jwt/login
    health.py          /health, /readiness
    audio_ws.py        WS /ws
    conversations.py   stub /api/conversations
    stubs.py           permissive device-registration stubs
storage/           gitignored audio dump
deploy/, caddy/    Pi deploy sketches (unused in Phase 0)
scripts/           hash_password.py
tests/             smoke tests
```
