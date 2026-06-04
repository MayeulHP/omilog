# Contributing to omilog

Thanks for the interest. omilog is primarily a personal tool that I open-sourced
because the architecture might be useful to others. That shapes the bar for PRs:

- Bug fixes and clear quality-of-life improvements: always welcome.
- New features: propose in an issue first so we can sanity-check scope before
  you spend an evening on it.
- Tests for non-trivial changes. The suite is fast and lives in `tests/`.

If you fork to build something materially different (e.g. you want a different
audio source, or you're stripping omilog down for a different platform), no
need to keep parity — file an issue with a link so other people running similar
forks can find each other.

## Dev setup

Same as the user-facing setup, no special path for contributors:

```bash
git clone https://github.com/MayeulHP/omilog
cd omilog
./scripts/start.sh
```

The first run creates a venv, installs deps via `uv` (falls back to `pip`),
prompts for credentials, writes `.env`, then starts uvicorn on `127.0.0.1:8000`.

If you only want to run the tests without booting anything:

```bash
uv sync --extra dev
uv run pytest
```

## Running tests and lint

```bash
uv run pytest                  # ~15 s, ~190 tests
uv run ruff check src tests    # lint
```

Tests mock ffmpeg, whisper.cpp, llama-server, and sherpa-onnx — no GPU, no
backend services, no system dependencies needed. They run on any machine with
Python 3.11+.

A handful of integration tests shell out to `ffmpeg` or `ffprobe` if installed,
and skip otherwise. Install ffmpeg locally if you want to exercise those.

## Project structure

See the **Project layout** section in [README.md](README.md). The short version:

- `src/omilog/api/` — JSON API
- `src/omilog/web/` — server-rendered HTML UI (Jinja + HTMX)
- `src/omilog/pipeline/` — VAD → STT → diarize → LLM stages, plus the runner
- `src/omilog/audio/`, `src/omilog/ics.py` — small focused modules
- `tests/` — one `test_*.py` per surface area
- `docs/` — setup guides for opt-in features + deferred-work TODO
- `scripts/` — bootstrap, deploy helpers, model downloads

## Schema changes

omilog doesn't ship a migration framework. Two acceptable patterns when you
need to evolve the schema:

1. **Adding a new table**: `SQLModel.metadata.create_all()` handles it on the
   next boot. No migration needed.
2. **Changing an existing table**: ship a one-shot
   `scripts/migrate_<short_name>.py` using raw `sqlite3.execute("ALTER TABLE …")`,
   idempotent (no-op if the column already exists). Reference it in your PR
   description so it's discoverable.

If your change is more invasive than `ALTER TABLE ADD COLUMN`, propose in an
issue first — we may want to introduce alembic at that point.

## Code style

- `ruff` handles lint and formatting (config in `pyproject.toml`).
- Type hints encouraged but not enforced.
- Comments should explain *why*, not *what*. The reader can see *what*.
- Tests named for the behaviour they exercise, not the function they call:
  `test_runner_continues_when_diarization_fails` not `test_process_stt_5`.

## Pull requests

- Keep them focused. One conceptual change per PR.
- Run `pytest` + `ruff check` locally before pushing. CI runs the same.
- Include a one-paragraph "why" in the PR description — this is much more
  useful than a description of *what* the diff does (the diff already shows
  that).

## Where to ask

- Bugs or feature ideas: GitHub Issues on this repo.
- Anything else: open an issue and mark it `discussion` — there's no Discord
  or Slack for the project (yet).

Thanks again.
