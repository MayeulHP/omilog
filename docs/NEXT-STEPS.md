# Next steps — STT & diarization quality push

Working notes from the 2026-06-11 engineering/product review. The full
reasoning lives in that conversation; this file is the actionable trail.
Constraint reminder: ~40 GB VRAM across 4 GPUs, currently ~18 GB used by
the LLM (qwen-3.6-27B dense, reasoning on, shared with hermes-agent) +
~0.5 GB whisper — so roughly 20 GB of headroom for everything below.

## Status

| Step | What | State |
|---|---|---|
| S0 | Eval harness (WER / DER / USER-attribution) + `/eval` labeling UI | ✅ shipped — needs **your labels** |
| S1 | Silero VAD backend (noise-robust segmentation) | ✅ shipped — opt-in, needs enabling |
| S2 | faster-whisper server + word-level timestamps | ⬜ next code step |
| D1 | Diarization moved to GPU (pyannote service) | ⬜ after S2 |
| D2 | Identification-first diarization vs known speakers | ⬜ after D1 |
| D3 | Voice enrollment page (enroll yourself = robust USER) | ⬜ after D1 |
| S4/D5 | LLM transcript repair + speaker smoothing stage | ⬜ after numbers exist |

## 1. Label the eval set ← everything is blocked on this

Pick 5–10 archived (📌) conversations covering real conditions — quiet
room, street, restaurant, one-on-one, group meal.

**Easy path — the `/eval` web UI:** on each conversation's page click
**📋 → eval case** (tick **HQ draft** for a better starting transcript),
then on the case page correct words + speakers in the row editor while the
audio plays (▶ on a row seeks to it), tick **Verified**, save. The
per-case **Run eval** button scores it against the live config.

**CLI equivalent** (headless / scriptable):

```bash
.venv/bin/python scripts/eval_bootstrap.py <session-uuid> --name dinner-noisy --hq
# correct the row-based reference_turns.json + reference.txt, set verified
```

Full labeling guidance: [eval/README.md](../eval/README.md). Then capture
the baseline **before changing any pipeline config**:

```bash
.venv/bin/python scripts/eval_run.py --note "baseline turbo-q5"
```

## 2. Free config experiments (no code, one .env line each)

Run `eval_run.py --note "<what changed>"` after each; compare via
`eval/results/history.jsonl`:

- `OMILOG_STT_LANGUAGE=fr` (vs `auto` — auto flips language mid-conversation
  on noisy audio; expect a visible WER win)
- `OMILOG_STT_INITIAL_PROMPT="Conversation en français. <recurring names,
  domain terms>"` vs empty
- `OMILOG_STT_TEMPERATURE=0.2` vs `0.0` on the noisy cases
- Confirm whisper-server runs with `--max-context 0` (see
  [whisper-server.md](whisper-server.md) — kills most hallucination loops)
- Use `--reuse-stt` when iterating on diarization knobs only

## 3. Enable Silero VAD (S1, shipped 2026-06-11)

On the Pi:

```bash
uv sync --extra silero                                # onnxruntime + numpy, no torch
.venv/bin/python scripts/download_silero_vad.py       # ~2 MB ONNX, MIT
# then OMILOG_VAD_BACKEND=silero in .env or /config, restart
```

Try it on a real noisy session first via `/tune` (backend dropdown) before
flipping the default. Validated on synthetic audio: with continuous
background noise, silencedetect reports **zero** silences (capture never
splits, noise reaches Whisper) while silero finds the same boundaries it
finds on clean audio. Knobs: `OMILOG_VAD_SILERO_THRESHOLD` (0.5; up to 0.7
if background TV counts as speech, down to 0.35 for whispers),
`OMILOG_VAD_SILERO_MIN_SPEECH_SECONDS` (0.3; blips shorter than this can't
break a long silence). Falls back to silencedetect with a logged warning
if deps/model are missing.

Note: S1 improves *future captures'* segmentation (fewer noise-fed
hallucinations, real conversation splits). It won't move WER on already-
segmented eval cases — its in-segment counterpart (speech-gating before
whisper) arrives free with S2's built-in VAD filter.

## 4. LLM token budget (reasoning-mode qwen-3.6-27B)

Reasoning-on means `<think>` tokens spend the output budget before the
JSON appears → truncations → `was_repaired` penalties. Today, no code:
`OMILOG_LLM_MAX_TOKENS=16384` in `/config` (generation budget, not
context — don't set it to 200k or a runaway reasoning loop spins for
minutes). The code side (shipped default, stale `llm_model` name,
unclosed-`<think>` parser hardening) is queued as a task chip.

## 5. Then, in order (ask Claude to build)

1. **S2 — faster-whisper/speaches server**: f16 large-v3 or turbo
   (~4–5 GB VRAM), built-in Silero speech-gating per request, batching,
   and **word-level timestamps** — the prerequisite for fixing
   speaker-attribution at turn boundaries. Client change is small
   (same OpenAI-ish HTTP shape as whisper.cpp).
2. **D1 — GPU diarization service**: ~150-line FastAPI wrapper around
   pyannote 3.x on a GPU box (<2 GB VRAM), same pattern as whisper-server;
   sherpa-onnx stays as the no-GPU fallback. Replaces the weakest model in
   the weakest place (TitaNet-small on Pi CPU at 3–5× real-time).
3. **D4 — word-level speaker fusion**: assign speakers per word, re-chunk
   transcript into turns. Fixes the "even a perfect diarizer attributes
   wrongly at boundaries" ceiling caused by 5–15 s whisper segments.
4. **D3 — voice enrollment** (`/speakers/enroll`, ~2 h per TODO.md) and
   **D2 — identification-first**: match turns against known Speaker
   embeddings first, cluster only the unknown remainder; enroll yourself
   so USER = "nearest to my voice" instead of the talk-time heuristic.
5. **S4/D5 — LLM repair stage**: one extra qwen call per conversation to
   fix French homophone/ASR errors and semantically-impossible speaker
   flips (DiarizationLM-style); cleaned view by default, raw behind a
   toggle.

Product backlog (independent, grab anytime): FTS5 search → "ask my day"
RAG, ntfy notifications for events/actions, per-segment confidence
rendering from whisper logprobs.

## Decision rule

Every change above gets judged by `eval_run.py` numbers against the
baseline — a knob that doesn't move WER/DER on real captures is noise,
revert it. When labeling more cases later, prefer the conditions where
the current numbers are worst.
