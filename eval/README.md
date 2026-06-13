# Eval set — ground truth for STT & diarization tuning

Everything under `eval/cases/` and `eval/results/` is **gitignored**: it
contains your personal audio and transcripts and must never reach the
public repo. Only this README is committed.

A case directory holds `audio.*` (any ffmpeg-readable copy of the source
audio), `reference_turns.json` (rows of `start`/`end`/`speaker`/`text` —
the labeling UI edits these; `speaker` is optional per row), `reference.txt`
(the words, derived from the rows; feeds WER) and `case.json` (metadata +
the `verified` flag). At scoring time, same-speaker rows ≤1 s apart are
bridged into turns on **both** the reference and hypothesis side, so the
segment-level quantization is symmetric.

## Why

Every STT/diarization knob (`/tune`, `.env`, model swaps) is currently
tuned by vibes. This harness gives you numbers instead: change one thing,
run the suite, compare against the history. A config change that doesn't
move WER/DER on real captures isn't an improvement.

## Workflow (web UI — the easy path)

1. **Pick 5–10 conversations** covering your real conditions: quiet room,
   street, restaurant, one-on-one, multi-speaker. 3–8 minutes each is the
   sweet spot (longer cases slow WER alignment down). Archive them (📌)
   so audio rotation never deletes the source.

2. On each conversation's page, click **📋 → eval case**. The optional
   **HQ draft** checkbox re-transcribes on the spot with quality-leaning
   settings (pinned language + a vocabulary prompt built from your known
   speaker/people names) so the draft needs fewer fixes — caveat: it's
   still the same model family you're evaluating, so anything you don't
   actually check against the audio biases WER optimistically.

3. On the case page (`/eval/<name>`): correct words and speakers in the
   row editor while the audio plays — ▶ on a row seeks to it. Guidance:

   * **Words** — scoring ignores case, punctuation and line breaks, and
     splits on apostrophes/hyphens, so fix *words*, not polish. Keep
     accents ("été" ≠ "ete"). Write numbers the way Whisper does
     (digits: "14h30"). Delete hallucinated rows entirely; add rows for
     missed speech.
   * **Speakers** — `USER` is the necklace wearer; other speakers can be
     any consistent label (`S1`, `marie`, …) — DER matches labels by
     optimal mapping, only `USER` is compared literally (for the
     wearer-attribution metric). A row can have no speaker (text-only
     cases skip DER). Boundary precision of ±0.2 s is fine; scoring
     uses a 0.25 s collar.

   Tick **Verified** and save. Unverified cases still run but are
   flagged ⚠: scoring machine output against itself reads as a fake 0%
   error.

4. **Score**: the **Run eval** button on a case page scores that case
   against the live config; for the whole suite use the CLI:

   ```bash
   .venv/bin/python scripts/eval_run.py --note "baseline large-v3-turbo-q5"
   .venv/bin/python scripts/eval_run.py --reuse-stt   # diarization-only iteration
   ```

   Both append to `eval/results/history.jsonl` together with the exact
   STT/diarization config that produced them.

Everything also works headless: `scripts/eval_bootstrap.py <session-uuid>
[--hq]` creates a case, and the files below are hand-editable — the web
UI is just a front-end over them.

## Metrics

* **WER** — word error rate on normalized text (substitutions +
  insertions + deletions over reference words). Computed on the
  repeat-collapsed segments, i.e. the transcript the UI shows.
* **DER** — diarization error rate (missed speech + false alarm +
  speaker confusion over reference speech time), 0.25 s collar,
  label-agnostic optimal speaker mapping. Computed on the *relabeled
  transcript segments*, so it scores the full attribution chain the
  product shows (diarizer + segment assignment + USER heuristic), not
  the raw diarizer in isolation.
* **USER%** — fraction of the wearer's reference speech time the system
  labeled `USER`. Directly measures the talk-time wearer heuristic.

Not covered: cross-conversation speaker linking / `is_user` promotion
(needs DB state, would mutate it). DER here is the per-conversation
pipeline only.

## Comparing runs

Each `eval_run.py` invocation appends one JSON line to
`eval/results/history.jsonl` with timestamp, `--note`, config snapshots,
per-case metrics and aggregates. Quick look at the trend:

```bash
jq -r '[.ts, .note, (.aggregate.wer // "-"), (.aggregate.der // "-")] | @tsv' \
    eval/results/history.jsonl
```
