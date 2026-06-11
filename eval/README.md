# Eval set — ground truth for STT & diarization tuning

Everything under `eval/cases/` and `eval/results/` is **gitignored**: it
contains your personal audio and transcripts and must never reach the
public repo. Only this README is committed.

## Why

Every STT/diarization knob (`/tune`, `.env`, model swaps) is currently
tuned by vibes. This harness gives you numbers instead: change one thing,
run the suite, compare against the history. A config change that doesn't
move WER/DER on real captures isn't an improvement.

## Workflow

1. **Pick 5–10 conversations** covering your real conditions: quiet room,
   street, restaurant, one-on-one, multi-speaker. 3–8 minutes each is the
   sweet spot (longer cases slow WER alignment down). Archive them (📌)
   so audio rotation never deletes the source.

2. **Bootstrap a case** from each:

   ```bash
   .venv/bin/python scripts/eval_bootstrap.py <session-uuid> --name dinner-noisy
   ```

   This copies the audio and exports the *machine* transcript + speaker
   turns as a starting point.

3. **Hand-correct, with the audio playing** (the conversation page's
   player, or the copied `audio.opus` in any player):

   * `reference.txt` — fix the words. Guidance:
     - Scoring ignores case, punctuation and line breaks, and splits on
       apostrophes/hyphens — don't polish those, fix *words*.
     - Keep accents correct ("été" ≠ "ete").
     - Write numbers the way Whisper does (digits: "14h30", not
       "quatorze heures trente") so formatting differences don't count
       as errors.
     - Delete hallucinated text entirely; add missed speech.
   * `reference_turns.json` — fix `start`/`end`/`speaker`. `USER` is the
     necklace wearer; other speakers can be any consistent label (`S1`,
     `marie`, …) — DER matches labels by optimal mapping, only `USER`
     is compared literally (for the wearer-attribution metric). Boundary
     precision of ±0.2 s is fine; scoring uses a 0.25 s collar.
   * `case.json` — set `"verified": true`. Unverified cases still run
     but are flagged ⚠: scoring machine output against itself reads as
     a fake 0% error.

4. **Run the suite**:

   ```bash
   .venv/bin/python scripts/eval_run.py --note "baseline large-v3-turbo-q5"
   .venv/bin/python scripts/eval_run.py --reuse-stt   # diarization-only iteration
   ```

   Results print as a table and append to `eval/results/history.jsonl`
   together with the exact STT/diarization config that produced them.

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
