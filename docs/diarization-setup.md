# Speaker diarization setup

Phase 4 of omilog: annotates each transcript segment with a speaker label —
`USER` (you, the necklace wearer) and `S1`, `S2`, … for other people in the
conversation. Heuristic: the speaker with the most cumulative talk time per
conversation is the user (a wearable mic puts your voice consistently loudest).

This is **opt-in**, because it pulls ~2 GB of PyTorch wheels and needs a
HuggingFace token. If you don't enable it, the rest of the pipeline runs
exactly as before.

## One-time setup (≈ 5 min)

1. **Install the extra**
   ```bash
   uv sync --extra diarization
   # or:  pip install -e ".[diarization]"
   ```
   First install downloads torch + pyannote-audio, ~2 GB.

2. **HuggingFace account + token**
   - Make an account at https://huggingface.co/join (free).
   - Generate a read-only token: https://huggingface.co/settings/tokens → New
     token → name it `omilog` → role `read` → copy the `hf_…` string.

3. **Accept the model licenses** (still free, but pyannote gates downloads)
   - https://huggingface.co/pyannote/segmentation-3.0 → click "Agree".
   - https://huggingface.co/pyannote/speaker-diarization-3.1 → click "Agree".

4. **Wire it into `.env`** on the Pi
   ```bash
   OMILOG_DIARIZATION_ENABLED=true
   OMILOG_HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ```

5. **Restart**
   ```bash
   ./scripts/start.sh
   ```
   The startup log should now show:
   ```
   pipeline: diarization enabled (model=pyannote/speaker-diarization-3.1)
   ```

   First captured conversation after restart will trigger the model download
   (~300 MB to `~/.cache/huggingface/`). Subsequent ones load from cache.

## What you'll see

Conversation detail pages get speakers added at the top:

> Speakers: **USER**, **S1**, **S2**

And each transcript line is color-coded per speaker. The LLM prompt also
sees the labels, so action items get attributed to the right person:
"Marie said she'd send the doc" instead of "someone will send the doc."

## Performance

On a Pi 5 (CPU only):
- First load: ~30 s (model into RAM)
- Per conversation: ~5-8× real-time (a 5-min conversation → 30-60 s of
  diarization on top of STT)
- RAM use: ~2 GB peak during inference

If this is too slow on your hardware, options are:
- Disable diarization (`OMILOG_DIARIZATION_ENABLED=false`) and run without
  labels — STT/LLM still work fine
- Run a pyannote service on your GPU box (not yet implemented in omilog —
  let me know if you want this)

## When diarization fails

It's intentionally non-blocking. If model loading fails, the model
disagrees, or pyannote crashes on a specific file, the pipeline logs a
warning and continues to LLM extraction **without speaker labels**:

```
pipeline: diarize <session-id> failed (DiarizationError: …) — continuing
          without speaker labels
```

The transcript still gets saved, the LLM still extracts events/actions,
the UI just won't color-code that conversation.

## Known limitations

- **2 speakers, close together in voice register**: pyannote may merge them
  into one. Common with two men or two women in similar pitch range.
- **Distant speakers in noisy rooms**: harder to distinguish from each
  other (your voice arrives loudest, so the USER label is usually right,
  but S1/S2 may swap or merge).
- **Cross-conversation linking**: speaker labels are **per-conversation
  only**. The "USER" in conversation A and "USER" in conversation B
  are the same person (it's you), but `S1` in two different conversations
  is *not* automatically the same person. Tracking recurring voices across
  conversations is a Phase 5 idea — see `docs/TODO.md`.
