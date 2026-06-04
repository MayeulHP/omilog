# Speaker diarization setup (local-only)

Phase 4 of omilog: annotates each transcript segment with a speaker label —
`USER` (you, the necklace wearer) and `S1`, `S2`, … for other people in the
conversation. Heuristic: the speaker with the most cumulative talk time per
conversation is the user (a wearable mic puts your voice consistently loudest).

This is **opt-in** and **fully local**. Two ONNX models (~50 MB total) get
downloaded once from sherpa-onnx GitHub releases; thereafter the diarizer
runs entirely on the Pi (or wherever omilog is hosted). **Your audio never
leaves the tailnet.**

## One-time setup (≈ 2 min)

1. **Install the system dependency for soundfile** (Debian / Ubuntu / Raspberry Pi only)
   ```bash
   sudo apt install libsndfile1
   ```
   `soundfile` is a Python wrapper around `libsndfile`. The wheel itself
   installs fine, but at import time it loads the system shared library and
   raises `OSError: cannot load library 'libsndfile.so'` if it isn't there.
   This is the single most common cause of "diarization deps failed to
   import" messages in the omilog startup log. macOS users get this for free
   via Homebrew's `pythonX` dependency chain; Linux users need the explicit
   install.

2. **Install the Python extra**
   ```bash
   uv sync --extra diarization
   # or:  pip install -e ".[diarization]"
   ```
   Pulls sherpa-onnx + soundfile + numpy + onnxruntime. ~80 MB total. No torch,
   no HuggingFace client, no PyTorch. We install `onnxruntime` explicitly
   because sherpa-onnx's Linux aarch64 wheel doesn't always bundle
   `libonnxruntime.so` cleanly; on x86_64 / macOS the duplicated install
   is harmless.

3. **Download the models**
   ```bash
   .venv/bin/python scripts/download_diarization_models.py
   ```
   Fetches:
   - `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` (~12 MB) —
     speech/silence + speaker-boundary detection (the same pyannote model
     pyannote-audio uses, just ONNX-converted)
   - `models/nemo_en_titanet_small.onnx` (~28 MB) — speaker embedding
     extractor for clustering similar voices
   
   Idempotent: re-runs skip already-downloaded files. Models come from
   `github.com/k2-fsa/sherpa-onnx/releases` (Apache 2 license, no account).

4. **Wire it into `.env`**
   ```env
   OMILOG_DIARIZATION_ENABLED=true
   # The download script prints the exact paths to copy here; the defaults
   # match where the script places the files.
   ```

5. **Restart**
   ```bash
   ./scripts/start.sh
   ```
   Startup log should now show:
   ```
   pipeline: diarization enabled (sherpa-onnx, models=model.onnx, nemo_en_titanet_small.onnx)
   ```

## What you'll see

Conversation detail pages get speakers added at the top:

> Speakers: **USER**, **S1**, **S2**

Each transcript line is color-coded per speaker. The LLM prompt also sees
the labels (`[USER] Salut Marie.` / `[S1] On se voit demain?`), so action
items get attributed to the right person.

## Performance

On a Pi 5 (CPU only):
- First inference per process: ~5 s model load + processing
- Per conversation (warm): ~3-5× real-time (a 5-min conversation = ~1-2 min
  of diarization)
- RAM use: ~300 MB during inference

That's faster and lighter than pyannote-audio + torch (which uses ~2 GB),
because sherpa-onnx's ONNX runtime skips most of PyTorch's overhead.

If you want it faster, the next step is running sherpa-onnx on your
existing GPU box as a small HTTP service (similar to whisper-server).
Not implemented yet — see `docs/TODO.md`.

## When diarization fails

Intentionally non-blocking. If models are missing, ONNX rejects the input,
or sherpa-onnx crashes on a specific file, the pipeline logs a warning and
continues to LLM extraction **without speaker labels**:

```
pipeline: diarize <session-id> failed (DiarizationError: …) — continuing
          without speaker labels
```

The transcript still saves, the LLM still extracts events/actions, the UI
just won't color-code that conversation.

## Known limitations

- **Two speakers with very similar voices**: sherpa-onnx (like pyannote)
  may merge them into one. Common with two men or two women in close pitch.
- **Distant speakers in noisy rooms**: harder to distinguish from each
  other. The USER label is usually right (your voice arrives loudest), but
  S1/S2 may swap or merge.
- **Cross-conversation linking**: speaker labels are **per-conversation
  only**. `USER` is always you across conversations, but `S1` in two
  different conversations is *not* automatically the same person. Tracking
  recurring voices is a Phase 5 idea — see `docs/TODO.md`.
- **English-trained embedding model**: NeMo TitaNet was trained on
  VoxCeleb (mostly English). Speaker embeddings are largely
  language-agnostic in practice, but if you find it under-distinguishing
  French speakers, swap to a French-trained embedding by overriding
  `OMILOG_DIARIZATION_EMBEDDING_MODEL`.
