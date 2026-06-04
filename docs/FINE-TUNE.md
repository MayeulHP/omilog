# Fine-tuning omilog for your data

Once you have a week or two of real captures, you'll notice the defaults
are calibrated for "general conversational French/English from a chest-mic
under typical home noise." Your situation is yours; this guide is the
recipe for tuning each stage against a recording you can listen to and
ground-truth in your head.

The two complaints I expect first:

1. **Whisper produces hallucinated, garbled, or just generally weak
   transcriptions.** Usually fixable; the model is fine, the inputs and
   decoding flags need adjustment.
2. **Diarization labels three people as five.** Very common with sherpa-onnx
   defaults. The 0.3s minimum speech threshold is too eager.

This doc walks through both, plus the workflow for iterating without
re-recording everything.

---

## The iteration workflow

You need a fixed recording you know intimately. Pick one conversation
where you remember roughly what was said and who was there. The audio file
lives at `storage/<audio_session_id>.opus`.

Re-process it with different settings using `scripts/replay_session.py`:

```bash
# rerun the full STT → diarize → LLM pipeline on one session
.venv/bin/python scripts/replay_session.py <audio_session_id>

# only re-do diarization (cheap, keeps the existing transcript)
.venv/bin/python scripts/replay_session.py <audio_session_id> --stage diarize

# only re-do LLM extraction (free, runs in a second)
.venv/bin/python scripts/replay_session.py <audio_session_id> --stage llm
```

Open the conversation in `/conversations/<id>` between runs, compare the
transcript, speaker labels, and extracted items. Don't try to tune more
than one knob at a time or you won't know which one moved the needle.

The `/tune` page does the same iterative loop for **VAD parameters**
specifically; it's the same idea, just with a UI.

---

## Whisper / STT quality

The single biggest win: **disable previous-text conditioning** on the
whisper-server side. See `docs/whisper-server.md` for the `--no-context`
flag. That alone removes 90% of "stuck-in-a-loop" hallucinations on quiet
audio.

Beyond that, the knobs we expose:

### `OMILOG_STT_INITIAL_PROMPT`

The single most impactful tuning knob. Whisper biases its vocabulary
toward whatever's in this prompt. Use it for:

- **Proper nouns**: names of frequent contacts, places, technical terms
  ("Marie, Paul, Bastille, Caddy, Tailscale, omilog").
- **Domain words**: if you're in tech, throw in the relevant jargon.
- **The dominant language**: explicitly mention "Cette conversation est en
  français" if Whisper keeps falling into English on ambiguous segments.

Keep it concrete, comma-separated, and under 200 characters. A long
abstract prompt is worse than a short one full of actual names.

Example for a French-speaking developer:

```env
OMILOG_STT_INITIAL_PROMPT="Conversation en français. Marie, Paul, Mayeul, Bastille, Tailscale, FastAPI, llama.cpp, Whisper, omilog."
```

### `OMILOG_STT_TEMPERATURE`

- `0.0` (default): deterministic, sticks to its best guess. Good when
  audio quality is consistently good.
- `0.2`: gives Whisper room to back out of a bad guess on noisy segments.
  Try this if you see segments that are clearly wrong but plausible-looking
  (Whisper is confidently making things up).
- Higher than `0.4`: not recommended. The output starts varying between
  runs in unhelpful ways.

### `OMILOG_STT_LANGUAGE`

- `auto` (default): Whisper detects language per segment. Good for
  multilingual speakers but can flip mid-conversation.
- `fr` / `en` / explicit ISO code: forces the language. Use this if your
  conversations are essentially one language and `auto` is making
  language-detection mistakes (Spanish becoming Italian, etc.).

### `OMILOG_STT_MODEL_NAME`

This is just the label stamped on transcripts; the actual model is whatever
your whisper-server has loaded. If quality is still poor after the above
adjustments, the model itself is the limit. Options, smallest to largest:

| Model | Size | Speed (Pi-CPU/GPU) | Notes |
|---|---|---|---|
| `large-v3-turbo-q5_0` | ~870 MB | fast | omilog default; quantized; quality drop vs full is small |
| `large-v3-turbo` | ~1.5 GB | medium | unquantized turbo; better for accents, code-switching |
| `large-v3` | ~3 GB | slow | best general-purpose; meaningful quality bump |
| `distil-large-v3` | ~750 MB | very fast | English-only; pure-English captures only |

Swap on the GPU box, restart whisper-server, no omilog change needed.

### Hardware-level

If Whisper is still struggling after these knobs, the audio itself may be
the bottleneck. Things to check:

- **Mic placement on the Omi**: a necklace mic 30cm from your mouth picks
  up a lot of ambient noise. Test the same room talking with the necklace
  closer (chest level, not stomach).
- **Audio levels**: if captures sound quiet in `<audio>` playback, the BLE
  encoder might be clipping or under-driving. Look at peak levels with
  `ffmpeg -i <file>.opus -filter:a volumedetect -f null /dev/null`. Peaks
  under -25 dB mean you're losing dynamic range.
- **Background noise**: kitchen, traffic, music. Whisper handles ~10 dB
  SNR OK and degrades fast below that. Not much we can do at the software
  layer without pre-processing (denoise → speech-isolated audio).

---

## Diarization quality

Default sherpa-onnx parameters bias toward finding **more** speakers than
**fewer**. This is the right trade-off for security-style use cases (don't
miss a speaker) but wrong for conversational omilog usage where false
splits make the UI noisy.

Knobs:

### `OMILOG_DIARIZATION_MIN_SPEECH_SECONDS`

The minimum length of a continuous speech segment for diarization to even
consider it. Default `0.3`s catches short utterances ("yeah", "okay") but
also catches ambient sound, kid-noises in the background, your own breath.
Each of those gets a fresh speaker cluster, so a 10-minute family meal
turns into S1 through S8.

Try bumping to `0.7` or `1.0` first. You'll lose the ability to label
single-word interjections, but the cluster count drops dramatically.

```env
OMILOG_DIARIZATION_MIN_SPEECH_SECONDS=0.8
```

### `OMILOG_DIARIZATION_MIN_SILENCE_SECONDS`

Minimum gap between two speech segments before they count as different
turns. Default `0.5`s. Lowering this merges adjacent speech into one turn;
raising it splits aggressively. If you're seeing the same speaker labelled
two different ways for back-to-back sentences, raise this. If two distinct
speakers are getting merged when they take quick turns, lower it.

### `OMILOG_SPEAKER_MATCH_THRESHOLD` (cross-conversation linking)

Once diarization assigns within-conversation labels (USER, S1, S2…),
omilog's linking step matches them across conversations via cosine
similarity against stored Speaker embeddings. Default `0.6`.

- **Too many duplicate Speakers** ("Marie" exists as three rows): drop
  to `0.5` or `0.45`.
- **Different people getting merged** ("Marie" and "Sophie" share a row):
  raise to `0.7` or `0.75`.

Tune via `/speakers` page: delete obvious duplicates, look at remaining
ones, adjust threshold, re-process a conversation and see whether the
right people get matched.

### When the model itself is the limit

NeMo TitaNet (the default embedding model in sherpa-onnx) was trained
primarily on English speakers. French same-gender same-age speakers tend
to embed closer together than English ones do, so it merges them more
often. Two options:

1. **Replace with a different embedding model** from sherpa-onnx's
   catalogue (some are multilingual). Drop the new `.onnx` into
   `models/`, update `OMILOG_DIARIZATION_EMBEDDING_MODEL`.
2. **Live with it and rely on manual rename** via `/speakers`. If you've
   correctly named Marie once, the model's mistake doesn't propagate; she
   stays Marie wherever her cluster appears.

For most users, option 2 is fine.

---

## LLM extraction quality

Quality (titles, summaries, extracted events/actions) is mostly a function
of:

- The model (qwen3-30b-a3b is a reasonable default; downgrade to 14b or 7b
  if your llama box is slow, upgrade to qwen3-72b for marginal gains).
- The prompt. Editable at `/config/prompt`.
- The transcript itself. Garbage in, garbage out — fix STT first.

Common things to tweak in the system prompt:

- Add concrete examples of "what counts as a calendar event" and "what
  doesn't" if you're seeing too many false-positive events.
- Tighten language about the wearer ("USER") if extractions keep
  attributing the wearer's commitments to other speakers.
- Add domain context ("user is a developer in Paris") if extractions need
  more cultural grounding.

Re-run on real data via `replay_session.py --stage llm` after each prompt
edit; the LLM step is the cheapest one to iterate on.

---

## A reasonable starting recipe

If you're not sure where to start, here's a config that's worked well for
French conversational use on a Pi + GPU box:

```env
# Whisper
OMILOG_STT_TEMPERATURE=0.0
OMILOG_STT_LANGUAGE=fr
OMILOG_STT_INITIAL_PROMPT="Conversation en français. [comma-separated proper nouns you actually use]"
# (plus --no-context on the whisper-server side, see docs/whisper-server.md)

# Diarization
OMILOG_DIARIZATION_MIN_SPEECH_SECONDS=0.8
OMILOG_DIARIZATION_MIN_SILENCE_SECONDS=0.5
OMILOG_SPEAKER_MATCH_THRESHOLD=0.55

# Quality scoring
OMILOG_DAILY_SUMMARY_THRESHOLD=0.4
```

Tune from there based on what bothers you most. Don't change everything
at once.
