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
whisper-server side via `--max-context 0` (note: the CLI binary uses
`--no-context` but whisper-server uses `--max-context`; see
`docs/whisper-server.md` for the full command line). That alone removes
90% of "stuck-in-a-loop" hallucinations on quiet audio.

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

There are TWO ways diarization can go wrong, and they need different
knobs to fix:

1. **Too many clusters within ONE conversation** (the "S1 said two
   sentences, S2 said one, S3 said three sentences, but actually it was
   all the same person" problem). This is the within-conversation
   clusterer over-segmenting. Fixed by ``num_clusters`` (force) or
   ``cluster_threshold`` (loosen merging).
2. **Same person appearing as multiple Speaker rows ACROSS
   conversations**. This is the cross-conversation linker treating
   today's "Marie" as a different voice from yesterday's. Fixed by
   ``OMILOG_SPEAKER_MATCH_THRESHOLD`` further down.

### `OMILOG_DIARIZATION_NUM_CLUSTERS` (force speaker count)

The cleanest fix for over-segmentation when you know the typical
conversation has 2-4 people. ``-1`` (default) lets sherpa-onnx
auto-detect; setting it to a positive integer pins the clusterer to
exactly that many speakers per conversation. The model still picks
which utterances go to which cluster — you're just capping the cluster
count.

```env
OMILOG_DIARIZATION_NUM_CLUSTERS=3
```

Trade-off: if a conversation legitimately has 5 people and you've
pinned it to 3, two of them will merge. For mixed use cases, ``-1`` +
a tuned ``cluster_threshold`` is more flexible.

### `OMILOG_DIARIZATION_CLUSTER_THRESHOLD` (limited effect in practice)

sherpa-onnx's internal cluster threshold (only active when
``num_clusters=-1``). The docs sell it as a cosine cutoff but
empirically — confirmed on real French conversation audio — pushing
it as low as 0.1 doesn't meaningfully change the cluster count. The
underlying clustering algorithm seems to lean on its auto-K
detection more than on this threshold. We expose the knob but **don't
rely on it** for over-segmentation fixes.

Use `post_merge_threshold` below instead.

```env
OMILOG_DIARIZATION_CLUSTER_THRESHOLD=0.5     # default; touching it rarely helps
```

### `OMILOG_DIARIZATION_POST_MERGE_THRESHOLD` ★ the reliable knob

After sherpa-onnx finishes, omilog runs an in-Python second pass:
compute one embedding per cluster (via the same NeMo TitaNet model),
all-pairs cosine compare, fold any pair whose similarity is ≥ this
threshold into one cluster. Union-find so transitive merges (A~B and
B~C ⇒ A=B=C) Just Work. The survivor of a merge is the lexicographically
smaller label (SPEAKER_00 wins over SPEAKER_05) so labels stay
predictable across re-runs.

Range 0.5-1.0, default 1.0 (disabled). When enabled:

- **0.9-0.95**: only fold near-identical clusters. Safe — never merges
  distinct people. Most useful when you trust sherpa-onnx's clustering
  mostly but want to catch the occasional false split.
- **0.8 (sweet spot)**: the recommended starting point for the
  "diarization shows 9 speakers but it was really 2 people" symptom.
  Will fold most over-splits without merging genuinely distinct voices.
- **0.7**: aggressive. Good for very over-split conversations on a small
  number of similar voices. Watch out for legitimately-distinct people
  with similar voices getting folded (same-gender, same-age, same-accent).
- **<0.7**: discouraged — high risk of merging real speakers.

```env
OMILOG_DIARIZATION_POST_MERGE_THRESHOLD=0.8
```

Unlike forcing `num_clusters`, this scales naturally: a real 5-person
meeting still produces 5 distinct clusters (their embeddings are
genuinely far apart); a real 2-person chat that got over-split into
9 clusters by sherpa-onnx folds back down to 2 because the over-split
embeddings cluster tightly. Best of both worlds.

Iterate on /tune/<session-id> to find the right value for your audio —
the page lets you preview the result of any threshold without restart.

### `OMILOG_DIARIZATION_MIN_SPEECH_SECONDS`

The minimum length of a continuous speech segment for diarization to even
consider it. Default `0.3`s catches short utterances ("yeah", "okay") but
also catches ambient sound, kid-noises in the background, your own breath.
Each of those gets a fresh speaker cluster.

Try bumping to `0.7` or `1.0` if you have lots of background noise. Less
effective than ``cluster_threshold`` for the "long-conversation-shows-
many-speakers" case, which is usually a clustering problem rather than a
noise problem.

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

NeMo TitaNet (the original default embedding model) was trained primarily
on English telephony audio. Same-gender French speakers tend to embed
closer together than English ones do, so it merges them more often. Two
options:

1. **Swap the embedding model.** `scripts/download_diarization_models.py`
   fetches a second candidate alongside TitaNet:
   `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` (~10 MB,
   VoxCeleb-trained, multi-scale ERes2Net architecture). It's a drop-in
   ONNX replacement — point the env var at the new file and restart:

   ```env
   OMILOG_DIARIZATION_EMBEDDING_MODEL=models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
   ```

   VoxCeleb's speaker pool is more diverse than TitaNet's English
   telephony bias and the multi-scale fusion produces more stable
   embeddings on short utterances — both of which matter for typical
   omilog conversations. This is the recommended first attempt if
   diarization quality is biting you.

2. **Live with the merging and rely on manual rename / merge** via
   `/speakers`. If you've correctly named Marie once, future occurrences
   of her voice match against the stored embedding by cosine similarity,
   and the labels propagate. The `/speakers` page also has a real merge
   button for the cases where the model created two separate rows for
   the same person.

For most users, option 1 is worth a try (5-minute test on a known
conversation via `/tune/<id>`); fall back to option 2 if the swap doesn't
improve things on your data.

---

## LLM extraction quality

Quality (titles, summaries, extracted events/actions) is mostly a function
of:

- The model (qwen-3.6-27b is the current default; smaller Qwen variants work
  if your llama box is slow). With reasoning-enabled models, keep
  `OMILOG_LLM_MAX_TOKENS` generous — thinking tokens spend the same budget
  as the answer — or leave `OMILOG_LLM_DISABLE_THINKING=true` to skip the
  think block per-request.
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
# (plus --max-context 0 on the whisper-server side, see docs/whisper-server.md)

# Diarization
OMILOG_DIARIZATION_MIN_SPEECH_SECONDS=0.8
OMILOG_DIARIZATION_MIN_SILENCE_SECONDS=0.5
OMILOG_DIARIZATION_POST_MERGE_THRESHOLD=0.8  # the reliable over-merge knob
OMILOG_SPEAKER_MATCH_THRESHOLD=0.55          # mid-strict cross-conv linking

# Quality scoring
OMILOG_DAILY_SUMMARY_THRESHOLD=0.4
```

Tune from there based on what bothers you most. Don't change everything
at once.
