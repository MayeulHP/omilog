# omilog TODO

Ideas surfaced during development that aren't built yet. Roughly ordered
by likely value; reorder based on what you actually want.

## Wake word → external agent dispatch

When a captured utterance starts with a configurable wake phrase
(`"Okay Jarvis"`, `"Hey omi"`, etc.), route the rest of that utterance
(or the next N seconds of conversation) to an external agent — e.g. the
user's `hermes-agent` running elsewhere on the tailnet.

**Sketch**

- New stage between transcription and LLM extraction: scan the transcript
  for any of `OMILOG_WAKE_PHRASES` (case-insensitive, fuzzy-match because
  Whisper occasionally garbles wake words → use `rapidfuzz` partial ratio).
- On match: isolate the post-wake portion (until next sentence boundary, or
  next N seconds, or end of conversation — configurable).
- POST the isolated text + a brief context window to
  `OMILOG_AGENT_BASE_URL` with an auth token.
- Persist the agent's response in a new `agent_invocations` table keyed
  to the originating conversation.
- Surface in the UI: a 🎯 badge on the conversation row, and a "Routed
  to agent" block in the detail page showing the agent's response.

**Open design questions**

- Multiple wake phrases mapping to different agents (Jarvis vs Hermes vs …)?
- Do we still extract events/actions from the post-wake portion, or skip
  the LLM step for it (delegate everything to the agent)?
- Privacy: agent gets the post-wake text only by default; configurable
  prepend of N seconds of preceding context.
- Latency: should the agent call happen inline (block the pipeline) or
  asynchronously (fire-and-store)?

**Config**

```env
OMILOG_WAKE_PHRASES="okay jarvis,hey omi"        # comma-separated
OMILOG_AGENT_BASE_URL=http://hermes.tailnet:9000
OMILOG_AGENT_TOKEN=hermes-shared-secret
OMILOG_WAKE_FOLLOWUP_SECONDS=30                  # how much post-wake content to grab
OMILOG_WAKE_CONTEXT_SECONDS=0                    # pre-wake context to include
```

---

## Cross-conversation speaker linking (Phase 5)

Phase 4 (diarization) labels speakers **within** one conversation as
`USER`/`S1`/`S2`. There's no link across conversations — the `S1` in
yesterday's conversation isn't tied to the `S1` in today's.

Building this:
- Persist a `speakers` table with `embedding BLOB` + `label TEXT NULL`.
- After diarization, compute a pyannote speaker embedding per cluster.
- Cluster embeddings across all conversations (HDBSCAN or similar) to
  find recurring voices.
- Surface a `/speakers` UI page: "this voice appeared in 8 conversations,
  what should I call them?" → user types name → backfills label to all
  associated `transcript_segments`.

---

## Voice enrollment

Optional add-on to Phase 5. Lets you preemptively label someone:
"upload 30 s of Marie's voice → all future occurrences get tagged `Marie`."

Requires the speaker-linking machinery above to be useful, so build that
first.

---

## Full-text search over transcripts

SQLite FTS5 is the right tool — built-in, fast, supports French
diacritics with the right tokenizer. Expose at `/search?q=…` and via a
small input on the conversation list page. Backfill the FTS table from
existing `transcripts.text` rows.

Becomes essential around 50+ captured conversations. Not before.

---

## Markdown-per-person CRM export

For each recurring `PersonMention.name`, write/append to a
`storage/people/<slug>.md` file matching an Obsidian vault layout:

```markdown
# Marie

- 2026-06-03 [Conversation #abc] — déjeuner discuté, on se voit demain
- 2026-05-21 [Conversation #def] — projet X mentionné
```

Lets the user grep / cross-link in their existing notes setup. ~50 lines,
no new deps.

---

## ICS export for past events

Currently the ICS feed includes both upcoming and past events. Past events
are useful as journal entries — but if a calendar app refuses to import
events older than N days (some do), filter accordingly. Add
`OMILOG_ICS_FEED_MAX_AGE_DAYS` (default ∞).

---

## Prompt tuning against real captures

After 1-2 weeks of real data, false-positive failure modes will be
obvious (e.g. "let's grab lunch sometime" still slips through as an
event). Build a `tests/prompt_eval.py` that runs the extraction prompt
against a held-out set of real transcripts with hand-graded expected
outputs, iterate on the system prompt.

---

## Streaming STT for lower latency

Instead of waiting for a 30-min segment to roll over, stream Opus
packets → whisper.cpp streaming endpoint. UI would update mid-conversation.
Significant architecture change (the runner becomes a real-time
consumer, not a batch processor) — probably not worth it unless you
specifically want live captioning.

---

## Multi-user support

The schema already has `user_id` everywhere — single-user is just a
convention, not a constraint. Adding multi-user means:
- Real signup / invite flow at `/signup`
- Per-user `.env`-style settings (LLM config, calendar tokens) backed by DB
- API + UI auth already supports it

Don't bother unless you actually have a second user. The constraint is
mostly social (who do you trust your wearable audio with?).

---

## Fork + strip the mobile app (long-term)

Currently we use the friend-lite/Chronicle pre-built APK unchanged. That's
the right call for v0/v1 — saves us from owning a React Native build chain
— but it caps what we can do at "whatever Chronicle happens to expose."

Eventually: copy the `app/` directory from Chronicle's repo into a new
`app/` here, strip it to the minimum we actually need, and build our own
APK. This unlocks two real wins:

### Win 1: physical Omi button control

The necklace has a hardware button. Today its events are either ignored
or absorbed by Chronicle's defaults. Owning the app lets us map it
ourselves:

- **single click**: mark this moment — insert a server-side marker event
  with the current timestamp into the active session. Surfaces in the UI
  as a "📌 marked at 14:23" pin on the transcript timeline. Useful for
  "remember this part" without speaking.
- **double click**: toggle capture on/off explicitly (vs. the implicit
  "always on while paired" behavior).
- **long press**: trigger the wake-word → agent flow without saying the
  wake phrase, for noisy environments. The next N seconds get routed to
  the configured agent (see the wake-word entry above).
- **triple click**: deliberate panic-delete of the last conversation
  (because the alternative is fumbling at the phone).

Mapping is configurable in `.env` server-side; the app just POSTs a
button event to `/api/button` and the backend interprets.

### Win 2: server data on the phone

Right now the phone is one-way: it sends audio, never sees what came
back. The app could pull from the same `/api/*` endpoints the web UI
uses and show:

- **today's upcoming events** (extracted from this morning's
  conversation: "you mentioned coffee with Marie at 14:00")
- **open action items** with one-tap "done" / "snooze"
- **a low-effort "what was that?" recall**: tap and see the
  last-5-minutes transcript snippet (useful when someone says "as I was
  saying earlier…" and you have no idea)
- **health indicator**: green when backend is reachable + pipeline is
  caught up, yellow when backlogged, red when the WS keeps disconnecting

### What "stripped down" means

Chronicle ships with a bunch of stuff we don't need:
- Plugin marketplace UI → remove
- Multi-account / cloud sync → remove (single tailnet backend by config)
- Their conversation/extraction UI → remove (we have the web UI)
- Multiple backend support → remove (lock to `OMILOG_BACKEND_URL`
  set at build time or in a single settings screen)

Keep:
- BLE pairing & device management
- Wyoming WS streaming
- JWT auth
- Background service keeping BLE + WS alive
- Foreground service mic permission plumbing (the AndroidManifest piece
  that bit us early on)

### Cost / friction

Real React Native build chain on macOS:
- Xcode for iOS, Android Studio for Android
- `eas build` for cloud signing or local keystore management
- F-Droid distribution if we want to be tidy about it, or signed APK
  hosted on the Pi behind Tailscale for self-distribution

For a project of this scope, lean toward Android-only first; iOS adds a
$99/yr developer membership for non-AltStore distribution, which isn't
worth it for a personal tool.

Don't start until: (a) the spec for what the app should do is stable —
no point shipping a thin client whose backend keeps changing — and (b)
you've actually felt the friction of not having button control / not
seeing server data, for at least a week of real use.
