# omilog TODO

Ideas surfaced during development that aren't built yet. Roughly ordered
by likely value; reorder based on what you actually want.

## Cross-conversation speaker linking (Phase 5)

Diarization (already shipped via sherpa-onnx) labels speakers **within**
one conversation as `USER`/`S1`/`S2`. There's no link across conversations:
the `S1` in yesterday's conversation isn't tied to the `S1` in today's.
This blocks the markdown-per-person CRM from being properly useful.

**Mechanics**

sherpa-onnx already computes a ~192-D NeMo TitaNet embedding per detected
speaker cluster — we just don't currently capture it. Adding the linking:

1. **Capture embeddings**: extend `pipeline/diarize.py` to grab the
   per-cluster embeddings out of sherpa-onnx (via `SpeakerEmbeddingExtractor`
   directly on the audio slice of each turn, since the
   `OfflineSpeakerDiarization` wrapper hides them).
2. **Persist**: new `speakers` table with `id`, `user_id`, `name` (nullable
   until labeled), `embedding BLOB`, `created_at`. Add a speaker reference
   to each transcript segment (either a new `transcript_segments` table, or
   inline in `segments_json`).
3. **Match**: on each new conversation, cosine-similarity each detected
   cluster's embedding against every stored labeled speaker. If max > ~0.6
   (typical threshold for these models), reuse that speaker id + name.
   Otherwise create a new unlabeled speaker row.
4. **Label**: UI affordance on the conversation detail page — click any
   `S1` / `S2` line, type a name, save. Backend updates the matched speaker
   row's `name` and re-labels its prior occurrences retroactively.
5. **Browse**: `/speakers` page listing all known voices with conversation
   counts and click-to-rename / merge.

**Effort estimate**: ~one focused day total — 30 min schema, 2 hours
embedding capture (longest part: sherpa-onnx's embedding API surface is
sparsely documented and likely needs a source read), 1 hour matching, 2-3
hours UI, 1 hour tests.

**Known caveat**: NeMo TitaNet was trained on English speakers. For French
voices it works but may merge two same-gender French speakers more often
than English ones. If that's a problem, swap to a different embedding model
from sherpa-onnx's catalogue — the linking logic is decoupled from the
embedding source.

---

## Voice enrollment

Optional add-on to the speaker linking above. Lets you preemptively label
someone: "upload 30 s of Marie's voice → all future occurrences get tagged
`Marie`." Requires the linking machinery above to be useful, so build that
first.

---

## Full-text search over transcripts

SQLite FTS5 is the right tool — built-in, fast, supports French
diacritics with the right tokenizer. Expose at `/search?q=…` and via a
small input on the conversation list page. Backfill the FTS table from
existing `transcripts.text` rows.

Effort: ~2 hours. Becomes essential around 30-50 captured conversations;
the conversation list is still scrollable below that.

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
no new deps. Works as-is on top of `PersonMention` rows, but gets
materially better once cross-conversation speaker linking lands (then we
can say "Marie *said* X" rather than just "Marie was mentioned in a
conversation involving X").

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

The system prompt is now editable per-deploy at `/config/prompt`
(landed in v0.1.x). Eval harness still TODO — needs ground-truth
transcripts which only emerge from real use.

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

## Fork + strip the mobile app (probably 2–3 days, not a week)

Currently we use the friend-lite/Chronicle pre-built APK unchanged. Audited
the repo at `~/Coding/chronicle` and the situation is friendlier than
expected:

- **App size**: ~4,826 LOC across 33 TS/TSX files. Small for a real React
  Native app.
- **Stack**: Expo SDK 53 managed (prebuild on demand), RN 0.79.6, React 19,
  new architecture on. No `android/` or `ios/` dirs checked in.
- **CI we inherit free**: `.github/workflows/android-apk-build.yml` already
  builds and releases an APK on every push to `main` touching `app/**`,
  via `eas build --platform android --profile local --local`.
- **Cruft to strip**: basically none. No plugin marketplace, no
  multi-account code. The "conversation UI to remove" is essentially
  `app/index.tsx` (~458 lines) plus its hooks.

**Concrete first move (proves the build chain before committing to a fork)**:

```bash
cd ~/Coding/chronicle/app
npm ci
npx expo prebuild --platform android
cd android
./gradlew assembleDebug
# → android/app/build/outputs/apk/debug/app-debug.apk
```

If that succeeds in under an hour on the maintainer's Mac, the rest is
realistic.

**Real concerns to settle before starting**:

- **Can the Omi button be captured at all** from a third-party app, or
  does pressing it just power-cycle the necklace? That's the showstopper
  question. Read `friend-lite-react-native`'s source for any button event
  characteristic; if absent, drop down to `react-native-ble-plx` directly
  (~½ day extra).
- `@siteed/expo-audio-studio`, `@notifee/react-native`, and
  `react-native-ble-plx` all need `expo prebuild` (config plugins in
  `app.json`). No Expo Go workflow.
- `with-ws-fgs.js` patches the AndroidManifest for foreground services;
  keep it.

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

- **Tooling on macOS**: Android Studio (for the SDK + emulator), JDK 17,
  `node` 20+. Xcode only if iOS is in scope, which it isn't.
- **Build**: `eas build --local` produces an unsigned debug APK without
  needing an EAS account. For releases, either generate a local keystore
  and self-sign, or use EAS cloud builds (free tier covers personal use).
- **Distribution**: host the APK behind Tailscale on the Pi itself
  (`http://pi.tailnet/omilog.apk`), or F-Droid if you want to be tidy.

Android-only first; iOS adds a $99/yr developer membership for
non-AltStore distribution, which isn't worth it for a personal tool.

Trigger to start: a concrete moment where you reach for your phone to do
something and think "this should just be a button press on the necklace,"
**plus** confirmation that the Omi button is actually capturable from a
third-party app (it may just be a power button at the BLE protocol
level).
