from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OMILOG_",
        case_sensitive=False,
        extra="ignore",
    )

    username: str = "you"
    password_hash: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    storage_dir: Path = Path("storage")
    db_path: Path = Path("omilog.db")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # STT — whisper.cpp server on the GPU box. Empty STT_BASE_URL disables
    # the pipeline worker (useful for the no-GPU-available case during dev).
    stt_base_url: str = ""
    stt_inference_path: str = "/inference"
    stt_language: str = "auto"
    stt_timeout_s: float = 120.0
    stt_model_name: str = "whisper-large-v3-turbo"
    # Optional decoding hints, passed straight through to whisper.cpp.
    #   initial_prompt biases vocabulary toward your usage (names of frequent
    #     contacts, technical terms, dominant language). Short and concrete is
    #     better than long and abstract.
    #   temperature 0 = deterministic; 0.2 gives Whisper room to back out of a
    #     bad guess on noisy audio.
    stt_initial_prompt: str = ""
    stt_temperature: float = 0.0

    # Pipeline runner cadence.
    pipeline_poll_seconds: float = 2.0

    # Run the pipeline worker in a separate thread (with its own asyncio
    # loop) instead of the same loop as the web server. Default True so
    # the web UI stays responsive on small boxes while STT / diarize /
    # LLM work is in flight — without this, even though every pipeline
    # step properly awaits, the sync chunks between awaits (multipart
    # encoding, DB roundtrips, JSON parsing) hold the GIL for long
    # enough to make a Pi feel locked up. Set to False to revert to the
    # legacy single-loop behavior — useful only for debugging.
    pipeline_in_thread: bool = True

    # WS rollover. If the BLE phone holds a single WS open for hours, we don't
    # want a single 12-hour file — we close the current segment every N seconds
    # and start a new one, so the pipeline can process chunks throughout the
    # day rather than once at session end. Set to 0 to disable rollover.
    ws_rollover_seconds: float = 1800.0    # 30 min
    # Receive timeout used to wake up the WS loop periodically so the rollover
    # check can fire even when the audio stream is bursty.
    ws_receive_timeout_seconds: float = 5.0

    # LLM extraction — llama.cpp server, OpenAI-compatible API. Empty
    # LLM_BASE_URL disables the LLM stage; sessions stay in pending_llm.
    llm_base_url: str = ""
    llm_model: str = "qwen3.6-35b-a3b"
    llm_temperature: float = 0.1
    # 4096 is comfortable for a typical conversation extraction (title +
    # summary + a few events/actions/people). Older 2048 default was tight —
    # long French conversations would truncate mid-output and fail JSON parse.
    llm_max_tokens: int = 4096
    llm_timeout_s: float = 180.0
    # Optional hint baked into the LLM system prompt. Free-text like "French"
    # or "Spanish". Empty (default) keeps the prompt language-neutral so the
    # model adapts to whatever language is in the transcript. Whisper handles
    # actual language detection from audio independently.
    llm_primary_language: str = ""
    # If this file exists, its contents replace the default LLM system prompt
    # entirely. Edit via the /config/prompt UI, or by hand in any text editor.
    # When set, `llm_primary_language` is ignored — your prompt is used
    # verbatim, so include language-specific wording yourself if you need it.
    llm_system_prompt_file: Path = Path("prompts/system_prompt.txt")

    # Per-category extraction toggles. When false, the LLM prompt's schema
    # omits the corresponding section (saves output tokens) AND the runner
    # ignores the field even if the model returns it anyway. Disabling a
    # category only affects FUTURE conversations — historical extractions
    # stay in the DB and remain visible in the UI. Toggle these to match
    # what you actually use: if you never look at /events, turn calendar
    # off and stop paying for the extraction.
    extract_calendar_events: bool = True
    extract_action_items: bool = True
    extract_decisions: bool = True
    extract_people_mentioned: bool = True
    extract_topics: bool = True

    # For resolving "demain"/"ce soir"/etc. against a real date.
    local_timezone: str = "Europe/Paris"

    # Speaker diarization (Phase 4). Optional; needs the diarization extra
    # installed and model files downloaded via
    # `scripts/download_diarization_models.py`. All inference is local —
    # nothing about the audio leaves the tailnet. Failures here never block
    # the pipeline; transcripts proceed to LLM without speaker labels.
    diarization_enabled: bool = False
    diarization_models_dir: Path = Path("models")
    diarization_segmentation_model: Path = Path(
        "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    )
    diarization_embedding_model: Path = Path(
        "models/nemo_en_titanet_small.onnx"
    )
    # 0.3s minimum speech, 0.5s minimum silence — sherpa-onnx defaults that
    # work well for conversational French.
    diarization_min_speech_seconds: float = 0.3
    diarization_min_silence_seconds: float = 0.5
    # Force a specific number of speakers in each conversation. -1 (default)
    # lets sherpa-onnx auto-detect via the cluster threshold below; a positive
    # value pins the clusterer to exactly that many clusters. Useful when you
    # know the typical conversation has, say, 2-4 people: setting this to 3
    # collapses the kind of over-segmentation where short responses ("d'accord",
    # "ok ça marche") get their own cluster each.
    diarization_num_clusters: int = -1
    # Cosine-similarity threshold used by the FastClustering algorithm when
    # ``num_clusters=-1``. Pairs of embeddings closer than this get merged.
    # Range 0..1. Default 0.5 matches sherpa-onnx's internal default. NOTE:
    # in practice this knob has surprisingly little effect on the cluster
    # count (sherpa-onnx's auto-K seems to ignore extreme values). Use
    # ``diarization_post_merge_threshold`` below for a reliable second-pass
    # merge that runs in our own Python.
    diarization_cluster_threshold: float = 0.5
    # Cosine-similarity threshold for the in-process post-merge that runs
    # AFTER sherpa-onnx. For each pair of clusters the diarizer returned,
    # we compute their per-cluster embeddings and fold them into one if
    # their cosine similarity is at least this much. 1.0 (default) disables
    # the merge entirely; values around 0.7-0.85 are the useful range for
    # the "9 clusters but really 2 people" case. Lower = more aggressive
    # merging; if distinct people start getting folded together, raise.
    # Unlike num_clusters, this scales naturally: a 5-person meeting gets
    # 5 clusters because their embeddings are dissimilar; a 2-person chat
    # gets 2 because the over-split clusters fold back together.
    diarization_post_merge_threshold: float = 1.0
    # ONNX Runtime intra-op thread cap. Without this, ORT uses every core
    # by default — on a 4-core Pi that saturates the box during diarization
    # and the asyncio web server starves for scheduling time, so the UI
    # freezes for minutes per conversation. 2 threads leaves CPU headroom
    # for the web loop without making diarization noticeably slower (the
    # workload is mostly memory-bound past 2 threads anyway). Bump on bigger
    # hosts.
    diarization_num_threads: int = 2
    # Cosine similarity threshold above which two embeddings are treated as
    # the same speaker. Range 0..1; 0.6 is a defensible default for NeMo
    # TitaNet on conversational audio. Lower → more merging (false positives);
    # higher → more new-speaker rows (false negatives).
    speaker_match_threshold: float = 0.6

    # Daily-summary quality cutoff. Conversations whose effective_quality
    # (override-or-score) is below this don't feed into the day's narrative.
    # 0.3 = exclude noise but keep normal+ daily chatter. Bump to 0.5 if your
    # days produce too much narrative-padding from ordinary small talk.
    daily_summary_threshold: float = 0.3

    # Audio retention rotation. After this many days, .opus files for done
    # conversations get auto-deleted (DB row + transcript + extraction kept,
    # only the audio blob goes). Set to 0 to disable rotation entirely; 30
    # is a reasonable starting point if storage is constrained. Archived
    # conversations (📌 pinned via the UI) are exempt no matter how old.
    audio_retention_days: int = 0

    # Web UI session cookie. Set OMILOG_COOKIE_SECURE=true once Caddy/Tailscale
    # serve fronts the app on HTTPS (default false for local-http dev).
    cookie_name: str = "omilog_token"
    cookie_secure: bool = False

    # ICS calendar feed. Empty token disables the feed entirely; calendar apps
    # that subscribe to /calendar.ics?token=<token> get a refreshing iCal.
    # Generate via: python -c 'import secrets; print(secrets.token_urlsafe(32))'
    ics_feed_token: str = ""
    ics_prodid: str = "-//omilog//EN"
    ics_calname: str = "omilog"
    # Events below this confidence are kept off the feed by default — overridable
    # per-request via ?min_confidence=N. The download-per-event endpoint always
    # exports regardless of confidence (it's a deliberate user click).
    ics_feed_min_confidence: float = 0.5

    # VAD / segmentation. Long BLE captures get split into conversation-sized
    # children at silence gaps >= vad_gap_seconds. Pure-silence captures get
    # marked status=silent and their file deleted.
    vad_enabled: bool = True
    vad_gap_seconds: float = 60.0
    # -40 dB: chosen for compressed Opus from a wearable mic where speech often
    # lands at -35..-40 dB after dynamic range compression. -30 was too eager
    # and clipped quiet speech.
    vad_threshold_db: float = -40.0
    vad_min_silence_seconds: float = 0.5
    # Tight margin around conversations so we don't cut off the first/last word.
    vad_pad_seconds: float = 0.4
    # Audio bitrate for extracted Opus children. 32k is comfortable for speech.
    vad_child_bitrate: str = "32k"


settings = Settings()


def assert_runtime_secrets() -> None:
    """Called on startup. Defaults are blank so tests can import without a .env;
    at boot we want a loud error rather than handing out forged tokens."""
    missing = [
        name
        for name, value in (
            ("OMILOG_PASSWORD_HASH", settings.password_hash),
            ("OMILOG_JWT_SECRET", settings.jwt_secret),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.template to .env and fill them in."
        )
