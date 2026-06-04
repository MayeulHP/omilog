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
    # Cosine similarity threshold above which two embeddings are treated as
    # the same speaker. Range 0..1; 0.6 is a defensible default for NeMo
    # TitaNet on conversational audio. Lower → more merging (false positives);
    # higher → more new-speaker rows (false negatives).
    speaker_match_threshold: float = 0.6

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
