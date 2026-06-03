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

    username: str = "mayeul"
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

    # Pipeline runner cadence.
    pipeline_poll_seconds: float = 2.0

    # LLM extraction — llama.cpp server, OpenAI-compatible API. Empty
    # LLM_BASE_URL disables the LLM stage; sessions stay in pending_llm.
    llm_base_url: str = ""
    llm_model: str = "qwen3.6-35b-a3b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 180.0

    # For resolving "demain"/"ce soir"/etc. against a real date.
    local_timezone: str = "Europe/Paris"

    # Web UI session cookie. Set OMILOG_COOKIE_SECURE=true once Caddy/Tailscale
    # serve fronts the app on HTTPS (default false for local-http dev).
    cookie_name: str = "omilog_token"
    cookie_secure: bool = False

    # VAD / segmentation. Long BLE captures get split into conversation-sized
    # children at silence gaps >= vad_gap_seconds. Pure-silence captures get
    # marked status=silent and their file deleted.
    vad_enabled: bool = True
    vad_gap_seconds: float = 60.0
    vad_threshold_db: float = -30.0     # ffmpeg silencedetect noise threshold
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
