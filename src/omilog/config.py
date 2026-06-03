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
