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
