from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from environment variables prefixed ILOVECSV_ (or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="ILOVECSV_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    database_url: str | None = None  # default: sqlite file inside data_dir
    max_upload_mb: float = 25.0
    max_uncompressed_mb: float = 200.0  # zip-bomb guard for .xlsx
    max_zip_entries: int = 10_000
    max_compression_ratio: float = 200.0

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'ilovecsv.db').as_posix()}"
