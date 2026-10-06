"""Settings loaded from environment variables (spec Section 4)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Names match the environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    DATA_DIR: Path = Path("./data")
    MAX_FILE_MB: int = 200
    MAX_DURATION_MIN: int = 120
    MIN_SPEECH_SEC: float = 2.0
    ALLOWED_EXT: str = "mp3,wav,m4a,ogg,flac,webm,mp4,aac"

    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"

    RECHECK_ENABLED: bool = True
    PARAKEET_MODEL: str = "nvidia/parakeet-tdt-0.6b-v2"
    RECHECK_CONF_THRESHOLD: float = 0.60
    RECHECK_WINDOW_CONF: float = 0.70
    RECHECK_PAD_SEC: float = 2.0
    RECHECK_MAX_SHARE: float = 0.35
    RECHECK_AUDIT_SHARE: float = 0.05

    DIARIZATION_ENABLED: bool = False
    HF_TOKEN: str = ""

    LLM_BACKEND: Literal["ollama", "vllm"] = "ollama"
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LM2_BASE_URL: str = ""  # optional: separate server for LM2 (vLLM serves one model per server)
    LM1_MODEL: str = "qwen3:14b"
    LM2_MODEL: str = "gemma3:27b"
    LLM_TIMEOUT_SEC: int = 600
    LLM_MAX_RETRIES: int = 2

    LM1_WINDOW_SEGMENTS: int = 40
    LM1_CONTEXT_SEGMENTS: int = 5
    LM1_MIN_CONFIDENCE: float = 0.70
    SOUND_ALIKE_THRESHOLD: float = 0.75
    QUOTE_MATCH_THRESHOLD: int = 90
    LM2_MAX_INPUT_TOKENS: int = 24000

    CORS_ORIGINS: str = "http://localhost:5173"
    MAX_CONCURRENT_UPLOADS: int = 3

    @property
    def allowed_ext(self) -> set[str]:
        """Allowed upload extensions, lowercase, without dots."""
        return {e.strip().lower().lstrip(".") for e in self.ALLOWED_EXT.split(",") if e.strip()}

    @property
    def cors_origins(self) -> list[str]:
        """Allowed CORS origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def jobs_dir(self) -> Path:
        """Folder holding one sub-folder per job."""
        return self.DATA_DIR / "jobs"

    @property
    def db_path(self) -> Path:
        """SQLite database file."""
        return self.DATA_DIR / "trace.db"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
