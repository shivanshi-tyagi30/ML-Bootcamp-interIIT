"""Settings loaded from environment variables (spec Section 4)."""

from __future__ import annotations

import os
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

    # "auto" picks the best option for the machine: GPU -> large-v3 / float16 / beam 5,
    # CPU -> large-v3-turbo / int8 / beam 1 with every core. Any explicit value wins.
    WHISPER_MODEL: str = "auto"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_BEAM_SIZE: int = 0  # 0 = auto
    WHISPER_CPU_THREADS: int = 0  # 0 = all cores

    RECHECK_ENABLED: bool = True
    PARAKEET_MODEL: str = "nvidia/parakeet-tdt-0.6b-v2"
    RECHECK_CONF_THRESHOLD: float = 0.60
    RECHECK_WINDOW_CONF: float = 0.70
    RECHECK_PAD_SEC: float = 2.0
    RECHECK_MAX_SHARE: float = 0.35
    RECHECK_AUDIT_SHARE: float = 0.05
    # Without Parakeet, Whisper words below this probability are marked disputed (0 = off).
    LOWCONF_DISPUTE_THRESHOLD: float = 0.45

    DIARIZATION_ENABLED: bool = True
    DIARIZATION_BACKEND: Literal["auto", "ecapa", "pyannote"] = "auto"
    ECAPA_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_DISTANCE_THRESHOLD: float = 0.6  # higher = fewer speakers; ignored when DIARIZATION_NUM_SPEAKERS is set
    DIARIZATION_NUM_SPEAKERS: int = 0  # 0 = detect; set it when you know how many people spoke (most accurate)
    HF_TOKEN: str = ""

    LLM_BACKEND: Literal["ollama", "vllm"] = "ollama"
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LM2_BASE_URL: str = ""  # optional: separate server for LM2 (vLLM serves one model per server)
    LM1_MODEL: str = "qwen3:14b"
    LM2_MODEL: str = "gemma3:27b"
    LLM_TIMEOUT_SEC: int = 600
    LLM_MAX_RETRIES: int = 2
    LLM_MAX_CONTEXT: int = 32768  # largest context window requested from Ollama
    OLLAMA_KEEP_ALIVE: str = "-1m"  # negative = keep models loaded forever, so no user waits for a reload
    LM2_SCRATCHPAD: bool = True
    # Load every model before the server accepts requests, so no user waits for loading.
    WARMUP_ON_START: bool = True
    # Free LM1's RAM before LM2 runs. Only for big models on a 16 GB laptop: LM1 must then reload for every job.
    LM1_UNLOAD_BEFORE_LM2: bool = False  # false: LM2 reasons silently (about half the output, much faster on CPU)

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

    def whisper_runtime(self) -> dict[str, object]:
        """Resolved Whisper settings (model, device, compute type, beam size, CPU threads)."""
        device = self.WHISPER_DEVICE.lower()
        if device == "auto" or (device == "cuda" and not cuda_available()):  # no GPU: never crash, use the CPU
            device = "cuda" if cuda_available() else "cpu"
        gpu = device == "cuda"
        compute = self.WHISPER_COMPUTE_TYPE.lower()
        if compute == "auto" or (not gpu and "float16" in compute):  # float16 is GPU-only
            compute = "float16" if gpu else "int8"
        return {
            "model": self.WHISPER_MODEL if self.WHISPER_MODEL != "auto" else ("large-v3" if gpu else "large-v3-turbo"),
            "device": device,
            "compute_type": compute,
            "beam_size": self.WHISPER_BEAM_SIZE or (5 if gpu else 1),
            "cpu_threads": self.WHISPER_CPU_THREADS or (os.cpu_count() or 4),
        }

    @property
    def ollama_host(self) -> str:
        """Ollama's native API root (LLM_BASE_URL without the /v1 suffix)."""
        return self.LLM_BASE_URL.rstrip("/").removesuffix("/v1")

    @property
    def jobs_dir(self) -> Path:
        """Folder holding one sub-folder per job."""
        return self.DATA_DIR / "jobs"

    @property
    def db_path(self) -> Path:
        """SQLite database file."""
        return self.DATA_DIR / "trace.db"


@lru_cache
def cuda_available() -> bool:
    """Whether CTranslate2 (faster-whisper's engine) can see a CUDA GPU."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - not installed or no driver
        return False


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
