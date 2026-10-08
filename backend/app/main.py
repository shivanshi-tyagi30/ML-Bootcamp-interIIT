"""FastAPI application: CORS, startup recovery, routers (spec Sections 9 and 13)."""

from __future__ import annotations

import asyncio
import logging
import time
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import routes_exports, routes_jobs
from app.api.deps import AppState, Runner
from app.config import Settings, get_settings
from app.core.db import JobDB
from app.core.errors import user_message
from app.core.events import EventBus
from app.llm.client import LLMClient, ollama_status
from app.models.record import MeetingRecord
from app.pipeline import diarize, recheck, stt_whisper
from app.pipeline.runner import Services, run_job


class _JobIdFilter(logging.Filter):
    """Give every log record a job_id so the format string always works."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Default job_id to '-'."""
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return True


def configure_logging() -> None:
    """Spec Section 15 log format."""
    handler = logging.StreamHandler()
    handler.addFilter(_JobIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] job=%(job_id)s %(message)s"))
    root = logging.getLogger()
    if not any(isinstance(f, _JobIdFilter) for h in root.handlers for f in h.filters):
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def scrub_stored_keys(settings: Settings) -> None:
    """Remove API keys that older versions saved in job folders (keys now live in memory only)."""
    from app.core.stages import STAGE_OUTPUT, Stage
    from app.core.storage import read_json, write_json

    for f in settings.jobs_dir.glob(f"*/{STAGE_OUTPUT[Stage.VALIDATING]}"):
        data = read_json(f)
        if isinstance(data, dict) and data.pop("api_key", None) is not None:
            write_json(f, data)


async def warmup(settings: Settings, services: Services) -> list[str]:
    """Load every model at startup; returns what could not be loaded (each step is best effort)."""
    log = logging.getLogger(__name__)
    failed: list[str] = []
    from app.pipeline.vad import silero_available

    if silero_available():  # optional: the hosted website uses the energy gate only
        try:
            from app.pipeline.vad import silero_regions
            import numpy as np

            await asyncio.to_thread(silero_regions, np.zeros(16000, dtype=np.float32), 16000)
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up: Silero VAD not loaded (%r)", e)
            failed.append("Silero VAD")
    if services.stt is None and settings.STT_BACKEND == "whisper":
        try:
            await asyncio.to_thread(stt_whisper.default_stt(settings)._load)
            log.info("warm-up: Whisper loaded")
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up: Whisper not loaded (%r)", e)
            failed.append("Whisper")
    if services.diarizer is None and settings.DIARIZATION_ENABLED and settings.DIARIZATION_BACKEND in ("auto", "ecapa"):
        try:
            d = diarize.default_ecapa_diarizer(settings)
            await asyncio.to_thread(d.identifier._load)
            log.info("warm-up: speaker model loaded")
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up: speaker model not loaded (%r)", e)
            failed.append("speaker model")
    if services.llm is None and not settings.GEMINI_API_KEY and settings.LOCAL_LLM_ENABLED:  # cloud: nothing to load
        client = LLMClient(settings.LLM_BASE_URL, settings.LLM_BACKEND, settings.LLM_TIMEOUT_SEC,
                           keep_alive=settings.OLLAMA_KEEP_ALIVE)
        # One-model-at-a-time mode: only LM1 (used first in every job) is preloaded; LM2 loads when needed.
        models = [settings.LM1_MODEL] if settings.LM1_UNLOAD_BEFORE_LM2 else [settings.LM1_MODEL, settings.LM2_MODEL]
        for m in dict.fromkeys(models):
            if not await client.preload(m):
                failed.append(m)
    return failed


def create_app(settings: Settings | None = None, services: Services | None = None, runner: Runner | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings, fake services or a fake runner."""
    settings = settings or get_settings()
    services = services or Services()
    db = JobDB(settings.db_path)
    bus = EventBus()

    async def default_runner(job_id: str) -> None:
        """Run the real pipeline."""
        await run_job(job_id, settings, db, bus, services)

    state = AppState(settings=settings, db=db, bus=bus, services=services, runner=runner or default_runner)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Create folders and the DB; fail jobs interrupted by a restart."""
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        await db.init()
        msg = user_message("E_INTERNAL")  # detail column explains the restart
        for job_id in await db.mark_interrupted("E_INTERNAL", msg):
            logging.getLogger(__name__).warning("server restarted; job marked failed", extra={"job_id": job_id})
        scrub_stored_keys(settings)
        state.not_loaded = []
        if settings.WARMUP_ON_START:
            t0 = time.perf_counter()
            state.not_loaded = await warmup(settings, services)  # uploads are accepted only after this
            log = logging.getLogger(__name__)
            if state.not_loaded:
                log.warning("started, but NOT loaded: %s (run python -m app.prefetch)", ", ".join(state.not_loaded))
            else:
                log.info("all models loaded in %.0f s; ready", time.perf_counter() - t0)
        state.ready = not state.not_loaded
        yield

    configure_logging()
    app = FastAPI(title="TRACE backend", version="1.0", lifespan=lifespan)
    app.state.trace = state
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Range", "Accept-Ranges"],
    )
    app.include_router(routes_jobs.router)
    app.include_router(routes_exports.router)

    @app.get("/api/schema/record", tags=["meta"])
    async def record_schema() -> dict[str, Any]:
        """JSON Schema of MeetingRecord."""
        return MeetingRecord.model_json_schema()

    @app.get("/api/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        """Liveness plus which models are loaded."""
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except ImportError:
            gpu = False
        loaded = lambda m: bool(m and getattr(m, "loaded", False))  # noqa: E731
        models = list(dict.fromkeys([settings.LM1_MODEL, settings.LM2_MODEL]))
        if settings.GEMINI_API_KEY:  # cloud model configured on the server
            llm = {"reachable": None, "missing": [], "cloud": settings.GEMINI_MODEL}
        elif not settings.LOCAL_LLM_ENABLED:  # hosted website: each user brings a key
            llm = {"reachable": None, "missing": [], "key_required": True}
        elif settings.LLM_BACKEND == "ollama":
            llm = await ollama_status(settings.ollama_host, models)
        else:
            llm = {"reachable": None, "missing": []}
        rt = settings.whisper_runtime()
        return {
            "status": "ok",
            "ready": state.ready,
            "not_loaded": state.not_loaded,
            "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "llm": {"backend": settings.LLM_BACKEND, "host": settings.ollama_host, **llm},
            "whisper": ({"model": settings.GROQ_STT_MODEL, "device": "Groq cloud", "compute_type": "-", "beam_size": 1}
                        if settings.STT_BACKEND == "groq"
                        else {k: rt[k] for k in ("model", "device", "compute_type", "beam_size")}),
            "setup_problems": settings.setup_problems(),
            "models": {
                "stt": loaded(services.stt or stt_whisper._default),
                "stt_check": loaded(services.recheck_asr or recheck._default),
                "diarization": loaded(services.diarizer or diarize._default),
                "lm1": settings.LM1_MODEL,
                "lm2": settings.LM2_MODEL,
            },
            "gpu": gpu,
        }

    if settings.SERVE_FRONTEND:
        mount_frontend(app, settings.FRONTEND_DIST)
    return app


def mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built single-page frontend at / (API routes registered earlier take precedence)."""
    log = logging.getLogger(__name__)
    if not (dist / "index.html").exists():
        log.warning("SERVE_FRONTEND is on but %s has no index.html; run `npm run build` in frontend/", dist)
        return
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    log.info("serving the website from %s", dist)


app = create_app()
