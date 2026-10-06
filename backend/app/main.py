"""FastAPI application: CORS, startup recovery, routers (spec Sections 9 and 13)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_exports, routes_jobs
from app.api.deps import AppState, Runner
from app.config import Settings, get_settings
from app.core.db import JobDB
from app.core.errors import user_message
from app.core.events import EventBus
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
        msg = user_message("E_INTERNAL")
        for job_id in await db.mark_interrupted("E_INTERNAL", msg):
            logging.getLogger(__name__).warning("server restarted; job marked failed", extra={"job_id": job_id})
        yield

    configure_logging()
    app = FastAPI(title="TRACE backend", version="1.0", lifespan=lifespan)
    app.state.trace = state
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"],
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
        return {
            "status": "ok",
            "models": {
                "stt": loaded(services.stt or stt_whisper._default),
                "stt_check": loaded(services.recheck_asr or recheck._default),
                "diarization": loaded(services.diarizer or diarize._default),
                "lm1": settings.LM1_MODEL,
                "lm2": settings.LM2_MODEL,
            },
            "gpu": gpu,
        }

    return app


app = create_app()
