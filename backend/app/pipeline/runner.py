"""The job state machine: runs stages in order with resume, progress and typed failures (spec Section 13)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import Settings
from app.core.db import JobDB
from app.core.errors import JobCancelled, PipelineError, user_message
from app.core.events import EventBus
from app.core.stages import PIPELINE_STAGES, STAGE_MESSAGES, STAGE_OUTPUT, Stage, overall_progress
from app.core.storage import job_dir, read_json, write_json
from app.exports import render_exports
from app.llm.client import JSONLLM, LLMClient
from app.models.record import MeetingRecord
from app.pipeline.assemble import assemble_raw
from app.pipeline.diarize import diarize
from app.pipeline.guard import guard
from app.pipeline.lm1_refine import refine
from app.pipeline.lm1_vocabulary import build_vocabulary
from app.pipeline.lm2_document import document
from app.pipeline.normalize import normalize
from app.pipeline.recheck import recheck
from app.pipeline.stt_whisper import transcribe
from app.pipeline.vad import speech_check
from app.pipeline.validate import validate
from app.pipeline.verify import verify_record

log = logging.getLogger(__name__)

StageFn = Callable[["JobContext"], Awaitable[None]]


@dataclass
class Services:
    """Model backends. None means "use the real default" (loaded lazily); tests inject fakes."""

    stt: Any = None
    recheck_asr: Any = None
    diarizer: Any = None
    vad: Any = None
    llm: JSONLLM | None = None


class JobLog(logging.LoggerAdapter):
    """Logger that adds job_id to every record."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        """Attach the job id."""
        kwargs.setdefault("extra", {})["job_id"] = self.extra["job_id"]
        return msg, kwargs


class JobContext:
    """Everything a stage needs: paths, settings, services, and progress reporting."""

    def __init__(self, job_id: str, settings: Settings, db: JobDB, bus: EventBus, services: Services) -> None:
        """Load the job's upload metadata."""
        self.job_id = job_id
        self.settings = settings
        self.db = db
        self.bus = bus
        self.services = services
        self.dir = job_dir(settings.jobs_dir, job_id)
        self.upload: dict[str, Any] = read_json(self.dir / STAGE_OUTPUT[Stage.VALIDATING]) or {}
        self.title: str = self.upload.get("title", "")
        self.stage: Stage = Stage.QUEUED
        self.warnings: list[str] = []
        self.log = JobLog(log, {"job_id": job_id})
        self._llm: JSONLLM | None = services.llm
        self._timings: dict[str, float] = read_json(self.dir / "timings.json") or {}
        self.cancel_event = CANCEL_EVENTS.setdefault(job_id, threading.Event())
        self.loop: asyncio.AbstractEventLoop | None = None

    def cancel_requested(self) -> bool:
        """Checked by worker threads at safe points."""
        return self.cancel_event.is_set()

    # ---- paths and files
    @property
    def original_path(self) -> Path:
        """The stored upload."""
        return self.dir / f"original.{self.upload.get('ext', 'bin')}"

    @property
    def wav_path(self) -> Path:
        """Normalized 16 kHz mono audio."""
        return self.dir / "audio_16k.wav"

    @staticmethod
    def output_name(stage: Stage) -> str:
        """Output file of a stage, relative to the job folder."""
        return STAGE_OUTPUT[stage]

    def read(self, name: str) -> Any:
        """Read a JSON file from the job folder (None if missing/invalid)."""
        return read_json(self.dir / name)

    def write(self, name: str, data: Any) -> None:
        """Atomically write a JSON file into the job folder."""
        write_json(self.dir / name, data)

    def stage_done(self, stage: Stage) -> bool:
        """Resume rule: the stage's output exists and is valid JSON."""
        return self.read(self.output_name(stage)) is not None

    @property
    def llm(self) -> JSONLLM:
        """LLM client (created on first use): the cloud model when an API key was given, else local Ollama.

        Never both: with a key every LM call goes to the cloud, and a cloud failure is shown, not hidden behind
        the much slower local model.
        """
        if self._llm is None:
            s = self.settings
            api_key = (JOB_API_KEYS.get(self.job_id) or s.GEMINI_API_KEY or "").strip()
            if api_key:
                from app.llm.client import GeminiClient

                self._llm = GeminiClient(api_key=api_key, model=s.GEMINI_MODEL, timeout=s.CLOUD_TIMEOUT_SEC,
                                         base_url=s.CLOUD_BASE_URL, reasoning_effort=s.CLOUD_REASONING_EFFORT)
            else:
                self._llm = LLMClient(
                    s.LLM_BASE_URL, s.LLM_BACKEND, s.LLM_TIMEOUT_SEC, {s.LM2_MODEL: s.LM2_BASE_URL},
                    max_context=s.LLM_MAX_CONTEXT, keep_alive=s.OLLAMA_KEEP_ALIVE,
                )
        return self._llm

    @property
    def cloud(self) -> bool:
        """Whether the cloud model handles LM1 and LM2 for this job."""
        from app.llm.client import GeminiClient

        return isinstance(self.llm, GeminiClient)

    def lm1_window(self) -> int:
        """Segments per LM1 call: a cloud model reads a whole long meeting at once."""
        return max(self.settings.LM1_WINDOW_SEGMENTS, CLOUD_LM1_WINDOW) if self.cloud else self.settings.LM1_WINDOW_SEGMENTS

    def add_warning(self, code: str) -> None:
        """Record a warning such as W_NON_ENGLISH (shown in events and meta)."""
        if code not in self.warnings:
            self.warnings.append(code)

    # ---- progress and status
    def event(self, status: str, within: float = 0.0, message: str | None = None, **extra: Any) -> dict[str, Any]:
        """Build an SSE progress payload."""
        return {
            "job_id": self.job_id,
            "stage": self.stage.value,
            "status": status,
            "progress": overall_progress(self.stage, within),
            "message": message or STAGE_MESSAGES.get(self.stage, ""),
            "raw_ready": (self.dir / STAGE_OUTPUT[Stage.RAW_SAVED]).exists(),
            "refined_ready": (self.dir / STAGE_OUTPUT[Stage.GUARDING]).exists(),
            "record_ready": (self.dir / STAGE_OUTPUT[Stage.VERIFYING]).exists(),
            "warnings": list(self.warnings),
            **extra,
        }

    async def queued(self) -> None:
        """Announce that the job waits for the GPU."""
        self.stage = Stage.QUEUED
        await self.db.update(self.job_id, status="queued", stage=Stage.QUEUED.value)
        self.bus.publish(self.job_id, self.event("queued"))

    async def set_stage(self, stage: Stage) -> None:
        """Enter a stage: update SQLite and push an event."""
        self.stage = stage
        await self.db.update(self.job_id, status="running", stage=stage.value)
        self.bus.publish(self.job_id, self.event("running"))

    async def progress(self, within: float, message: str) -> None:
        """Report progress inside the current stage."""
        self.bus.publish(self.job_id, self.event("running", within, message))

    def progress_threadsafe(self, within: float, message: str) -> None:
        """Report progress from a worker thread (e.g. while Whisper decodes)."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.bus.publish, self.job_id, self.event("running", within, message))

    def record_timing(self, stage: Stage, seconds: float) -> None:
        """Store a stage duration in timings.json."""
        self._timings[stage.value] = round(seconds, 2)
        self.write("timings.json", self._timings)

    async def complete(self) -> None:
        """Mark the job completed."""
        self.stage = Stage.COMPLETED
        await self.db.update(self.job_id, status="completed", stage=Stage.COMPLETED.value, error_code=None,
                             error_message=None, error_detail=None)
        self.bus.publish(self.job_id, self.event("completed"))

    async def fail(self, code: str, stage: Stage, message: str, detail: str = "") -> None:
        """Mark the job failed; earlier outputs stay available."""
        self.log.error("failed at %s: %s %s", stage.value, code, detail)
        failed_at = stage
        hint = detail_for_user(detail)
        await self.db.update(self.job_id, status="failed", stage=Stage.FAILED.value, error_code=code,
                             error_message=message, error_detail=hint)
        self.stage = failed_at
        self.bus.publish(self.job_id, self.event("failed", error_code=code, error_message=message,
                                                 error_detail=hint, failed_stage=failed_at.value))


async def render(ctx: JobContext) -> None:
    """RENDERING stage: exports generated only from record.json."""
    try:
        rec = MeetingRecord(**ctx.read(ctx.output_name(Stage.VERIFYING)))
        await asyncio.to_thread(render_exports, rec, ctx.dir / "exports")
    except Exception as e:  # noqa: BLE001
        raise PipelineError("E_RENDER_FAILED", Stage.RENDERING, repr(e)) from e


PIPELINE: list[tuple[Stage, StageFn]] = [
    (Stage.VALIDATING, validate), (Stage.NORMALIZING, normalize),
    (Stage.SPEECH_CHECK, speech_check), (Stage.TRANSCRIBING, transcribe),
    (Stage.RECHECKING, recheck), (Stage.DIARIZING, diarize),
    (Stage.RAW_SAVED, assemble_raw), (Stage.VOCABULARY, build_vocabulary),
    (Stage.REFINING, refine), (Stage.GUARDING, guard),
    (Stage.DOCUMENTING, document), (Stage.VERIFYING, verify_record),
    (Stage.RENDERING, render),
]
assert [s for s, _ in PIPELINE] == PIPELINE_STAGES

# One job on the GPU at a time. Created per event loop (tests use several loops).
_locks: dict[int, asyncio.Semaphore] = {}

# job_id -> cloud API key pasted in the app for that job. Memory only; never written to disk.
JOB_API_KEYS: dict[str, str] = {}
CLOUD_LM1_WINDOW = 400  # ~30-40 minutes of meeting in one refine call on a cloud model
CLOUD_LM2_INPUT_TOKENS = 200_000  # cloud models read very long transcripts in one call


def remember_api_key(job_id: str, key: str) -> None:
    """Keep a job's cloud API key in memory (also used by Retry while the server runs)."""
    JOB_API_KEYS[job_id] = key


def has_api_key(job_id: str) -> bool:
    """Whether a cloud API key for this job is in memory (lost when the server restarts)."""
    return bool(JOB_API_KEYS.get(job_id))


# job_id -> event set when the user cancels; worker threads poll it.
CANCEL_EVENTS: dict[str, threading.Event] = {}


def detail_for_user(detail: str) -> str | None:
    """A short technical reason shown under the error message (no stack traces)."""
    detail = (detail or "").strip()
    if not detail:
        return None
    return detail if len(detail) <= 300 else detail[:297] + "..."


def request_cancel(job_id: str) -> None:
    """Ask a running job to stop at the next safe point."""
    CANCEL_EVENTS.setdefault(job_id, threading.Event()).set()


def gpu_lock() -> asyncio.Semaphore:
    """The global GPU semaphore for the running event loop."""
    loop = asyncio.get_running_loop()
    return _locks.setdefault(id(loop), asyncio.Semaphore(1))


async def run_job(job_id: str, settings: Settings, db: JobDB, bus: EventBus, services: Services) -> None:
    """Run (or resume) a job through every stage."""
    CANCEL_EVENTS[job_id] = threading.Event()  # fresh flag for this run (also on Retry)
    ctx = JobContext(job_id, settings, db, bus, services)
    ctx.loop = asyncio.get_running_loop()
    whisper = ctx.read(STAGE_OUTPUT[Stage.TRANSCRIBING]) or {}
    for w in whisper.get("warnings", []):
        ctx.add_warning(w)
    try:
        await ctx.queued()
        async with gpu_lock():
            for stage, fn in PIPELINE:
                if ctx.cancel_requested():
                    raise JobCancelled()
                if ctx.stage_done(stage):
                    continue
                await ctx.set_stage(stage)
                t0 = time.perf_counter()
                ctx.log.info("stage %s start", stage.value)
                await fn(ctx)
                ctx.record_timing(stage, time.perf_counter() - t0)
                ctx.log.info("stage %s done in %.1fs", stage.value, time.perf_counter() - t0)
            await ctx.complete()
    except (JobCancelled, asyncio.CancelledError):
        ctx.log.info("job cancelled at %s", ctx.stage.value)
        await ctx.fail("E_CANCELLED", ctx.stage, user_message("E_CANCELLED"), "Cancelled by the user.")
    except PipelineError as e:
        await ctx.fail(e.code, e.stage, e.user_message, e.detail)
    except Exception as e:  # noqa: BLE001
        ctx.log.exception("job crashed")
        await ctx.fail("E_INTERNAL", ctx.stage, user_message("E_INTERNAL"), repr(e))
    finally:
        CANCEL_EVENTS.pop(job_id, None)


def rename_record(jobs_dir: Path, job_id: str, title: str) -> bool:
    """Update record.meta.title and regenerate exports; False if there is no record yet."""
    d = job_dir(jobs_dir, job_id)
    data = read_json(d / STAGE_OUTPUT[Stage.VERIFYING])
    upload = read_json(d / STAGE_OUTPUT[Stage.VALIDATING])
    if upload is not None:
        upload["title"] = title
        write_json(d / STAGE_OUTPUT[Stage.VALIDATING], upload)
    if data is None:
        return False
    rec = MeetingRecord(**data)
    rec.meta.title = title
    write_json(d / STAGE_OUTPUT[Stage.VERIFYING], rec)
    if (d / "exports").exists():
        render_exports(rec, d / "exports")
    return True


EDITED_TRANSCRIPTS = {"raw_transcript": "raw_transcript.edited.json",
                      "refined_transcript": "refined_transcript.edited.json"}


def save_edits(jobs_dir: Path, job_id: str, edits: dict[str, Any]) -> bool:
    """Store the user's hand corrections and regenerate every export from them.

    With a record: the record is updated (the untouched original is kept once as record.original.json) and
    exports are rebuilt. Without one (writing the record failed): the corrected transcripts are kept next to
    the pipeline's own files, which stay untouched so Retry still resumes correctly. False: nothing to save.
    """
    d = job_dir(jobs_dir, job_id)
    edits = {k: v for k, v in edits.items() if v is not None}
    if not edits:
        return False
    data = read_json(d / STAGE_OUTPUT[Stage.VERIFYING])
    if data is None:
        for key, name in EDITED_TRANSCRIPTS.items():
            if key in edits:
                write_json(d / name, edits[key])
        return any(k in edits for k in EDITED_TRANSCRIPTS)
    if not (d / "record.original.json").exists():
        write_json(d / "record.original.json", data)
    rec = MeetingRecord(**{**data, **edits})
    write_json(d / STAGE_OUTPUT[Stage.VERIFYING], rec)
    render_exports(rec, d / "exports")
    return True


def dump_event(event: dict[str, Any]) -> str:
    """Serialize an event for SSE."""
    return json.dumps(event, ensure_ascii=False)
