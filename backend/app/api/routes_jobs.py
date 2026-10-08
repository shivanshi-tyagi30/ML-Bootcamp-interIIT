"""Job routes: create, list, get, rename, delete, SSE progress (spec Section 9)."""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api.deps import AppState, error_response, get_job_or_none, get_state
from app.core.errors import PipelineError, user_message
from app.core.events import TERMINAL
from app.core.stages import STAGE_MESSAGES, STAGE_OUTPUT, Stage
from app.core.storage import job_dir, read_json, write_json
from app.models.api import (
    CreateJobResponse, EditsRequest, JobDetail, JobList, JobSummary, PartialResults, RenameRequest,
)
from app.models.record import MeetingRecord, Refinement, Segment
from app.pipeline.runner import (
    EDITED_TRANSCRIPTS, dump_event, has_api_key, remember_api_key, rename_record, request_cancel, save_edits,
)
from app.pipeline.validate import extension_of, stream_upload, validate_file

router = APIRouter(prefix="/api", tags=["jobs"])


def default_title(filename: str, title: str | None) -> str:
    """Trimmed title (max 100 chars), or '<filename> – <D Mon YYYY>'."""
    title = (title or "").strip()[:100]
    if title:
        return title
    d = datetime.now()
    return f"{filename} – {d.day} {d:%b %Y}"[:100]


def made_with_cloud(jobs_dir: Path, job_id: str) -> bool:
    """Whether a finished job's record was written by the cloud model (so a re-upload reuses it only when
    it would be made the same way: a key -> cloud, no key -> local)."""
    rec = read_json(job_dir(jobs_dir, job_id) / STAGE_OUTPUT[Stage.VERIFYING]) or {}
    return "(cloud)" in str((rec.get("meta") or {}).get("models", {}).get("lm2", ""))


@router.post("/jobs", status_code=202, response_model=CreateJobResponse)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    glossary: str | None = Form(None),
    api_key: str | None = Form(None),
    force: bool = Query(False),
    state: AppState = Depends(get_state),
) -> Any:
    """Upload a recording. Validation errors return immediately; processing runs in the background."""
    s = state.settings
    effective_key = (api_key or request.headers.get("x-gemini-key") or request.headers.get("x-api-key")
                     or getattr(s, "GEMINI_API_KEY", "") or "").strip()
    if not effective_key and not s.LOCAL_LLM_ENABLED:
        return error_response("E_NEEDS_KEY")  # before the upload is stored: nothing to clean up
    if state.active_uploads >= s.MAX_CONCURRENT_UPLOADS:
        return error_response("E_BUSY")
    state.active_uploads += 1
    job_id = uuid4().hex
    d = job_dir(s.jobs_dir, job_id)
    filename = Path(file.filename or "recording").name[:200]
    try:
        stored = d / f"original.{extension_of(filename) or 'bin'}"
        try:
            await stream_upload(file, stored, s.MAX_FILE_MB * 1024 * 1024)
            info = await asyncio.to_thread(validate_file, stored, filename, s)
        except PipelineError as e:
            shutil.rmtree(d, ignore_errors=True)
            return JSONResponse(status_code=e.http_status, content={"code": e.code, "message": e.user_message})

        if not force:
            cached = await state.db.find_completed_by_sha(info["sha256"])
            if cached and made_with_cloud(s.jobs_dir, cached["id"]) == bool(effective_key):
                shutil.rmtree(d, ignore_errors=True)
                return JSONResponse(status_code=200, content={"job_id": cached["id"], "status": "completed",
                                                              "cached": True})

        job_title = default_title(filename, title)
        terms = [t.strip() for t in (glossary or "").split(",") if t.strip()][:100]
        write_json(d / STAGE_OUTPUT[Stage.VALIDATING], {
            **info, "job_id": job_id, "title": job_title, "source_file": filename, "glossary": terms,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if effective_key:
            remember_api_key(job_id, effective_key)  # memory only: a key is never written to disk
        await state.db.insert({
            "id": job_id, "title": job_title, "source_file": filename, "file_sha256": info["sha256"],
            "status": "queued", "stage": Stage.QUEUED.value,
        })
        state.schedule(job_id)
        return CreateJobResponse(job_id=job_id, status="queued", cached=False)
    finally:
        state.active_uploads -= 1


@router.get("/jobs", response_model=JobList)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), state: AppState = Depends(get_state),
) -> JobList:
    """Past and current meetings, newest first."""
    rows, total = await state.db.list(limit, offset)
    return JobList(items=[JobSummary.from_row(r) for r in rows], total=total)


def _segments(path: Path) -> list[Segment] | None:
    """Segments from a transcript file, if present."""
    data = read_json(path)
    return [Segment(**s) for s in data] if isinstance(data, list) else None


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, state: AppState = Depends(get_state)) -> Any:
    """Job summary, the record when ready, and transcripts available so far."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    d = job_dir(state.settings.jobs_dir, job_id)
    rec = read_json(d / STAGE_OUTPUT[Stage.VERIFYING])
    refinement = read_json(d / STAGE_OUTPUT[Stage.GUARDING])
    return JobDetail(
        job=JobSummary.from_row(row),
        record=MeetingRecord(**rec) if rec else None,
        partial=PartialResults(
            raw_transcript=_segments(d / EDITED_TRANSCRIPTS["raw_transcript"])
            or _segments(d / STAGE_OUTPUT[Stage.RAW_SAVED]),
            refined_transcript=_segments(d / EDITED_TRANSCRIPTS["refined_transcript"])
            or _segments(d / "refined_transcript.json"),
            refinement=Refinement(**refinement) if refinement else None,
        ),
        timings=read_json(d / "timings.json") or None,
    )


def snapshot_event(row: dict[str, Any], d: Path) -> dict[str, Any]:
    """An event built from the database, for clients that connect after the job's last event."""
    status = row["status"]
    stage = row["stage"]
    ev: dict[str, Any] = {
        "job_id": row["id"], "stage": stage, "status": status,
        "progress": 1.0 if status == "completed" else 0.0,
        "message": STAGE_MESSAGES.get(Stage(stage), "") if stage in Stage._value2member_map_ else "",
        "raw_ready": (d / STAGE_OUTPUT[Stage.RAW_SAVED]).exists(),
        "refined_ready": (d / STAGE_OUTPUT[Stage.GUARDING]).exists(),
        "record_ready": (d / STAGE_OUTPUT[Stage.VERIFYING]).exists(),
        "warnings": [],
    }
    if status == "failed":
        ev.update(error_code=row.get("error_code"), error_message=row.get("error_message"),
                  error_detail=row.get("error_detail"))
    return ev


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, state: AppState = Depends(get_state)) -> Any:
    """Server-Sent Events: `progress` events until the job completes or fails."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    d = job_dir(state.settings.jobs_dir, job_id)

    async def stream():
        q = state.bus.subscribe(job_id)
        try:
            latest = state.bus.latest(job_id) or snapshot_event(row, d)
            yield {"event": "progress", "data": dump_event(latest)}
            if latest["status"] in TERMINAL:
                return
            while not await request.is_disconnected():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=10)
                except asyncio.TimeoutError:
                    continue
                yield {"event": "progress", "data": dump_event(ev)}
                if ev["status"] in TERMINAL:
                    return
        finally:
            state.bus.unsubscribe(job_id, q)

    return EventSourceResponse(stream(), ping=15)


@router.patch("/jobs/{job_id}", response_model=JobSummary)
async def rename_job(job_id: str, body: RenameRequest, state: AppState = Depends(get_state)) -> Any:
    """Rename a meeting; updates the record and regenerates exports."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    title = body.title.strip()[:100] or row["title"]
    await state.db.update(job_id, title=title)
    await asyncio.to_thread(rename_record, state.settings.jobs_dir, job_id, title)
    return JobSummary.from_row(await state.db.get(job_id))  # type: ignore[arg-type]


@router.put("/jobs/{job_id}/edits", status_code=204, response_model=None, response_class=Response)
async def put_edits(job_id: str, body: EditsRequest, state: AppState = Depends(get_state)) -> Any:
    """Save hand corrections (words, lines, speaker names) so the page and every download show them."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    if row["status"] in ("queued", "running"):
        return error_response("E_JOB_RUNNING")
    await asyncio.to_thread(save_edits, state.settings.jobs_dir, job_id, body.model_dump(mode="json", exclude_none=True))
    return Response(status_code=204)


@router.delete("/jobs/{job_id}", status_code=204, response_model=None, response_class=Response)
async def delete_job(job_id: str, state: AppState = Depends(get_state)) -> Any:
    """Delete a meeting and its files; refused while it is processing."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    if row["status"] in ("queued", "running"):
        return error_response("E_JOB_RUNNING")
    shutil.rmtree(job_dir(state.settings.jobs_dir, job_id), ignore_errors=True)
    await state.db.delete(job_id)
    state.bus.forget(job_id)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/cancel", response_model=JobSummary)
async def cancel_job(job_id: str, state: AppState = Depends(get_state)) -> Any:
    """Stop a queued or running job. Finished stages stay saved, so Retry continues from there."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    if row["status"] in ("queued", "running"):
        request_cancel(job_id)
        task = state.task_for(job_id)
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        row = await state.db.get(job_id) or row
        if row["status"] in ("queued", "running"):  # no live task (e.g. stale row): fail it directly
            await state.db.update(job_id, status="failed", stage="failed", error_code="E_CANCELLED",
                                  error_message=user_message("E_CANCELLED"), error_detail="Cancelled by the user.")
            row = await state.db.get(job_id) or row
            ev = snapshot_event(row, job_dir(state.settings.jobs_dir, job_id))
            state.bus.publish(job_id, ev)
    return JobSummary.from_row(row)


@router.post("/jobs/{job_id}/retry", status_code=202, response_model=JobSummary)
async def retry_job(job_id: str, request: Request, api_key: str | None = Form(None),
                    state: AppState = Depends(get_state)) -> Any:
    """Run a failed or cancelled job again, resuming after the last saved stage.

    The cloud API key is sent again with the retry: keys live in memory only, so a restarted server
    would otherwise retry on the local model.
    """
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    if row["status"] in ("queued", "running") or state.task_for(job_id) is not None:
        return error_response("E_JOB_RUNNING")
    if row["status"] == "completed":
        return JobSummary.from_row(row)
    key = (api_key or request.headers.get("x-gemini-key") or "").strip()
    if key:
        remember_api_key(job_id, key)
    elif not state.settings.LOCAL_LLM_ENABLED and not has_api_key(job_id) and not state.settings.GEMINI_API_KEY:
        return error_response("E_NEEDS_KEY")
    await state.db.update(job_id, status="queued", stage="queued", error_code=None, error_message=None,
                          error_detail=None)
    state.schedule(job_id)
    return JobSummary.from_row(await state.db.get(job_id))  # type: ignore[arg-type]
