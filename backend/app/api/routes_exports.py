"""Downloads and audio streaming (spec Section 9)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.api.deps import AppState, error_response, get_job_or_none, get_state
from app.core.stages import STAGE_OUTPUT, Stage
from app.core.storage import job_dir, read_json, slugify
from app.exports import EXPORT_FILES, render_exports
from app.exports.txt import transcript_txt
from app.models.record import MeetingRecord, Segment

router = APIRouter(prefix="/api", tags=["exports"])

AUDIO_TYPES = {
    "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4", "ogg": "audio/ogg", "flac": "audio/flac",
    "webm": "audio/webm", "mp4": "video/mp4", "aac": "audio/aac",
}
EXPORT_TYPES = {
    "json": ("application/json", "record.json"),
    "md": ("text/markdown; charset=utf-8", "record.md"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "record.docx"),
    "txt_raw": ("text/plain; charset=utf-8", "transcript_raw.txt"),
    "txt_refined": ("text/plain; charset=utf-8", "transcript_refined.txt"),
}


def _iter_file(path: Path, start: int, length: int, chunk: int = 1 << 16) -> Iterator[bytes]:
    """Yield `length` bytes of a file starting at `start`."""
    with open(path, "rb") as f:
        f.seek(start)
        while length > 0:
            data = f.read(min(chunk, length))
            if not data:
                break
            length -= len(data)
            yield data


def ranged_file(path: Path, request: Request, media_type: str) -> Response:
    """Serve a file with HTTP Range support so the browser can seek."""
    size = path.stat().st_size
    header = request.headers.get("range")
    if not header:
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:  # suffix range: last N bytes
        start, end = max(0, size - int(m.group(2))), size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    length = end - start + 1
    return StreamingResponse(
        _iter_file(path, start, length), status_code=206, media_type=media_type,
        headers={"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length),
                 "Accept-Ranges": "bytes"},
    )


@router.get("/jobs/{job_id}/audio")
async def job_audio(job_id: str, request: Request, state: AppState = Depends(get_state)) -> Any:
    """The original recording, seekable."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    d = job_dir(state.settings.jobs_dir, job_id)
    ext = (read_json(d / STAGE_OUTPUT[Stage.VALIDATING]) or {}).get("ext", "")
    path = d / f"original.{ext}"
    if not ext or not path.exists():
        return error_response("E_NOT_FOUND")
    return ranged_file(path, request, AUDIO_TYPES.get(ext, "application/octet-stream"))


@router.get("/jobs/{job_id}/export")
async def job_export(
    job_id: str, fmt: str = Query(..., pattern="^(json|md|docx|txt_raw|txt_refined)$"),
    state: AppState = Depends(get_state),
) -> Any:
    """Download one export. Transcripts are available even if the record failed."""
    row = await get_job_or_none(state, job_id)
    if row is None:
        return error_response("E_NOT_FOUND")
    d = job_dir(state.settings.jobs_dir, job_id)
    media, suffix = EXPORT_TYPES[fmt]
    headers = {"Content-Disposition": f'attachment; filename="{slugify(row["title"])}_{suffix}"'}
    data = read_json(d / STAGE_OUTPUT[Stage.VERIFYING])

    if data is None:  # no record: only the transcripts can be downloaded
        source = {"txt_raw": STAGE_OUTPUT[Stage.RAW_SAVED], "txt_refined": "refined_transcript.json"}.get(fmt)
        segs = read_json(d / source) if source else None
        if not isinstance(segs, list):
            return error_response("E_NOT_FOUND")
        return Response(transcript_txt([Segment(**s) for s in segs]), media_type=media, headers=headers)

    rec = MeetingRecord(**data)
    if fmt == "json":
        return Response(rec.model_dump_json(indent=2), media_type=media, headers=headers)
    out = d / "exports" / EXPORT_FILES[fmt]
    await asyncio.to_thread(render_exports, rec, d / "exports")
    return FileResponse(out, media_type=media, headers=headers)
