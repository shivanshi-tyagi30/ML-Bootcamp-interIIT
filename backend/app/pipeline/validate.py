"""VALIDATING: size, extension and real file type checks (spec Section 10.1)."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.core.errors import PipelineError
from app.core.hashing import sha256_file
from app.core.stages import Stage

if TYPE_CHECKING:
    from fastapi import UploadFile

    from app.pipeline.runner import JobContext

log = logging.getLogger(__name__)
CHUNK = 1 << 20


def extension_of(filename: str) -> str:
    """Lowercase extension without the dot ('' if none)."""
    return Path(filename or "").suffix.lower().lstrip(".")


async def stream_upload(upload: "UploadFile", dest: Path, max_bytes: int) -> int:
    """Copy an upload to disk in 1 MB chunks; aborts with E_TOO_LARGE once over `max_bytes`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise PipelineError("E_TOO_LARGE", Stage.VALIDATING, f"over {max_bytes} bytes")
            f.write(chunk)
    return size


def _ffprobe_has_audio(path: Path) -> bool:
    """True if ffprobe finds an audio stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        streams = json.loads(out.stdout or "{}").get("streams", [])
        return out.returncode == 0 and any(s.get("codec_type") == "audio" for s in streams)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return False


def sniff_mime(path: Path) -> str | None:
    """MIME type from file content via libmagic, or None if python-magic is unavailable."""
    try:
        import magic
    except ImportError:
        log.warning("python-magic not installed; skipping content type check")
        return None
    return magic.from_file(str(path), mime=True)


def validate_file(path: Path, original_name: str, settings: Settings) -> dict[str, Any]:
    """Check a saved upload; returns its metadata or raises a validation PipelineError."""
    size = path.stat().st_size
    if size == 0:
        raise PipelineError("E_EMPTY_FILE", Stage.VALIDATING)
    if size > settings.MAX_FILE_MB * 1024 * 1024:
        raise PipelineError("E_TOO_LARGE", Stage.VALIDATING)
    ext = extension_of(original_name)
    if ext not in settings.allowed_ext:
        raise PipelineError("E_UNSUPPORTED_FORMAT", Stage.VALIDATING, f"extension {ext!r}")
    mime = sniff_mime(path)
    if mime is not None:
        ok = mime.startswith(("audio/", "video/")) or (mime == "application/octet-stream" and _ffprobe_has_audio(path))
        if not ok:
            raise PipelineError("E_UNSUPPORTED_FORMAT", Stage.VALIDATING, f"mime {mime}")
    return {"ext": ext, "size_bytes": size, "mime": mime, "sha256": sha256_file(path)}


async def validate(ctx: "JobContext") -> None:
    """Stage function: re-validate the stored original (normally done at upload time)."""
    info = dict(ctx.upload)
    info.update(validate_file(ctx.original_path, info.get("source_file", ctx.original_path.name), ctx.settings))
    ctx.write(ctx.output_name(Stage.VALIDATING), info)
