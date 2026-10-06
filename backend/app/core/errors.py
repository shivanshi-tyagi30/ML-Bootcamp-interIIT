"""Typed pipeline errors and the user-facing error table (spec Section 7)."""

from __future__ import annotations

from typing import TypedDict

from app.config import get_settings
from app.core.stages import Stage


class ErrorSpec(TypedDict):
    """One row of the error table."""

    http: int
    message: str


ERROR_TABLE: dict[str, ErrorSpec] = {
    "E_UNSUPPORTED_FORMAT": {
        "http": 415,
        "message": "This file type isn't supported. Please upload an audio file (MP3, WAV, M4A, OGG, FLAC, WEBM, AAC).",
    },
    "E_EMPTY_FILE": {"http": 400, "message": "The file is empty. Please choose a recording that contains audio."},
    "E_TOO_LARGE": {"http": 413, "message": "The file is too large. The maximum size is {MAX_FILE_MB} MB."},
    "E_TOO_LONG": {"http": 413, "message": "The recording is too long. The maximum length is {MAX_DURATION_MIN} minutes."},
    "E_FFMPEG_MISSING": {
        "http": 500,
        "message": "ffmpeg is not installed on the server, so the audio can't be converted. Install ffmpeg and restart the backend.",
    },
    "E_UNREADABLE": {"http": 422, "message": "The file could not be read. It may be corrupt or have no audio track."},
    "E_NO_SPEECH": {"http": 422, "message": "No speech was detected in this recording."},
    "E_STT_FAILED": {"http": 500, "message": "Transcription failed. Please try again."},
    "E_LM1_FAILED": {"http": 500, "message": "Transcript refinement failed. The raw transcript is still available."},
    "E_LM2_FAILED": {"http": 500, "message": "Writing the meeting record failed. Both transcripts are still available."},
    "E_RENDER_FAILED": {"http": 500, "message": "The record was created but the downloads could not be generated."},
    "E_BUSY": {"http": 429, "message": "The server is busy with other uploads. Please try again in a moment."},
    "E_NOT_FOUND": {"http": 404, "message": "This meeting could not be found."},
    "E_JOB_RUNNING": {"http": 409, "message": "This meeting is still being processed. Try again when it has finished."},
    "E_CANCELLED": {"http": 200, "message": "Processing was cancelled."},
    "E_INTERNAL": {"http": 500, "message": "Something unexpected went wrong. Please try again."},
    "W_NON_ENGLISH": {"http": 200, "message": "This recording may not be in English; results may be less accurate."},
}


def user_message(code: str) -> str:
    """The exact user-facing text for an error code, with limits filled in."""
    s = get_settings()
    spec = ERROR_TABLE.get(code, ERROR_TABLE["E_INTERNAL"])
    return spec["message"].format(MAX_FILE_MB=s.MAX_FILE_MB, MAX_DURATION_MIN=s.MAX_DURATION_MIN)


class PipelineError(Exception):
    """A failure with a known code; its message is safe to show to users."""

    def __init__(self, code: str, stage: Stage, detail: str = "") -> None:
        """Create an error for `code` raised while running `stage`."""
        super().__init__(f"{code} at {stage.value}: {detail}")
        if code not in ERROR_TABLE:
            code = "E_INTERNAL"
        self.code = code
        self.stage = stage
        self.detail = detail
        self.user_message = user_message(code)
        self.http_status = ERROR_TABLE[code]["http"]


class JobCancelled(Exception):
    """The user cancelled the job; raised from worker threads at safe points."""
