"""API request and response models (spec Section 9)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.record import MeetingRecord, Refinement, Segment


class JobSummary(BaseModel):
    """A row of the jobs list."""

    id: str
    title: str
    source_file: str
    status: str
    stage: str
    error_code: str | None = None
    error_message: str | None = None
    error_detail: str | None = None
    duration_s: float | None = None
    n_decisions: int = 0
    n_tasks: int = 0
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "JobSummary":
        """Build from a database row."""
        return cls(**{k: row.get(k) for k in cls.model_fields})


class CreateJobResponse(BaseModel):
    """Response of POST /api/jobs."""

    job_id: str
    status: str
    cached: bool = False


class JobList(BaseModel):
    """Response of GET /api/jobs."""

    items: list[JobSummary]
    total: int


class PartialResults(BaseModel):
    """Transcripts available before (or without) a full record."""

    raw_transcript: list[Segment] | None = None
    refined_transcript: list[Segment] | None = None
    refinement: Refinement | None = None


class JobDetail(BaseModel):
    """Response of GET /api/jobs/{id}."""

    job: JobSummary
    record: MeetingRecord | None = None
    partial: PartialResults


class RenameRequest(BaseModel):
    """Body of PATCH /api/jobs/{id}."""

    title: str = Field(min_length=1, max_length=100)


class ErrorBody(BaseModel):
    """Error response body."""

    code: str
    message: str
