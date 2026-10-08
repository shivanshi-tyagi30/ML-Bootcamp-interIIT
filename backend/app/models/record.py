"""MeetingRecord and sub-models: the single source of truth (spec Section 6)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

UNSPECIFIED = "Unspecified"

EditCategory = Literal["acronym", "technical_term", "proper_noun", "product", "homophone", "symbol"]
RejectReason = Literal[
    "not_found", "number_changed", "negation_changed", "modal_changed", "over_rewrite",
    "not_sound_alike", "low_confidence", "name_changed", "touches_disputed_frozen",
]
VerifierFlag = Literal[
    "owner_downgraded", "deadline_downgraded", "pointer_invalid", "audio_unclear", "self_assignment_unverified",
    "owner_from_speaker",
]


class Word(BaseModel):
    """One recognised word with timing and confidence."""

    w: str
    start: float
    end: float
    conf: float
    disputed: bool = False
    alt: str | None = None


class Segment(BaseModel):
    """A transcript line with a stable id (S001...)."""

    id: str
    start: float
    end: float
    speaker: str | None = None
    text: str
    words: list[Word] = []


class Edit(BaseModel):
    """A terminology fix proposed by LM1."""

    segment_id: str
    original: str
    replacement: str
    category: EditCategory
    rationale: str = ""
    confidence: float


class RejectedEdit(Edit):
    """An LM1 edit blocked by the guard."""

    reject_reason: RejectReason


class Refinement(BaseModel):
    """LM1 vocabulary plus accepted and rejected edits."""

    vocabulary: dict[str, Any] = {}
    accepted: list[Edit] = []
    rejected: list[RejectedEdit] = []


class CitedSentence(BaseModel):
    """A sentence with the transcript segments it is based on."""

    text: str
    evidence_segment_ids: list[str] = Field(min_length=1)


class MinutesTopic(BaseModel):
    """Minutes grouped under a topic."""

    topic: str
    points: list[CitedSentence]


class Decision(BaseModel):
    """An explicitly agreed decision."""

    id: str
    decision: str
    agreement_evidence: str
    evidence_segment_ids: list[str] = Field(min_length=1)


class Proposal(BaseModel):
    """An idea raised without agreement."""

    proposal: str
    evidence_segment_ids: list[str] = Field(min_length=1)
    demoted_from_decision: bool = False


class Pointer(BaseModel):
    """Where in the transcript a value was said."""

    segment_id: str
    exact_words: str


class ActionItem(BaseModel):
    """A task. Owner and deadline are always set by the verifier."""

    id: str
    task: str
    owner: str = UNSPECIFIED
    owner_evidence: Pointer | None = None
    deadline: str = UNSPECIFIED
    deadline_evidence: Pointer | None = None
    evidence_quote: str
    evidence_segment_ids: list[str] = Field(min_length=1)
    verifier_flags: list[VerifierFlag] = []


class Fidelity(BaseModel):
    """Counts that show how faithful the record is to the recording."""

    numbers_preserved: str
    negations_preserved: str
    edits_accepted: int
    edits_rejected: int
    disputed_words: int
    items_downgraded_by_verifier: int
    sentences_removed_by_verifier: int
    transcript_coverage_pct: float


class Meta(BaseModel):
    """Job and model information."""

    job_id: str
    title: str
    source_file: str
    duration_s: float
    language: str
    language_probability: float
    models: dict[str, str]
    warnings: list[str] = []
    generated_at: datetime


class MeetingRecord(BaseModel):
    """Everything the app shows and exports for one recording."""

    meta: Meta
    raw_transcript: list[Segment]
    refined_transcript: list[Segment]
    refinement: Refinement
    summary: list[CitedSentence]
    minutes: list[MinutesTopic]
    decisions: list[Decision]
    open_proposals: list[Proposal]
    action_items: list[ActionItem]
    fidelity: Fidelity
