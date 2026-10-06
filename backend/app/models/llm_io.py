"""Raw LM1/LM2 output models; used as JSON schemas for constrained decoding."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.record import CitedSentence, EditCategory, MinutesTopic, Pointer


class VocabularyTerm(BaseModel):
    """A term LM1 found in the meeting, with how it was misheard."""

    term: str
    heard_as: list[str] = []
    evidence_segment_ids: list[str] = []


class Vocabulary(BaseModel):
    """LM1 pass A output."""

    domain: str
    terms: list[VocabularyTerm] = Field(default_factory=list, max_length=60)


class LM1Edit(BaseModel):
    """One edit proposed by LM1 pass B."""

    segment_id: str
    original: str
    replacement: str
    category: EditCategory
    rationale: str = ""
    confidence: float = Field(ge=0, le=1)


class LM1Output(BaseModel):
    """LM1 pass B output for one window."""

    domain_guess: str = ""
    edits: list[LM1Edit] = []


class LM2Task(BaseModel):
    """A task as LM2 returns it: pointers, never typed owner/deadline."""

    task: str
    owner_evidence: Pointer | None = None
    deadline_evidence: Pointer | None = None
    evidence_quote: str
    evidence_segment_ids: list[str]


class LM2Decision(BaseModel):
    """A decision as LM2 returns it."""

    decision: str
    agreement_evidence: str
    evidence_segment_ids: list[str]


class LM2Proposal(BaseModel):
    """An open proposal as LM2 returns it."""

    proposal: str
    evidence_segment_ids: list[str]


class LM2Output(BaseModel):
    """LM2 output. `scratchpad` is discarded after parsing."""

    scratchpad: str = ""
    summary: list[CitedSentence]
    minutes: list[MinutesTopic]
    decisions: list[LM2Decision]
    open_proposals: list[LM2Proposal]
    action_items: list[LM2Task]


class SummaryPick(BaseModel):
    """Chunked mode: indexes of existing summary sentences to keep (max 5)."""

    keep: list[int] = Field(max_length=5)
