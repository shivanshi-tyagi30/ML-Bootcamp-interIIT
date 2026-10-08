"""Raw LM1/LM2 output models; used as JSON schemas for constrained decoding."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.record import EditCategory, Pointer


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


class LM2Sentence(BaseModel):
    """A summary/minutes sentence as LM2 returns it; uncited sentences are dropped by the verifier.

    Lenient on purpose: a strict min_length would fail the whole answer (and re-generate it) for one
    uncited sentence.
    """

    text: str
    evidence_segment_ids: list[str] = []


class LM2Topic(BaseModel):
    """A minutes topic as LM2 returns it."""

    topic: str
    points: list[LM2Sentence] = []


class LM2Task(BaseModel):
    """A task as LM2 returns it: pointers, never typed owner/deadline."""

    task: str
    owner_evidence: Pointer | None = None
    deadline_evidence: Pointer | None = None
    evidence_quote: str = ""
    evidence_segment_ids: list[str] = []


class LM2Decision(BaseModel):
    """A decision as LM2 returns it."""

    decision: str
    agreement_evidence: str = ""
    evidence_segment_ids: list[str] = []


class LM2Proposal(BaseModel):
    """An open proposal as LM2 returns it."""

    proposal: str
    evidence_segment_ids: list[str] = []


class LM2Speaker(BaseModel):
    """LM2's reading of who a "Speaker N" label is; checked by code before it is used."""

    label: str
    name: str
    evidence_segment_id: str = ""


class LM2Record(BaseModel):
    """LM2 output without a scratchpad (LM2_SCRATCHPAD=false). Missing lists count as empty; the verifier
    removes anything that is not properly cited, so a small slip no longer fails the whole job."""

    speakers: list[LM2Speaker] = []
    summary: list[LM2Sentence] = []
    minutes: list[LM2Topic] = []
    decisions: list[LM2Decision] = []
    open_proposals: list[LM2Proposal] = []
    action_items: list[LM2Task] = []


class LM2Output(BaseModel):
    """LM2 output with the reasoning scratchpad first (generated before the record); discarded after parsing."""

    scratchpad: str = ""
    speakers: list[LM2Speaker] = []
    summary: list[LM2Sentence] = []
    minutes: list[LM2Topic] = []
    decisions: list[LM2Decision] = []
    open_proposals: list[LM2Proposal] = []
    action_items: list[LM2Task] = []


class SummaryPick(BaseModel):
    """Chunked mode: indexes of existing summary sentences to keep (max 5)."""

    keep: list[int] = Field(max_length=5)
