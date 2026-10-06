"""Fidelity scorecard (spec Section 10.13)."""

from __future__ import annotations

from app.core.text import negation_count, numbers
from app.models.record import Fidelity, MeetingRecord, Refinement, Segment


def preserved(raw: list[Segment], refined: list[Segment], kind: str) -> str:
    """'k/n' for numbers or negations: n in raw, k in segments where refinement kept them unchanged."""
    ref = {s.id: s for s in refined}
    n = k = 0
    for r in raw:
        f = ref.get(r.id, r)
        if kind == "numbers":
            cnt = sum(numbers(r.text).values())
            same = numbers(r.text) == numbers(f.text)
        else:
            cnt = negation_count(r.text)
            same = cnt == negation_count(f.text)
        n += cnt
        k += cnt if same else 0
    return f"{k}/{n}"


def cited_ids(rec: MeetingRecord) -> set[str]:
    """Every segment id cited anywhere in the record."""
    ids: set[str] = set()
    for s in rec.summary:
        ids.update(s.evidence_segment_ids)
    for t in rec.minutes:
        for p in t.points:
            ids.update(p.evidence_segment_ids)
    for group in (rec.decisions, rec.open_proposals, rec.action_items):
        for x in group:
            ids.update(x.evidence_segment_ids)
    return ids


def compute_fidelity(
    rec: MeetingRecord, refinement: Refinement, dropped_items: int, demoted: int, sentences_removed: int,
) -> Fidelity:
    """Fidelity counts for an assembled record."""
    downgraded_tasks = sum(
        1 for a in rec.action_items if {"owner_downgraded", "deadline_downgraded"} & set(a.verifier_flags)
    )
    total = len(rec.raw_transcript)
    coverage = round(100 * len(cited_ids(rec) & {s.id for s in rec.raw_transcript}) / total, 1) if total else 0.0
    return Fidelity(
        numbers_preserved=preserved(rec.raw_transcript, rec.refined_transcript, "numbers"),
        negations_preserved=preserved(rec.raw_transcript, rec.refined_transcript, "negations"),
        edits_accepted=len(refinement.accepted),
        edits_rejected=len(refinement.rejected),
        disputed_words=sum(1 for s in rec.raw_transcript for w in s.words if w.disputed),
        items_downgraded_by_verifier=downgraded_tasks + demoted + dropped_items,
        sentences_removed_by_verifier=sentences_removed,
        transcript_coverage_pct=coverage,
    )
