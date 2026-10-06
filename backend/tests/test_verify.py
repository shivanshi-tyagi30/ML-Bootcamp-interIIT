"""Verifier: owners, deadlines, decisions and no-new-facts (spec Section 10.12)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.models.llm_io import LM2Output
from app.models.record import UNSPECIFIED, Meta, Pointer, Refinement
from app.pipeline.verify import resolve_pointer, verify
from tests.conftest import DEFAULT_LM2, sample_segments, seg

S = Settings()


def segmap(*segs):
    return {s.id: s for s in segs}


def test_pointer_to_absent_words():
    m = segmap(seg("S001", 0, "Can you fix the build?"))
    assert resolve_pointer(Pointer(segment_id="S001", exact_words="Arjun"), ["S001"], m, False) == (
        UNSPECIFIED, "pointer_invalid")


def test_null_pointer():
    assert resolve_pointer(None, ["S001"], {}, False) == (UNSPECIFIED, None)


def test_self_intro_same_speaker_copies_name():
    m = segmap(seg("S001", 0, "Hi all, this is Priya from data.", "Speaker 1"),
               seg("S002", 3, "I'll update the dashboards.", "Speaker 1"))
    assert resolve_pointer(Pointer(segment_id="S001", exact_words="priya"), ["S002"], m, True) == ("Priya", None)


def test_self_intro_without_diarization():
    m = segmap(seg("S001", 0, "Hi all, this is Priya from data."), seg("S002", 3, "I'll update the dashboards."))
    assert resolve_pointer(Pointer(segment_id="S001", exact_words="Priya"), ["S002"], m, False) == (
        UNSPECIFIED, "self_assignment_unverified")


def test_disputed_owner_word():
    m = segmap(seg("S001", 0, "Meera will review the keys.", disputed={"Meera": "Mira"}))
    assert resolve_pointer(Pointer(segment_id="S001", exact_words="Meera"), ["S001"], m, False) == (
        UNSPECIFIED, "audio_unclear")


def test_self_correction_deadline():
    m = segmap(seg("S007", 0, "I'll update the dashboards by Friday... no, Monday."))
    assert resolve_pointer(Pointer(segment_id="S007", exact_words="Monday"), ["S007"], m, False, "deadline") == (
        "Monday", None)


def _run(lm2=DEFAULT_LM2, diarization=False):
    raw = sample_segments()
    meta = Meta(job_id="x" * 32, title="t", source_file="a.wav", duration_s=24, language="en",
                language_probability=1, models={}, generated_at=datetime.now(timezone.utc))
    return verify(LM2Output(**lm2), raw, raw, Refinement(), meta, S, diarization)


def test_full_verification():
    rec = _run()
    # Decision without an agreement cue moves to proposals.
    assert [d.decision for d in rec.decisions] == ["Ship v2 without the reranker"]
    assert rec.open_proposals[0].proposal == "Move the vector store to Postgres"
    assert rec.open_proposals[0].demoted_from_decision
    # "15 to 50" matches "fifteen to fifty"; invented name and number are removed.
    assert [s.text for s in rec.summary] == [
        "The team agreed to ship v2 without the reranker.", "Latency went from 15 to 50 milliseconds."]
    # "CI" is not in the cited raw text ("sea eye") and "30" is invented, so the topic empties out.
    assert rec.minutes == []
    assert rec.fidelity.sentences_removed_by_verifier == 3
    # Owner via self-introduction needs diarization; deadline is the corrected value.
    t1, t2 = rec.action_items
    assert (t1.id, t1.owner, t1.deadline) == ("T1", UNSPECIFIED, "Monday")
    assert "owner_downgraded" in t1.verifier_flags and "self_assignment_unverified" in t1.verifier_flags
    assert (t2.owner, t2.deadline, t2.verifier_flags) == (UNSPECIFIED, UNSPECIFIED, [])


def test_owner_with_diarization():
    rec = _run(diarization=True)
    assert rec.action_items[0].owner == "Priya"


def test_task_with_unmatched_quote_is_dropped():
    lm2 = {**DEFAULT_LM2, "action_items": [{
        "task": "Write docs", "owner_evidence": None, "deadline_evidence": None,
        "evidence_quote": "We must write the documentation tonight.", "evidence_segment_ids": ["S003"]}]}
    rec = _run(lm2)
    assert rec.action_items == []
    assert rec.fidelity.items_downgraded_by_verifier >= 1


def test_unknown_ids_dropped():
    lm2 = {**DEFAULT_LM2, "summary": [{"text": "Something happened.", "evidence_segment_ids": ["S999"]}]}
    assert _run(lm2).summary == []
