"""Accuracy aids: the keyword checklist, verifier repairs, owners from named speakers, confident respellings."""

from __future__ import annotations

from app.models.llm_io import LM2Output
from app.models.record import Meta, Refinement
from app.pipeline.candidates import find_candidates
from app.pipeline.lm2_document import user_message
from app.pipeline.verify import verify
from tests.conftest import seg

MEETING = [
    ("S001", "Shivanshi", "Should we also capture live audio from Google Meet?"),
    ("S002", "Prachi", "Let's postpone that. For the demo we'll accept uploaded recordings only."),
    ("S003", "Shivanshi", "Agreed. Prachi, can you write the README by Thursday?"),
    ("S004", "Prachi", "Sure, I'll do it."),
    ("S005", "Shivanshi", "I'll fix the upload bug tonight."),
    ("S006", "Prachi", "The weather was nice today."),
]


def segs():
    return [seg(i, n * 3.0, t, spk) for n, (i, spk, t) in enumerate(MEETING)]


def meta():
    return Meta(job_id="x", title="t", source_file="f", duration_s=1, language="en", language_probability=1,
                models={}, generated_at="2026-01-01T00:00:00Z")


def test_checklist_flags_tasks_decisions_and_proposals_only():
    found = {c["ids"][0]: c["kinds"] for c in find_candidates(segs())}
    assert "proposal" in found["S001"]
    assert "decision" in found["S002"]
    assert "task" in found["S003"] and "task" in found["S005"]
    assert "S006" not in found
    msg = user_message([f"[{i}] ({spk}) {t}" for i, spk, t in MEETING], candidates=find_candidates(segs()))
    assert "CHECKLIST" in msg and "[S005] looks like: task" in msg
    assert msg.rstrip().endswith("Every item cites segment ids.")


def test_paraphrased_task_quote_is_repaired_from_the_transcript(settings):
    lm2 = LM2Output(action_items=[{"task": "Fix the upload bug", "evidence_quote": "she will fix the bug",
                                   "evidence_segment_ids": ["S005"]}])
    rec = verify(lm2, segs(), segs(), Refinement(), meta(), settings, diarization_on=True)
    assert [t.evidence_quote for t in rec.action_items] == ["I'll fix the upload bug tonight."]


def test_task_without_a_task_line_is_still_dropped(settings):
    lm2 = LM2Output(action_items=[{"task": "Enjoy the weather", "evidence_quote": "nice weather",
                                   "evidence_segment_ids": ["S006"]}])
    assert verify(lm2, segs(), segs(), Refinement(), meta(), settings, diarization_on=True).action_items == []


def test_owner_from_the_named_speaker_who_took_it_on(settings):
    lm2 = LM2Output(action_items=[
        {"task": "Fix the upload bug", "evidence_quote": "I'll fix the upload bug tonight.",
         "evidence_segment_ids": ["S005"]},
        {"task": "Write the README", "evidence_quote": "can you write the README by Thursday?",
         "evidence_segment_ids": ["S003"],
         "deadline_evidence": {"segment_id": "S003", "exact_words": "by Thursday"}},
    ])
    rec = verify(lm2, segs(), segs(), Refinement(), meta(), settings, diarization_on=True)
    got = {t.task: (t.owner, t.deadline, t.verifier_flags) for t in rec.action_items}
    assert got["Fix the upload bug"] == ("Shivanshi", "Unspecified", ["owner_from_speaker"])
    assert got["Write the README"][:2] == ("Prachi", "by Thursday")
    readme = next(t for t in rec.action_items if t.task == "Write the README")
    assert "S004" in readme.evidence_segment_ids  # Prachi's "Sure, I'll do it." is the evidence


def test_no_owner_from_unnamed_speakers_or_without_diarization(settings):
    unnamed = [seg("S001", 0, "I'll fix the upload bug tonight.", "Speaker 1")]
    lm2 = LM2Output(action_items=[{"task": "Fix the upload bug", "evidence_quote": "I'll fix the upload bug tonight.",
                                   "evidence_segment_ids": ["S001"]}])
    assert verify(lm2, unnamed, unnamed, Refinement(), meta(), settings, True).action_items[0].owner == "Unspecified"
    named = LM2Output(action_items=[{"task": "Fix the upload bug", "evidence_quote": "I'll fix the upload bug tonight.",
                                     "evidence_segment_ids": ["S005"]}])
    assert verify(named, segs(), segs(), Refinement(), meta(), settings, False).action_items[0].owner == "Unspecified"


def test_decision_agreement_is_taken_from_the_transcript(settings):
    lm2 = LM2Output(decisions=[{"decision": "Postpone live Google Meet capture",
                                "agreement_evidence": "they agreed to postpone", "evidence_segment_ids": ["S001", "S002"]}])
    rec = verify(lm2, segs(), segs(), Refinement(), meta(), settings, diarization_on=True)
    assert [d.agreement_evidence for d in rec.decisions] == ["Let's postpone that. For the demo we'll accept uploaded recordings only."]


def test_hedged_suggestion_is_not_promoted_to_a_decision(settings):
    s = [seg("S001", 0, "Maybe we should move to Postgres.", "Shivanshi"),
         seg("S002", 3, "Okay, final call: we ship v2 without the reranker.", "Prachi")]
    lm2 = LM2Output(decisions=[{"decision": "Move to Postgres", "agreement_evidence": "", "evidence_segment_ids": ["S001"]}])
    rec = verify(lm2, s, s, Refinement(), meta(), settings, diarization_on=True)
    assert rec.decisions == [] and rec.open_proposals[0].demoted_from_decision


def test_confident_respellings_without_lists_but_never_people():
    from app.config import Settings
    from app.models.record import Edit
    from app.pipeline.guard import check_edit, people_names

    s = Settings()
    meeting = [seg("S001", 0, "Hi Rahul, I am Priya.", "Speaker 1"),
               seg("S002", 3, "We travel to Shilong and deploy with Doker.", "Speaker 2"),
               seg("S003", 6, "Rahul said the Kubernets job failed.", "Speaker 1")]
    people = people_names(meeting)
    e = lambda o, r, sid, conf=0.9: Edit(segment_id=sid, original=o, replacement=r,  # noqa: E731
                                         category="proper_noun", rationale="", confidence=conf)
    by = {x.id: x for x in meeting}
    ok = lambda ed: check_edit(ed, by[ed.segment_id], set(), s, people)  # noqa: E731
    assert ok(e("Doker", "Docker", "S002")) is None
    assert ok(e("Kubernets", "Kubernetes", "S003")) is None
    assert ok(e("Rahul", "Raul", "S003")) == "name_changed"  # sentence-initial person name
    extra = seg("S004", 9, "The new Grafanna dashboard is live.", "Speaker 2")
    by["S004"] = extra
    assert ok(e("Grafanna", "Grafana", "S004")) is None  # not in any list: confident respelling
    assert ok(e("Grafanna", "Grafana", "S004", conf=0.7)) == "name_changed"  # not confident enough


WHISPERX = [
    ("S001", "Speaker 1", "Hi Sara, alright let's finalize the model architecture for the meeting assistance."),
    ("S002", "Speaker 1", "I think we should just use standard whisper and a basic API call to save time."),
    ("S003", "Speaker 2", "Hello Alex."),
    ("S004", "Speaker 2", "No, I strongly disagree. Standard whisper is terrible for speaker diarization."),
    ("S005", "Speaker 2", "We are not using standard whisper. We need whisper X so we can map tasks to specific people."),
    ("S006", "Speaker 1", "Ok"),
    ("S007", "Speaker 1", "fine, you are right. Whisper X it is. What about the transcript refinement?"),
    ("S008", "Speaker 2", "We need to fix domain terms like PyTorch or LoRa fine tuning."),
]


def test_conversational_agreement_makes_a_decision(settings):
    s = [seg(i, n * 3.0, t, spk) for n, (i, spk, t) in enumerate(WHISPERX)]
    lm2 = LM2Output(decisions=[{"decision": "Use WhisperX instead of standard Whisper",
                                "agreement_evidence": "they agreed", "evidence_segment_ids": ["S005", "S007"]}])
    rec = verify(lm2, s, s, Refinement(), meta(), settings, diarization_on=True)
    assert [d.decision for d in rec.decisions] == ["Use WhisperX instead of standard Whisper"]
    assert "Whisper X it is" in rec.decisions[0].agreement_evidence
    # A bare "Ok" right after the proposal also counts as acceptance.
    lm2 = LM2Output(decisions=[{"decision": "Use WhisperX", "agreement_evidence": "",
                                "evidence_segment_ids": ["S005"]}])
    rec = verify(lm2, s, s, Refinement(), meta(), settings, diarization_on=True)
    assert rec.decisions and rec.decisions[0].agreement_evidence == "Ok"


def test_phrases_that_are_not_agreement():
    from app.pipeline.verify import AGREEMENT_CUE

    for text in ["We need LoRa fine tuning.", "That is what it is.", "I don't know where it is.", "Is it fine tuning?"]:
        assert not AGREEMENT_CUE.search(text), text
    for text in ["Whisper X it is.", "You're right.", "Okay, fine.", "Fair enough, let's do that.", "That works."]:
        assert AGREEMENT_CUE.search(text), text


import pytest  # noqa: E402


@pytest.mark.parametrize("said, written", [
    ("Please check user underscore id underscore v2 in config dot py.", "Please check user_id_v2 in config.py."),
    ("Mail me at shivanshi at the rate gmail dot com today.", "Mail me at shivanshi@gmail.com today."),
    ("Tag at the rate frontend in the channel.", "Tag @frontend in the channel."),
    ("Open iitg dot ac dot in and hashtag launch it.", "Open iitg.ac.in and #launch it."),
    ("Interest is charged at the rate of 5 percent.", "Interest is charged at the rate of 5 percent."),
    ("The dot product is fine and the score went up.", "The dot product is fine and the score went up."),
])
def test_spoken_symbols(said, written):
    from app.pipeline.symbols import symbol_edits

    out, edits = symbol_edits(seg("S001", 0, said))
    assert out.text == written
    assert all(e.category == "symbol" for e in edits)
    assert out.words[0].start == 0.0  # word timings kept for playback
