"""Speaker names taken from the meeting itself ("Hello Prachi, I am Shivanshi")."""

from __future__ import annotations

import asyncio

import pytest

from app.pipeline.speaker_names import apply_speaker_names, find_speaker_names
from tests.conftest import seg


def names(lines):
    segs = [seg(f"S{i + 1:03d}", i * 3.0, t, spk) for i, (spk, t) in enumerate(lines)]
    return {k: v["name"] for k, v in find_speaker_names(segs).items()}


@pytest.mark.parametrize("lines, expected", [
    ([("Speaker 1", "Hello Prachi, I am Shivanshi."), ("Speaker 2", "Hi Shivanshi, nice to meet you.")],
     {"Speaker 1": "Shivanshi", "Speaker 2": "Prachi"}),
    ([("Speaker 1", "hello prachi i am shivanshi"), ("Speaker 2", "okay")],
     {"Speaker 1": "Shivanshi", "Speaker 2": "Prachi"}),
    ([("Speaker 1", "I'm Shivanshi from the ML team."), ("Speaker 2", "This is Prachi here.")],
     {"Speaker 1": "Shivanshi", "Speaker 2": "Prachi"}),
    ([("Speaker 1", "Prachi, can you check the upload?"), ("Speaker 2", "Sure."), ("Speaker 3", "Rahul here.")],
     {"Speaker 2": "Prachi", "Speaker 3": "Rahul"}),
])
def test_names_found(lines, expected):
    assert names(lines) == expected


@pytest.mark.parametrize("lines", [
    [("Speaker 1", "Hello Prachi."), ("Speaker 2", "Hi! Let's begin."), ("Speaker 1", "I am going to share.")],
    [("Speaker 1", "I am fine, thanks. This is Python code."), ("Speaker 2", "I'm in Guwahati today.")],
    [("Speaker 1", "i am tired today, i'm working on it, i am the lead"), ("Speaker 2", "hello everyone")],
])
def test_ordinary_words_places_and_tech_terms_are_not_names(lines):
    found = names(lines)
    assert not set(found.values()) - {"Prachi"}  # only the real addressee may be named


def test_self_introduction_beats_being_addressed_and_names_are_unique():
    found = names([
        ("Speaker 1", "Thanks Rahul."), ("Speaker 2", "I'm Prachi, actually."),  # addressed as Rahul, says Prachi
        ("Speaker 3", "Rahul here."),
    ])
    assert found == {"Speaker 2": "Prachi", "Speaker 3": "Rahul"}


def test_no_speakers_no_names():
    assert names([(None, "Hello Prachi, I am Shivanshi.")]) == {}


def test_apply_renames_every_segment_of_that_speaker():
    segs = [seg("S001", 0, "I am Shivanshi.", "Speaker 1"), seg("S002", 3, "Okay.", "Speaker 2"),
            seg("S003", 6, "Let's start.", "Speaker 1")]
    out = apply_speaker_names(segs, find_speaker_names(segs))
    assert [s.speaker for s in out] == ["Shivanshi", "Speaker 2", "Shivanshi"]


def test_pipeline_uses_names_in_transcript_and_record(settings, fixtures_dir, monkeypatch):
    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    class TwoSpeakers:
        name = "fake diarizer"

        def run(self, path):
            return [{"start": 0.0, "end": 2.99, "label": "a"}, {"start": 3.0, "end": 5.99, "label": "b"},
                    {"start": 6.0, "end": 99.0, "label": "a"}]

    async def go():
        db, d = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(),
                      Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM(), diarizer=TwoSpeakers()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    raw = read_json(d / "raw_transcript.json")
    # S001 "Hi all, this is Priya from the data team." is said by the first voice.
    assert raw[0]["speaker"] == "Priya"
    assert read_json(d / "speaker_names.json")["Speaker 1"]["evidence"][0]["segment_id"] == "S001"
    rec = read_json(d / "record.json")
    assert rec["raw_transcript"][0]["speaker"] == "Priya"


@pytest.mark.parametrize("text", [
    "Thanks, can you share the screen?", "Hi, can you hear me?", "Okay can we start?", "Thanks, could you check?",
    "Hello, should we begin?", "Hi, please go ahead.", "Thank you, will do.",
])
def test_helper_verbs_after_a_greeting_are_never_names(text):
    from app.pipeline.speaker_names import ADDRESS, _find

    assert _find(ADDRESS, text) == []


def test_real_meeting_lines_name_both_speakers():
    lines = [
        ("Speaker 1", "Hi Prachi, let's finalize our online Python workshop for the 12th of October."),
        ("Speaker 1", "We need to confirm the content, registrations and deadlines."),
        ("Speaker 2", "We have registrations from Goa, Bangalore and Kozhikode."),
        ("Speaker 1", "Thanks, can you share the list?"),
        ("Speaker 2", "Sure."),
        ("Speaker 2", "Shivanshi, should we include TensorFlow as well?"),
        ("Speaker 1", "Let's leave TensorFlow for a later workshop."),
    ]
    assert names(lines) == {"Speaker 1": "Shivanshi", "Speaker 2": "Prachi"}


def test_lowercase_name_only_from_a_self_introduction():
    assert names([("Speaker 1", "my name is shivanshi"), ("Speaker 2", "okay")]) == {"Speaker 1": "Shivanshi"}
    assert names([("Speaker 1", "thanks, okay then"), ("Speaker 2", "sure")]) == {}


ALEX_SARA = [
    ("S001", "Speaker 1", "Hi Sara, alright let's finalize the model architecture for the meeting assistance."),
    ("S002", "Speaker 1", "I think we should just use standard whisper and a basic API call to save time."),
    ("S003", "Speaker 2", "Hello Alex."),
    ("S004", "Speaker 2", "No, I strongly disagree. Standard whisper is terrible for speaker diarization."),
    ("S005", "Speaker 2", "We are not using standard whisper. We need whisper X so we can map tasks."),
    ("S006", "Speaker 1", "Ok"),
]


def _alex_sara():
    return [seg(i, n * 3.0, t, spk) for n, (i, spk, t) in enumerate(ALEX_SARA)]


def test_rules_never_take_standard_as_a_name():
    assert "Standard" not in {v["name"] for v in find_speaker_names(_alex_sara()).values()}


def test_model_names_are_checked_against_the_transcript():
    from app.models.llm_io import LM2Speaker
    from app.pipeline.speaker_names import verify_llm_names

    proposed = [
        LM2Speaker(label="Speaker 2", name="Sara", evidence_segment_id="S001"),   # addressed, then she speaks
        LM2Speaker(label="Speaker 1", name="Alex", evidence_segment_id="S003"),   # greeted right after speaking
        LM2Speaker(label="Speaker 1", name="Priya", evidence_segment_id="S002"),  # not said there: rejected
    ]
    got = verify_llm_names(proposed, _alex_sara())
    assert {k: v["name"] for k, v in got.items()} == {"Speaker 2": "Sara", "Speaker 1": "Alex"}


@pytest.mark.parametrize("label, name, sid", [
    ("Speaker 1", "Standard", "S004"),  # a common word in someone else's sentence
    ("Speaker 2", "Alex", "S002"),      # Alex is not said in S002
    ("Speaker 2", "Sara", "S005"),      # not said there
    ("Speaker 1", "Sara", "S001"),      # Speaker 1 says "Hi Sara": they are not Sara
    ("Speaker 9", "Sara", "S001"),      # no such speaker
])
def test_unsupported_model_names_are_rejected(label, name, sid):
    from app.models.llm_io import LM2Speaker
    from app.pipeline.speaker_names import verify_llm_names

    assert verify_llm_names([LM2Speaker(label=label, name=name, evidence_segment_id=sid)], _alex_sara()) == {}


def test_greeting_rule_never_rewrites_voice_labels():
    segs = [seg("S001", 0, "Hi Sara.", "Speaker 1"), seg("S002", 3, "Hello Alex.", "Speaker 2"),
            seg("S003", 6, "Let me add one thing.", "Speaker 2"), seg("S004", 9, "And another.", "Speaker 2"),
            seg("S005", 12, "Me too.", "Speaker 3")]
    before = [s.speaker for s in segs]
    find_speaker_names(segs)
    assert [s.speaker for s in segs] == before


def test_pipeline_applies_verified_model_names_to_transcript_and_record(settings, fixtures_dir):
    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline.runner import Services, run_job
    from tests.conftest import DEFAULT_LM2, FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    class TwoVoices:
        name = "fake diarizer"

        def run(self, path):
            return [{"start": 0.0, "end": 2.99, "label": "a"}, {"start": 3.0, "end": 99.0, "label": "b"}]

    lm2 = {**DEFAULT_LM2, "speakers": [{"label": "Speaker 2", "name": "Arjun", "evidence_segment_id": "S002"}],
           "summary": [{"text": "Speaker 2 said the CUDA kernels wait for the CI pipeline.", "evidence_segment_ids": ["S002"]}]}

    async def go():
        db, d = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(),
                      Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM(lm2=lm2), diarizer=TwoVoices()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    rec = read_json(d / "record.json")
    # "Arjun" is not said in S002, so the model's guess is rejected and the label stays.
    assert all(s["speaker"] != "Arjun" for s in rec["refined_transcript"])
