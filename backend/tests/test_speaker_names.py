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
