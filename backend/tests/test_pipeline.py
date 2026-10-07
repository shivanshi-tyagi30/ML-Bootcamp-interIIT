"""End-to-end pipeline with fake models: real ffmpeg, guard, verifier and exports."""

from __future__ import annotations

import asyncio
import shutil

from app.core.db import JobDB
from app.core.events import EventBus
from app.core.stages import STAGE_OUTPUT, Stage
from app.core.storage import read_json, write_json
from app.models.record import MeetingRecord
from app.pipeline.runner import Services, run_job
from tests.conftest import FakeLLM, FakeSTT

JOB = "b" * 32


async def _setup(settings, fixtures_dir):
    db = JobDB(settings.db_path)
    await db.init()
    d = settings.jobs_dir / JOB
    d.mkdir(parents=True)
    shutil.copy(fixtures_dir / "tone.wav", d / "original.wav")
    write_json(d / STAGE_OUTPUT[Stage.VALIDATING], {"ext": "wav", "title": "Weekly sync", "source_file": "tone.wav",
                                                    "glossary": ["CUDA"]})
    await db.insert({"id": JOB, "title": "Weekly sync", "source_file": "tone.wav", "file_sha256": "x",
                     "status": "queued", "stage": "queued"})
    return db, d


def _vad(samples, sr):
    return [{"start": 0.0, "end": 4.5}]


def test_pipeline_end_to_end(settings, fixtures_dir):
    async def go():
        db, d = await _setup(settings, fixtures_dir)
        bus = EventBus()
        events = []
        q = bus.subscribe(JOB)
        await run_job(JOB, settings, db, bus, Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM()))
        while not q.empty():
            events.append(q.get_nowait())
        return db, d, events

    db, d, events = asyncio.run(go())
    row = asyncio.run(db.get(JOB))
    assert row["status"] == "completed", row
    rec = MeetingRecord(**read_json(d / "record.json"))

    raw = {s.id: s.text for s in rec.raw_transcript}
    ref = {s.id: s.text for s in rec.refined_transcript}
    assert "cuda" in raw["S002"] and "CUDA kernels" in ref["S002"] and "CI pipeline" in ref["S002"]
    assert ref["S003"] == raw["S003"]  # fifteen -> fifty was blocked
    assert [e.reject_reason for e in rec.refinement.rejected] == ["number_changed"]
    assert rec.fidelity.numbers_preserved.split("/")[0] == rec.fidelity.numbers_preserved.split("/")[1]
    assert rec.meta.models["stt"] == "fake-whisper" and rec.meta.title == "Weekly sync"
    assert [t.deadline for t in rec.action_items] == ["Monday", "Unspecified"]
    assert (d / "exports" / "record.docx").exists() and (d / "timings.json").exists()
    assert events[-1]["status"] == "completed" and events[-1]["record_ready"]
    assert any(e["raw_ready"] and not e["record_ready"] for e in events)
    assert row["n_decisions"] == 1 and row["n_tasks"] == 2


def test_lm2_failure_keeps_transcripts(settings, fixtures_dir):
    async def go():
        db, d = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM(fail_lm2=True)))
        return db, d

    db, d = asyncio.run(go())
    row = asyncio.run(db.get(JOB))
    assert row["status"] == "failed" and row["error_code"] == "E_LM2_FAILED"
    assert "Both transcripts are still available" in row["error_message"]
    assert (d / "raw_transcript.json").exists() and (d / "refined_transcript.json").exists()
    assert not (d / "record.json").exists()


def test_resume_skips_finished_stages(settings, fixtures_dir):
    async def go():
        db, d = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM(fail_lm2=True)))
        llm = FakeLLM()
        await run_job(JOB, settings, db, EventBus(), Services(stt=None, vad=_vad, llm=llm))
        return db, llm

    db, llm = asyncio.run(go())
    assert asyncio.run(db.get(JOB))["status"] == "completed"
    expected_model = "LM2Output" if settings.LM2_SCRATCHPAD else "LM2Record"
    assert llm.calls == [expected_model]  # only the failed stage and later ones ran again
