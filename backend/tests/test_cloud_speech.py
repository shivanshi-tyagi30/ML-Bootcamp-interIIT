"""Hosted website speech: Whisper large-v3 on Groq for the words, AssemblyAI for who spoke when."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.pipeline import cloud_speech
from app.pipeline.cloud_speech import groq_to_segments

GROQ = {
    "language": "english",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": " Hi Sara, the CUDA kernels wait for CI.", "avg_logprob": -0.1,
         "no_speech_prob": 0.01, "compression_ratio": 1.2},
        {"start": 2.2, "end": 4.4, "text": " Hello Alex, I will fix it.", "avg_logprob": -0.05,
         "no_speech_prob": 0.01, "compression_ratio": 1.1},
    ],
    "words": [{"word": w, "start": 0.25 * i, "end": 0.25 * i + 0.2} for i, w in enumerate(
        ["Hi", "Sara,", "the", "CUDA", "kernels", "wait", "for", "CI."])]
    + [{"word": w, "start": 2.2 + 0.3 * i, "end": 2.2 + 0.3 * i + 0.25} for i, w in enumerate(
        ["Hello", "Alex,", "I", "will", "fix", "it."])],
}


def test_groq_answer_becomes_whisper_segments_with_words():
    out = groq_to_segments(GROQ)
    assert out["language"] == "en"
    assert [len(s["words"]) for s in out["segments"]] == [8, 6]
    first = out["segments"][0]["words"][0]
    assert first["w"] == "Hi" and first["start"] == 0.0 and 0.9 < first["conf"] <= 1.0


class FakeCloud:
    """Groq and AssemblyAI, as httpx.request sees them."""

    def __init__(self):
        self.calls = []

    def __call__(self, method, url, **kw):
        self.calls.append((method, url))
        if url.endswith("/audio/transcriptions"):
            assert kw["headers"]["Authorization"] == "Bearer groq-key"
            assert kw["data"]["model"] == "whisper-large-v3"
            assert kw["files"]["file"][0].endswith(".ogg")  # the compressed copy, not the WAV
            return httpx.Response(200, json=GROQ)
        if url.endswith("/v2/upload"):
            assert kw["headers"]["authorization"] == "aai-key"
            return httpx.Response(200, json={"upload_url": "https://cdn.example/audio"})
        if url.endswith("/v2/transcript"):
            assert kw["json"]["speaker_labels"] is True
            return httpx.Response(200, json={"id": "t1", "status": "queued"})
        if url.endswith("/v2/transcript/t1"):
            return httpx.Response(200, json={"status": "completed", "utterances": [
                {"speaker": "A", "start": 0, "end": 2100}, {"speaker": "B", "start": 2150, "end": 4500}]})
        raise AssertionError(url)


def test_hosted_pipeline_uses_groq_whisper_and_assemblyai_speakers(settings, fixtures_dir, monkeypatch):
    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM
    from tests.test_pipeline import JOB, _setup, _vad

    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    cloud = FakeCloud()
    monkeypatch.setattr(cloud_speech.httpx, "request", cloud)
    monkeypatch.setattr(cloud_speech.time, "sleep", lambda s: None)
    s = settings.model_copy(update={
        "STT_BACKEND": "groq", "GROQ_API_KEY": "groq-key", "DIARIZATION_ENABLED": True,
        "DIARIZATION_BACKEND": "assemblyai", "ASSEMBLYAI_API_KEY": "aai-key",
    })

    async def go():
        db, d = await _setup(s, fixtures_dir)
        await run_job(JOB, s, db, EventBus(), Services(vad=_vad, llm=FakeLLM()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    rec = read_json(d / "record.json")
    assert [x["text"] for x in rec["raw_transcript"]] == ["Hi Sara, the CUDA kernels wait for CI.",
                                                          "Hello Alex, I will fix it."]
    # Speaker turns from AssemblyAI, then names from the conversation ("Hi Sara" -> she answers).
    assert len({x["speaker"] for x in rec["raw_transcript"]}) == 2
    assert "Groq" in rec["meta"]["models"]["stt"] and "AssemblyAI" in rec["meta"]["models"]["diarization"]
    # AssemblyAI was started before Groq answered, so both ran at the same time.
    order = [u.rsplit("/", 1)[-1] for _, u in cloud.calls]
    assert order.index("upload") < order.index("t1")
    assert json.loads((d / "assemblyai.json").read_text())["id"] == "t1"


def test_missing_speech_keys_are_refused_at_upload(settings, fixtures_dir):
    from fastapi.testclient import TestClient

    from app.main import create_app

    async def idle(job_id):
        return None

    s = settings.model_copy(update={"STT_BACKEND": "groq", "DIARIZATION_ENABLED": True,
                                    "DIARIZATION_BACKEND": "assemblyai"})
    with TestClient(create_app(settings=s, runner=idle)) as c:
        assert len(c.get("/api/health").json()["setup_problems"]) == 2
        with open(fixtures_dir / "tone.wav", "rb") as f:
            r = c.post("/api/jobs", files={"file": ("tone.wav", f)})
        assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["message"]


def test_speaker_service_failure_never_fails_the_job(settings, fixtures_dir, monkeypatch):
    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM
    from tests.test_pipeline import JOB, _setup, _vad

    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("needs ffmpeg")

    def groq_only(method, url, **kw):
        if url.endswith("/audio/transcriptions"):
            return httpx.Response(200, json=GROQ)
        return httpx.Response(401, text="bad key")

    monkeypatch.setattr(cloud_speech.httpx, "request", groq_only)
    s = settings.model_copy(update={"STT_BACKEND": "groq", "GROQ_API_KEY": "k", "DIARIZATION_ENABLED": True,
                                    "DIARIZATION_BACKEND": "assemblyai", "ASSEMBLYAI_API_KEY": "wrong"})

    async def go():
        db, d = await _setup(s, fixtures_dir)
        await run_job(JOB, s, db, EventBus(), Services(vad=_vad, llm=FakeLLM()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    from app.core.stages import STAGE_OUTPUT, Stage

    diar = read_json(d / STAGE_OUTPUT[Stage.DIARIZING])
    assert diar["skipped"] and "API key" in diar["reason"]
    assert read_json(d / "record.json")["raw_transcript"]
