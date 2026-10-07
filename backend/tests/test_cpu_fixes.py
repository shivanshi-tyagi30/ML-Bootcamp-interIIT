"""Cancel/retry, low-confidence fallback, context sizing and lenient LM2 parsing."""

from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient

from app.core.errors import JobCancelled
from app.llm.client import context_size, is_thinking_model
from app.main import create_app
from app.models.llm_io import LM2Output
from app.pipeline.recheck import low_confidence_fallback


def _wait(client, job_id, statuses):
    for _ in range(100):
        row = client.get(f"/api/jobs/{job_id}").json()["job"]
        if row["status"] in statuses:
            return row
        time.sleep(0.05)
    raise AssertionError(f"job never reached {statuses}")


def test_cancel_then_retry(settings, fixtures_dir):
    runs = {"n": 0}
    holder = {}

    async def runner(job_id: str) -> None:
        """First run hangs until cancelled; the retry completes."""
        state = holder["app"].state.trace
        runs["n"] += 1
        await state.db.update(job_id, status="running", stage="transcribing")
        if runs["n"] == 1:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await state.db.update(job_id, status="failed", stage="failed", error_code="E_CANCELLED",
                                      error_message="Processing was cancelled.", error_detail="Cancelled by the user.")
                return
        await state.db.update(job_id, status="completed", stage="completed")

    app = create_app(settings=settings, runner=runner)
    holder["app"] = app
    with TestClient(app) as c:
        with open(fixtures_dir / "tone.wav", "rb") as f:
            job_id = c.post("/api/jobs", files={"file": ("tone.wav", f)}).json()["job_id"]
        _wait(c, job_id, {"running"})
        assert c.delete(f"/api/jobs/{job_id}").status_code == 409
        r = c.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200 and r.json()["error_code"] == "E_CANCELLED"
        assert r.json()["error_detail"] == "Cancelled by the user."
        r = c.post(f"/api/jobs/{job_id}/retry")
        assert r.status_code == 202 and r.json()["status"] == "queued"
        assert _wait(c, job_id, {"completed"})["error_code"] is None
        assert runs["n"] == 2
        assert c.post("/api/jobs/" + "a" * 32 + "/retry").status_code == 404


def test_cancel_without_live_task_marks_failed(settings, fixtures_dir):
    async def runner(job_id: str) -> None:
        return None  # leaves the row queued, as after a crash

    with TestClient(create_app(settings=settings, runner=runner)) as c:
        with open(fixtures_dir / "tone.wav", "rb") as f:
            job_id = c.post("/api/jobs", files={"file": ("tone.wav", f)}).json()["job_id"]
        time.sleep(0.1)
        body = c.post(f"/api/jobs/{job_id}/cancel").json()
        assert body["status"] == "failed" and body["error_code"] == "E_CANCELLED"


def test_health_reports_tools(settings):
    with TestClient(create_app(settings=settings)) as c:
        h = c.get("/api/health").json()
    assert h["ffmpeg"] in (True, False)
    assert h["llm"]["reachable"] in (True, False)
    assert h["whisper"]["device"] in ("cpu", "cuda")


def test_low_confidence_fallback():
    segs = [{"words": [{"w": "a", "conf": 0.9}, {"w": "b", "conf": 0.2}]}, {"words": [{"w": "c", "conf": 0.1}]}]
    out = low_confidence_fallback(segs, 0.45)
    assert out == {"1": {"disputed": True, "alt": None}, "2": {"disputed": True, "alt": None}}
    assert low_confidence_fallback(segs, 0) == {}


def test_context_size_buckets():
    assert context_size("x" * 100, "y" * 100, 1024, 32768) == 8192
    assert context_size("x" * 40000, "y" * 40000, 4096, 32768) in (32768,)
    assert context_size("x" * 400000, "", 4096, 16384) == 16384
    assert is_thinking_model("qwen3:8b") and not is_thinking_model("gemma3:12b")


def test_lm2_lenient_parse():
    out = LM2Output.model_validate({"summary": [{"text": "Hi", "evidence_segment_ids": ["S001"]}],
                                    "action_items": [{"task": "Do it", "evidence_segment_ids": ["S001"]}]})
    assert out.decisions == [] and out.action_items[0].evidence_quote == ""
    assert list(LM2Output.model_json_schema()["properties"])[0] == "scratchpad"


def test_job_cancelled_is_exception():
    assert issubclass(JobCancelled, Exception)


def test_pipeline_cancel_mid_transcription(settings, fixtures_dir):
    from app.core.events import EventBus
    from app.pipeline.runner import Services, request_cancel, run_job
    from tests.conftest import FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    class CancellingSTT(FakeSTT):
        def transcribe(self, path, initial_prompt, progress=None, should_stop=None):
            request_cancel(JOB)
            if should_stop and should_stop():
                raise JobCancelled()
            return super().transcribe(path, initial_prompt)

    async def go():
        db, d = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(), Services(stt=CancellingSTT(), vad=_vad))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "failed" and row["error_code"] == "E_CANCELLED"
    assert not (d / "03_whisper.json").exists()


def test_ollama_request_shape_and_errors(monkeypatch):
    import httpx
    import pytest

    from app.llm.client import LLMClient, LLMUnavailable
    from app.models.llm_io import Vocabulary

    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(req.content)
        seen.append(body)
        if body["model"] == "missing:1b":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json={"message": {"content": '{"domain": "ml", "terms": []}'},
                                         "prompt_eval_count": 12})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    llm = LLMClient("http://localhost:11434/v1", "ollama", 30)
    out = asyncio.run(llm.json_call("qwen3:8b", "sys", "hello", Vocabulary, 1, max_tokens=512))
    assert out.domain == "ml"
    b = seen[0]
    assert b["think"] is False and b["stream"] is False and b["format"]["type"] == "object"
    assert b["options"]["num_ctx"] == 8192 and b["options"]["num_predict"] == 512
    assert b["messages"][1]["content"].endswith("/no_think")
    with pytest.raises(LLMUnavailable, match="ollama pull missing:1b"):
        asyncio.run(llm.json_call("missing:1b", "sys", "hi", Vocabulary, 1))


def test_pipeline_without_scratchpad(settings, fixtures_dir):
    from app.core.events import EventBus
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    s = settings.model_copy(update={"LM2_SCRATCHPAD": False})
    llm = FakeLLM()

    async def go():
        db, _ = await _setup(s, fixtures_dir)
        await run_job(JOB, s, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=llm))
        return await db.get(JOB)

    row = asyncio.run(go())
    assert row["status"] == "completed", row
    assert "LM2Record" in llm.calls and row["n_tasks"] == 2


def test_unload_posts_keep_alive_zero(monkeypatch):
    import json

    import httpx

    from app.llm.client import LLMClient

    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    asyncio.run(LLMClient("http://localhost:11434/v1", "ollama", 30).unload("qwen3:8b"))
    assert seen == [("/api/generate", {"model": "qwen3:8b", "keep_alive": 0})]


def test_no_guessed_speakers_without_diarization():
    from app.pipeline.assemble import assemble_segments

    segs = [{"words": [{"w": "Is it done?", "start": 0.0, "end": 0.5, "conf": 0.9},
                       {"w": "Yes.", "start": 3.0, "end": 3.5, "conf": 0.9}]}]
    assert all(s.speaker is None for s in assemble_segments(segs, {}, []))


def test_place_name_spelling_is_corrected_but_people_names_are_not():
    from app.config import Settings
    from app.models.record import Edit
    from app.pipeline.guard import check_edit
    from tests.conftest import seg

    s = Settings()
    edit = lambda o, r, cat="proper_noun": Edit(  # noqa: E731
        segment_id="S001", original=o, replacement=r, category=cat, rationale="", confidence=0.9)
    sg = seg("S001", 0, "We met at IIT Guhati with Priya last week.")
    assert check_edit(edit("Guhati", "Guwahati"), sg, set(), s) is None
    assert check_edit(edit("Guhati", "Guwahati", "homophone"), sg, set(), s) is None
    assert check_edit(edit("Priya", "Pria"), sg, set(), s) == "name_changed"
    sg2 = seg("S001", 0, "The demo uses Lang Chain for retrieval.")
    assert check_edit(edit("Lang Chain", "LangChain", "product"), sg2, {"langchain"}, s) is None


def test_verifier_borrows_only_from_neighbouring_lines(settings):
    from app.models.llm_io import LM2Output
    from app.models.record import Meta, Refinement
    from app.pipeline.verify import verify
    from tests.conftest import sample_segments

    segs = sample_segments()
    lm2 = LM2Output(summary=[
        # "Priya" is in S001, right before the cited S002: kept, with S001 added as evidence.
        {"text": "The kernels wait on the pipeline, said Priya.", "evidence_segment_ids": ["S002"]},
        # "Priya" is only in S001, far from S008: removed.
        {"text": "The test data needs cleaning, said Priya.", "evidence_segment_ids": ["S008"]},
    ])
    meta = Meta(job_id="x", title="t", source_file="f", duration_s=1, language="en", language_probability=1,
                models={}, generated_at="2026-01-01T00:00:00Z")
    rec = verify(lm2, segs, segs, Refinement(), meta, settings, diarization_on=False)
    assert [(s.text[:11], s.evidence_segment_ids) for s in rec.summary] == [("The kernels", ["S001", "S002"])]
