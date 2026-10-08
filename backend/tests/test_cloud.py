"""Cloud LLM client (Gemini / OpenAI-compatible): request shape, failure handling, busy models, key handling."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.llm.client import GeminiClient, LLMUnavailable
from app.llm.json_repair import InvalidModelOutput
from app.models.llm_io import LM2Output, Vocabulary

GOOD = {"domain": "ml", "terms": []}


def mock(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def reply(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, req):
        self.bodies.append(json.loads(req.content))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def run(client, schema=Vocabulary, max_tokens=1024):
    return asyncio.run(client.json_call("lm", "sys", "user", schema, 1, max_tokens=max_tokens))


def test_asks_for_our_exact_json_structure_and_leaves_room_for_thinking(monkeypatch):
    rec = Recorder(reply(json.dumps(GOOD)))
    mock(monkeypatch, rec)
    out = run(GeminiClient("key", "gemini-2.5-flash"))
    assert out.domain == "ml"
    body = rec.bodies[0]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and "$ref" not in json.dumps(fmt) and "$defs" not in json.dumps(fmt)
    assert body["max_tokens"] >= 8192 and body["reasoning_effort"] == "low"
    assert body["model"] == "gemini-2.5-flash"


def test_full_record_schema_is_inlined_for_the_provider():
    from app.llm.client import inline_schema

    text = json.dumps(inline_schema(LM2Output))
    assert "$ref" not in text and "speakers" in text and "evidence_segment_ids" in text


def test_provider_rejecting_json_schema_gets_the_structure_in_the_prompt(monkeypatch):
    rec = Recorder(httpx.Response(400, text="Invalid JSON schema in response_format"), reply(json.dumps(GOOD)))
    mock(monkeypatch, rec)
    run(GeminiClient("key"))
    assert rec.bodies[1]["response_format"] == {"type": "json_object"}
    assert "exactly this structure" in rec.bodies[1]["messages"][-1]["content"]


def test_bad_key_is_a_clear_error(monkeypatch):
    mock(monkeypatch, Recorder(httpx.Response(401, text="API key not valid")))
    with pytest.raises(LLMUnavailable, match="API key"):
        run(GeminiClient("bad"))


def test_rate_limit_waits_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda s: asyncio.sleep(0) if False else _noop())
    rec = Recorder(httpx.Response(429, headers={"retry-after": "1"}), reply(json.dumps(GOOD)))
    mock(monkeypatch, rec)
    assert run(GeminiClient("key")).domain == "ml" and len(rec.bodies) == 2


async def _noop():
    return None


def test_no_internet_is_a_clear_error_never_the_local_model(monkeypatch):
    mock(monkeypatch, Recorder(httpx.ConnectError("no route")))
    with pytest.raises(LLMUnavailable, match="internet.*Retry"):
        run(GeminiClient("key"))


def test_cut_off_or_invalid_answer_is_retried(monkeypatch):
    rec = Recorder(reply('{"domain": "ml", "ter', finish="length"), reply(json.dumps(GOOD)))
    mock(monkeypatch, rec)
    assert run(GeminiClient("key")).domain == "ml" and len(rec.bodies) == 2


def test_invalid_twice_raises(monkeypatch):
    mock(monkeypatch, Recorder(reply("not json"), reply("still not json")))
    with pytest.raises(InvalidModelOutput):
        run(GeminiClient("key"))


def test_api_key_is_never_written_to_disk(settings, fixtures_dir):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.pipeline.runner import JOB_API_KEYS

    async def runner(job_id):
        return None

    with TestClient(create_app(settings=settings, runner=runner)) as c:
        with open(fixtures_dir / "tone.wav", "rb") as f:
            job_id = c.post("/api/jobs", files={"file": ("tone.wav", f)}, data={"api_key": "SECRET-123"}).json()["job_id"]
    on_disk = "".join(p.read_text(errors="ignore") for p in settings.DATA_DIR.rglob("*") if p.is_file()
                      and p.suffix in (".json", ".log", ".txt"))
    assert "SECRET-123" not in on_disk
    assert JOB_API_KEYS[job_id] == "SECRET-123"


def test_old_saved_keys_are_scrubbed_at_startup(settings):
    from fastapi.testclient import TestClient

    from app.core.storage import read_json, write_json
    from app.main import create_app

    f = settings.jobs_dir / ("c" * 32) / "00_upload.json"
    write_json(f, {"title": "x", "api_key": "OLD-SECRET"})

    async def runner(job_id):
        return None

    with TestClient(create_app(settings=settings, runner=runner)):
        pass
    assert "api_key" not in read_json(f)


def test_record_names_the_cloud_model(settings, fixtures_dir, monkeypatch):
    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    class FakeCloud(GeminiClient, FakeLLM):
        def __init__(self):
            GeminiClient.__init__(self, "key", "gemini-2.5-flash")
            FakeLLM.__init__(self)

        json_call = FakeLLM.json_call

    async def go():
        db, d = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=FakeCloud()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    models = read_json(d / "record.json")["meta"]["models"]
    assert models["lm1"] == models["lm2"] == "gemini-2.5-flash (cloud)"


def test_retired_model_is_replaced_by_the_newest_flash_model(monkeypatch):
    models = {"data": [{"id": m} for m in [
        "models/gemini-2.0-flash", "models/gemini-3.0-flash", "models/gemini-3.0-flash-lite",
        "models/gemini-3.0-pro", "models/gemini-3.1-flash-preview-09-2026", "models/text-embedding-004",
        "models/gemini-3.0-flash-image"]]}
    seen = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=models)
        body = json.loads(req.content)
        seen.append(body["model"])
        if body["model"] == "gemini-2.5-flash":
            return httpx.Response(404, json={"error": {"message": "models/gemini-2.5-flash is not found"}})
        return reply(json.dumps(GOOD))

    mock(monkeypatch, handler)
    client = GeminiClient("key", "gemini-2.5-flash")
    assert run(client).domain == "ml"
    assert seen == ["gemini-2.5-flash", "gemini-3.0-flash"] and client.model == "gemini-3.0-flash"


def test_unknown_model_without_alternatives_lists_what_the_key_can_use(monkeypatch):
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "models/text-embedding-004"}]})
        return httpx.Response(404, text="not found")

    mock(monkeypatch, handler)
    with pytest.raises(LLMUnavailable, match="text-embedding-004"):
        run(GeminiClient("key", "gemini-2.5-flash"))


def test_overloaded_model_switches_to_another_cloud_model_not_the_local_one(monkeypatch):
    from app.llm import client as client_mod

    monkeypatch.setattr(client_mod, "RESOLVED_MODELS", {})
    monkeypatch.setattr(asyncio, "sleep", lambda s: _noop())
    models = {"data": [{"id": f"models/{m}"} for m in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]]}
    seen = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=models)
        body = json.loads(req.content)
        seen.append(body["model"])
        if body["model"] == "gemini-2.5-flash":
            return httpx.Response(503, json={"error": {"code": 503, "message": "high demand"}})
        return reply(json.dumps(GOOD))

    mock(monkeypatch, handler)
    client = GeminiClient("key", "gemini-2.5-flash")
    assert run(client).domain == "ml"
    assert seen == ["gemini-2.5-flash"] * 4 + ["gemini-2.0-flash"]
    assert client.model == "gemini-2.0-flash" and client.name == "gemini-2.0-flash (cloud)"


def test_every_model_overloaded_is_a_clear_retry_message(monkeypatch):
    from app.llm import client as client_mod

    monkeypatch.setattr(client_mod, "RESOLVED_MODELS", {})
    monkeypatch.setattr(asyncio, "sleep", lambda s: _noop())

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-flash"}]})
        return httpx.Response(503, text="high demand")

    mock(monkeypatch, handler)
    with pytest.raises(LLMUnavailable, match="overloaded.*Retry"):
        run(GeminiClient("key", "gemini-2.5-flash"))
