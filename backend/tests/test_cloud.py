"""Cloud LLM client (Gemini / OpenAI-compatible): request shape, failure handling, fallback, key handling."""

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


class FakeLocal:
    def __init__(self):
        self.calls = 0

    async def json_call(self, model, system, user, schema, *a, **kw):
        self.calls += 1
        return schema(**GOOD)


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


def test_bad_key_is_a_clear_error_and_never_falls_back(monkeypatch):
    mock(monkeypatch, Recorder(httpx.Response(401, text="API key not valid")))
    local = FakeLocal()
    with pytest.raises(LLMUnavailable, match="API key"):
        run(GeminiClient("bad", fallback=local))
    assert local.calls == 0


def test_rate_limit_waits_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda s: asyncio.sleep(0) if False else _noop())
    rec = Recorder(httpx.Response(429, headers={"retry-after": "1"}), reply(json.dumps(GOOD)))
    mock(monkeypatch, rec)
    assert run(GeminiClient("key")).domain == "ml" and len(rec.bodies) == 2


async def _noop():
    return None


def test_no_internet_uses_the_local_model_once_and_says_so(monkeypatch):
    mock(monkeypatch, Recorder(httpx.ConnectError("no route")))
    local = FakeLocal()
    client = GeminiClient("key", fallback=local)
    assert run(client).domain == "ml"
    assert local.calls == 1 and client.used_fallback and "local fallback" in client.name


def test_cut_off_or_invalid_answer_is_retried_not_sent_to_the_local_model(monkeypatch):
    rec = Recorder(reply('{"domain": "ml", "ter', finish="length"), reply(json.dumps(GOOD)))
    mock(monkeypatch, rec)
    local = FakeLocal()
    assert run(GeminiClient("key", fallback=local)).domain == "ml"
    assert local.calls == 0 and len(rec.bodies) == 2


def test_invalid_twice_raises(monkeypatch):
    mock(monkeypatch, Recorder(reply("not json"), reply("still not json")))
    with pytest.raises(InvalidModelOutput):
        run(GeminiClient("key", fallback=FakeLocal()))


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
