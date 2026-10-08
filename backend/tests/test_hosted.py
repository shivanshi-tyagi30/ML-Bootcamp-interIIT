"""Hosted website mode (LOCAL_LLM_ENABLED=false): every upload brings a cloud key; the page is served too."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline.runner import JOB_API_KEYS


def _app(settings, runner=None):
    async def idle(job_id: str) -> None:
        return None

    return create_app(settings=settings.model_copy(update={"LOCAL_LLM_ENABLED": False}), runner=runner or idle)


def test_upload_without_a_key_is_refused_with_a_clear_message(settings, fixtures_dir):
    with TestClient(_app(settings)) as c:
        with open(fixtures_dir / "tone.wav", "rb") as f:
            r = c.post("/api/jobs", files={"file": ("tone.wav", f)})
        assert r.status_code == 400 and r.json()["code"] == "E_NEEDS_KEY"
        assert "Gemini API key" in r.json()["message"]
        assert c.get("/api/jobs").json()["total"] == 0  # nothing stored
        health = c.get("/api/health").json()
        assert health["llm"]["key_required"] is True


def test_upload_with_a_key_is_accepted_and_retry_after_a_restart_asks_for_it_again(settings, fixtures_dir):
    async def fail(job_id: str) -> None:
        state = holder["app"].state.trace
        await state.db.update(job_id, status="failed", stage="failed", error_code="E_LM2_FAILED")

    holder = {}
    app = _app(settings, fail)
    holder["app"] = app
    with TestClient(app) as c:
        with open(fixtures_dir / "tone.wav", "rb") as f:
            r = c.post("/api/jobs", files={"file": ("tone.wav", f)}, data={"api_key": "user-key"})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        for _ in range(50):
            if c.get(f"/api/jobs/{job_id}").json()["job"]["status"] == "failed":
                break
        JOB_API_KEYS.pop(job_id, None)  # the server restarted: keys are memory-only
        assert c.post(f"/api/jobs/{job_id}/retry").json()["code"] == "E_NEEDS_KEY"
        assert c.post(f"/api/jobs/{job_id}/retry", data={"api_key": "user-key"}).status_code == 202
    JOB_API_KEYS.pop(job_id, None)


def test_website_is_served_from_the_same_address(settings, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>Trace</title>")
    s = settings.model_copy(update={"SERVE_FRONTEND": True, "FRONTEND_DIST": dist})
    with TestClient(_app(s)) as c:
        assert "<title>Trace</title>" in c.get("/").text
        assert c.get("/api/health").json()["status"] == "ok"
