"""API with a mocked pipeline (spec Section 18)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.core.stages import STAGE_OUTPUT, Stage
from app.core.storage import write_json
from app.exports import render_exports
from app.main import create_app
from app.models.record import MeetingRecord
from tests.conftest import make_record


@pytest.fixture
def client(settings):
    holder = {}

    async def fake_runner(job_id: str) -> None:
        """Mocked pipeline: write a record and its exports, then complete."""
        state = holder["app"].state.trace
        d = settings.jobs_dir / job_id
        row = await state.db.get(job_id)
        rec = make_record(job_id, row["title"])
        write_json(d / STAGE_OUTPUT[Stage.RAW_SAVED], [s.model_dump(mode="json") for s in rec.raw_transcript])
        write_json(d / "refined_transcript.json", [s.model_dump(mode="json") for s in rec.refined_transcript])
        write_json(d / STAGE_OUTPUT[Stage.VERIFYING], rec)
        render_exports(rec, d / "exports")
        await state.db.update(job_id, status="completed", stage="completed", n_decisions=1, n_tasks=1)

    app = create_app(settings=settings, runner=fake_runner)
    holder["app"] = app
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id):
    for _ in range(50):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["job"]["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def upload(client, path, name=None, **data):
    with open(path, "rb") as f:
        return client.post("/api/jobs", files={"file": (name or path.name, f)}, data=data)


def test_empty_file_rejected(client, fixtures_dir):
    r = upload(client, fixtures_dir / "empty.mp3")
    assert r.status_code == 400
    assert r.json()["code"] == "E_EMPTY_FILE" and r.json()["message"]


def test_wrong_type_rejected(client, fixtures_dir):
    r = upload(client, fixtures_dir / "notes.pdf")
    assert r.status_code == 415 and r.json()["code"] == "E_UNSUPPORTED_FORMAT"


def test_full_lifecycle(client, fixtures_dir):
    r = upload(client, fixtures_dir / "tone.wav", title="  Sprint planning week 41  ", glossary="CI, CUDA")
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert r.json() == {"job_id": job_id, "status": "queued", "cached": False}

    body = wait_done(client, job_id)
    assert body["job"]["title"] == "Sprint planning week 41"
    assert body["record"]["decisions"][0]["id"] == "D1"
    assert body["partial"]["raw_transcript"]

    items = client.get("/api/jobs").json()
    assert items["total"] == 1 and items["items"][0]["id"] == job_id

    # Same file again: cached, unless forced.
    again = upload(client, fixtures_dir / "tone.wav")
    assert again.status_code == 200 and again.json() == {"job_id": job_id, "status": "completed", "cached": True}
    with open(fixtures_dir / "tone.wav", "rb") as f:
        forced = client.post("/api/jobs?force=true", files={"file": ("tone.wav", f)})
    assert forced.status_code == 202 and forced.json()["job_id"] != job_id
    wait_done(client, forced.json()["job_id"])
    # A result made by the local model is not reused for an upload with a cloud key.
    with open(fixtures_dir / "tone.wav", "rb") as f:
        keyed = client.post("/api/jobs", files={"file": ("tone.wav", f)}, data={"api_key": "k"})
    assert keyed.status_code == 202 and keyed.json()["cached"] is False
    wait_done(client, keyed.json()["job_id"])

    # Rename updates the row, the record and the download name.
    r = client.patch(f"/api/jobs/{job_id}", json={"title": "Renamed meeting"})
    assert r.status_code == 200 and r.json()["title"] == "Renamed meeting"
    assert client.get(f"/api/jobs/{job_id}").json()["record"]["meta"]["title"] == "Renamed meeting"

    r = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "json"})
    assert r.status_code == 200
    assert 'filename="renamed-meeting_record.json"' in r.headers["content-disposition"]
    MeetingRecord.model_validate_json(r.content)
    md = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "md"}).text
    assert md.startswith("# Renamed meeting") and "| T1 | Clean up the test data | Unspecified | Unspecified |" in md
    assert client.get(f"/api/jobs/{job_id}/export", params={"fmt": "docx"}).status_code == 200
    assert client.get(f"/api/jobs/{job_id}/export", params={"fmt": "txt_raw"}).text.startswith("[00:00:00] Speaker 1:")

    # Audio supports Range requests.
    full = client.get(f"/api/jobs/{job_id}/audio")
    part = client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-99"})
    assert full.status_code == 200 and part.status_code == 206 and len(part.content) == 100
    assert part.headers["content-range"].endswith(f"/{len(full.content)}")

    # SSE: a late subscriber gets the final state immediately.
    with client.stream("GET", f"/api/jobs/{job_id}/events") as s:
        text = "".join(s.iter_text())
    assert "event: progress" in text and '"status": "completed"' in text

    # Delete removes the row and the folder.
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get("/api/jobs").json()["total"] == 2  # the forced and the keyed upload remain


def test_bad_job_id(client):
    assert client.get("/api/jobs/../../etc").status_code == 404
    assert client.get("/api/jobs/" + "z" * 32).status_code == 404


def test_schema_and_health(client):
    assert "MeetingRecord" in client.get("/api/schema/record").json()["title"]
    assert client.get("/api/health").json()["status"] == "ok"
