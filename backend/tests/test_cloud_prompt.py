"""The cloud minute-taking prompt: its own worked example must pass the verifier untouched."""

from __future__ import annotations

import json
import re

from app.llm.client import load_prompt
from app.models.llm_io import LM2Record
from app.models.record import Refinement
from app.pipeline.verify import apply_model_names, verify
from tests.conftest import seg
from tests.test_accuracy import meta


def _example():
    """Transcript lines and the expected JSON of the first worked example in the prompt."""
    text = load_prompt("lm2_document_cloud")
    block = text[text.index("EXAMPLE OF THE STANDARD"):text.index("(WRONG summary")]
    lines = re.findall(r"^\[(S\d+)\] \((Speaker \d+)\) (.+)$", block, re.M)
    segs = [seg(i, n * 3.0, t, spk) for n, (i, spk, t) in enumerate(lines)]
    fields = {}
    for name in ("speakers", "summary", "minutes", "decisions", "open_proposals", "action_items"):
        m = re.search(name + r": (\[.*?\])\s*(?=\n\s*\w+: \[|\Z)", block[block.index("->"):], re.S)
        fields[name] = json.loads(m.group(1))
    return segs, LM2Record(**fields)


def test_prompt_example_survives_the_verifier(settings):
    segs, lm2 = _example()
    lm2, raw, refined, named, _ = apply_model_names(lm2, segs, segs)
    assert {k: v["name"] for k, v in named.items()} == {"Speaker 2": "Sara", "Speaker 1": "Alex"}
    rec = verify(lm2, raw, refined, Refinement(), meta(), settings, diarization_on=True)
    assert len(rec.summary) == len(lm2.summary)
    assert sum(len(t.points) for t in rec.minutes) == sum(len(t.points) for t in lm2.minutes)
    assert [d.decision for d in rec.decisions] == ["Use WhisperX instead of standard Whisper"]
    assert len(rec.open_proposals) == 1
    task = rec.action_items[0]
    assert task.owner == "Alex" and task.deadline == "by Friday"
    # Names replace the labels in the model's own text too.
    assert rec.summary[1].text.startswith("Sara argued")


def test_cloud_record_uses_the_minute_taking_prompt(settings, fixtures_dir, monkeypatch):
    import asyncio

    import app.llm.client as client_mod
    from app.core.events import EventBus
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    systems = {}

    class CloudFake(FakeLLM):
        """Counts as the cloud model (isinstance check) and records each system prompt."""

        name = "fake-flash (cloud)"

        async def json_call(self, model, system, user, schema, *a, **kw):
            systems[schema.__name__] = system
            return await super().json_call(model, system, user, schema, *a, **kw)

    monkeypatch.setattr(client_mod, "GeminiClient", CloudFake)

    async def go():
        db, _ = await _setup(settings, fixtures_dir)
        await run_job(JOB, settings, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=CloudFake()))
        return await db.get(JOB)

    row = asyncio.run(go())
    assert row["status"] == "completed", row
    lm2 = next(v for k, v in systems.items() if k.startswith("LM2"))
    assert "expert minute-taker" in lm2 and "scratchpad" not in lm2.lower()
