"""ECAPA speaker labels: burst building and clustering (synthetic embeddings, no model download)."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.ecapa_diarize import cluster_bursts, make_bursts

pytest.importorskip("sklearn")


def _voice(rng, base, noise=0.25):
    v = base + noise * rng.standard_normal(192)
    return v / np.linalg.norm(v)


def _voices(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng, [rng.standard_normal(192) for _ in range(n)]


def test_make_bursts_splits_on_pause_and_length():
    words = [{"start": 0.0, "end": 0.4}, {"start": 0.5, "end": 0.9}, {"start": 2.0, "end": 2.3}]
    assert make_bursts(words) == [(0.0, 0.9), (2.0, 2.3)]
    long = [{"start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(40)]  # 20 s without a pause
    assert all(b - a <= 8.0 for a, b in make_bursts(long))


def test_same_voice_after_a_long_gap_keeps_its_label():
    rng, (a, b, _) = _voices()
    seq = [a, b, a, b, a, a, b]
    labels = cluster_bursts(np.stack([_voice(rng, v) for v in seq]), [2.5] * len(seq))
    assert labels == [0, 1, 0, 1, 0, 0, 1]


def test_short_replies_join_an_existing_speaker():
    rng, (a, b, _) = _voices(seed=1)
    seq = [(a, 3.0), (b, 3.0), (b, 0.4), (a, 2.0), (a, 0.3)]  # "Yes." / "Okay." are short
    labels = cluster_bursts(np.stack([_voice(rng, v) for v, _ in seq]), [d for _, d in seq])
    assert labels == [0, 1, 1, 0, 0]
    assert max(labels) == 1  # no extra speaker from the short clips


def test_tiny_cluster_is_merged_but_fixed_count_is_respected():
    rng, (a, b, c) = _voices(seed=2)
    seq = [(a, 4.0), (b, 4.0), (a, 4.0), (c, 1.2), (b, 4.0)]  # c talks for only 1.2 s
    embs = np.stack([_voice(rng, v) for v, _ in seq])
    durs = [d for _, d in seq]
    assert max(cluster_bursts(embs, durs)) == 1  # noise does not become "Speaker 3"
    assert max(cluster_bursts(embs, durs, n_speakers=3)) == 2  # unless the user says there were 3


def test_single_burst():
    assert cluster_bursts(np.ones((1, 192)), [1.0]) == [0]
    assert cluster_bursts(np.zeros((0, 192)), []) == []


def test_stage_labels_reach_the_transcript(settings, fixtures_dir, monkeypatch):
    import asyncio

    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline import diarize
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    class FakeEcapa:
        name = "ECAPA-TDNN (fake)"
        loaded = True

        def run(self, path, whisper_segments, should_stop=None):
            words = [w for s in whisper_segments for w in s["words"]]
            mid = words[len(words) // 2]["start"]
            return [{"start": 0.0, "end": mid - 0.01, "label": "7"}, {"start": mid, "end": 99.0, "label": "3"}]

    monkeypatch.setattr(diarize, "default_ecapa_diarizer", lambda s: FakeEcapa())
    s = settings.model_copy(update={"DIARIZATION_ENABLED": True, "DIARIZATION_BACKEND": "ecapa"})

    async def go():
        db, d = await _setup(s, fixtures_dir)
        await run_job(JOB, s, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    speakers = [x["speaker"] for x in read_json(d / "raw_transcript.json")]
    # The first voice says "this is Priya", so it is named; the other keeps its label.
    assert set(speakers) == {"Priya", "Speaker 2"} and speakers[0] == "Priya"
    assert read_json(d / "record.json")["meta"]["models"]["diarization"] == "ECAPA-TDNN (fake)"


def test_stage_skips_cleanly_without_speechbrain(settings, fixtures_dir, monkeypatch):
    import asyncio
    import builtins

    from app.core.events import EventBus
    from app.core.storage import read_json
    from app.pipeline import diarize
    from app.pipeline.runner import Services, run_job
    from tests.conftest import FakeLLM, FakeSTT
    from tests.test_pipeline import JOB, _setup, _vad

    real_import = builtins.__import__

    def no_speechbrain(name, *a, **kw):
        if name.startswith("speechbrain"):
            raise ImportError("no speechbrain", name="speechbrain")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_speechbrain)
    monkeypatch.setattr(diarize, "_default_ecapa", None)
    s = settings.model_copy(update={"DIARIZATION_ENABLED": True, "DIARIZATION_BACKEND": "ecapa"})

    async def go():
        db, d = await _setup(s, fixtures_dir)
        await run_job(JOB, s, db, EventBus(), Services(stt=FakeSTT(), vad=_vad, llm=FakeLLM()))
        return await db.get(JOB), d

    row, d = asyncio.run(go())
    assert row["status"] == "completed", row
    assert "speechbrain is not installed" in read_json(d / "record.json")["meta"]["models"]["diarization"]
