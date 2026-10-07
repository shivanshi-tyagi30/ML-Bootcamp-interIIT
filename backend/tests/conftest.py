"""Shared fixtures: generated audio files, settings in a temp folder, fake models (spec Section 18)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.models.llm_io import LM1Output, LM2Output, LM2Record, Vocabulary
from app.models.record import (
    Fidelity, MeetingRecord, Meta, Refinement, Segment, Word,
)


import shutil
import numpy as np
import soundfile as sf


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """empty.mp3, fake.mp3, silence.wav, tone.wav and notes.pdf, generated once."""
    d = tmp_path_factory.mktemp("fixtures")
    (d / "empty.mp3").write_bytes(b"")
    (d / "fake.mp3").write_text("this is not audio, just text renamed to mp3\n" * 20)
    (d / "notes.pdf").write_bytes(b"%PDF-1.4\n%fake pdf\n")
    if shutil.which("ffmpeg"):
        _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", str(d / "silence.wav"))
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "5", str(d / "tone.wav"))
    else:
        # Generate silence and tone directly via soundfile when ffmpeg is not on PATH
        sr = 16000
        silence_data = np.zeros(sr * 5, dtype=np.float32)
        sf.write(str(d / "silence.wav"), silence_data, sr)
        t = np.linspace(0, 5, sr * 5, endpoint=False, dtype=np.float32)
        tone_data = np.sin(2 * np.pi * 440 * t)
        sf.write(str(d / "tone.wav"), tone_data, sr)
    return d


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with data in a temp folder and optional models off."""
    return Settings(DATA_DIR=tmp_path / "data", RECHECK_ENABLED=False, DIARIZATION_ENABLED=False, WARMUP_ON_START=False)


def seg(id_: str, start: float, text: str, speaker: str | None = None, disputed: dict[str, str] | None = None) -> Segment:
    """A segment whose words are spread evenly over 3 seconds."""
    toks = text.split()
    step = 3.0 / max(1, len(toks))
    disputed = disputed or {}
    words = [
        Word(w=t, start=start + i * step, end=start + (i + 1) * step, conf=0.9,
             disputed=t.strip(".,?!") in disputed, alt=disputed.get(t.strip(".,?!")))
        for i, t in enumerate(toks)
    ]
    return Segment(id=id_, start=start, end=start + 3.0, speaker=speaker, text=text, words=words)


SAMPLE_LINES = [
    ("S001", "Speaker 1", "Hi all, this is Priya from the data team."),
    ("S002", "Speaker 2", "We can't push the cuda kernels until the sea eye pipeline passes."),
    ("S003", "Speaker 2", "Latency went from fifteen to fifty milliseconds."),
    ("S004", "Speaker 3", "Maybe we should move the vector store to Postgres."),
    ("S005", "Speaker 1", "Okay, final call: we ship v2 without the reranker."),
    ("S006", "Speaker 2", "Agreed."),
    ("S007", "Speaker 1", "I'll update the dashboards by Friday... no, Monday."),
    ("S008", "Speaker 2", "Someone needs to clean up the test data."),
]


def sample_segments() -> list[Segment]:
    """The trap-case meeting used across tests."""
    return [seg(i, n * 3.0, t, spk) for n, (i, spk, t) in enumerate(SAMPLE_LINES)]


def make_record(job_id: str, title: str = "Weekly sync") -> MeetingRecord:
    """A small valid record (used by the mocked pipeline in API tests)."""
    raw = sample_segments()
    return MeetingRecord(
        meta=Meta(job_id=job_id, title=title, source_file="weekly.wav", duration_s=24.0, language="en",
                  language_probability=0.99, models={"stt": "fake", "lm1": "fake", "lm2": "fake"},
                  generated_at=datetime.now(timezone.utc)),
        raw_transcript=raw, refined_transcript=raw, refinement=Refinement(),
        summary=[{"text": "The team agreed to ship v2 without the reranker.", "evidence_segment_ids": ["S005", "S006"]}],
        minutes=[{"topic": "Launch", "points": [{"text": "v2 ships without the reranker.", "evidence_segment_ids": ["S005"]}]}],
        decisions=[{"id": "D1", "decision": "Ship v2 without the reranker", "agreement_evidence": "Agreed.",
                    "evidence_segment_ids": ["S005", "S006"]}],
        open_proposals=[{"proposal": "Move the vector store to Postgres", "evidence_segment_ids": ["S004"]}],
        action_items=[{"id": "T1", "task": "Clean up the test data", "evidence_quote": "Someone needs to clean up the test data.",
                       "evidence_segment_ids": ["S008"]}],
        fidelity=Fidelity(numbers_preserved="2/2", negations_preserved="2/2", edits_accepted=0, edits_rejected=0,
                          disputed_words=0, items_downgraded_by_verifier=0, sentences_removed_by_verifier=0,
                          transcript_coverage_pct=50.0),
    )


class FakeSTT:
    """Returns the sample meeting as Whisper would, regardless of the audio."""

    name = "fake-whisper"
    loaded = True

    def transcribe(self, path: str, initial_prompt: str | None, progress=None, should_stop=None) -> dict[str, Any]:
        segments = []
        for s in sample_segments():
            segments.append({
                "start": s.start, "end": s.end, "text": s.text, "avg_logprob": -0.2, "no_speech_prob": 0.01,
                "compression_ratio": 1.2,
                "words": [{"w": w.w, "start": w.start, "end": w.end, "conf": w.conf} for w in s.words],
            })
        return {"language": "en", "language_probability": 0.98, "segments": segments}


class FakeLLM:
    """Scripted LM1/LM2 answers keyed by output schema."""

    def __init__(self, lm2: dict[str, Any] | None = None, fail_lm2: bool = False) -> None:
        self.lm2 = lm2
        self.fail_lm2 = fail_lm2
        self.calls: list[str] = []

    async def json_call(self, model, system, user, schema, max_retries, max_tokens=4096, job_id="-", on_retry=None):
        self.calls.append(schema.__name__)
        if schema is Vocabulary:
            return Vocabulary(domain="ML engineering", terms=[
                {"term": "CI", "heard_as": ["sea eye"], "evidence_segment_ids": ["S002"]},
                {"term": "CUDA", "heard_as": ["cuda"], "evidence_segment_ids": ["S002"]},
            ])
        if schema is LM1Output:
            edits = [
                {"segment_id": "S002", "original": "cuda", "replacement": "CUDA", "category": "acronym",
                 "rationale": "GPU term", "confidence": 0.95},
                {"segment_id": "S002", "original": "sea eye", "replacement": "CI", "category": "acronym",
                 "rationale": "pipeline", "confidence": 0.9},
                {"segment_id": "S003", "original": "fifteen", "replacement": "fifty", "category": "homophone",
                 "rationale": "?", "confidence": 0.8},
            ]
            return LM1Output(domain_guess="ML", edits=[e for e in edits if f'"{e["segment_id"]}"' in user])
        if schema in (LM2Output, LM2Record):
            if self.fail_lm2:
                from app.llm.json_repair import InvalidModelOutput

                raise InvalidModelOutput("bad json")
            return schema(**(self.lm2 or DEFAULT_LM2))
        raise AssertionError(f"unexpected schema {schema}")


DEFAULT_LM2: dict[str, Any] = {
    "scratchpad": "...",
    "summary": [
        {"text": "The team agreed to ship v2 without the reranker.", "evidence_segment_ids": ["S005", "S006"]},
        {"text": "Latency went from 15 to 50 milliseconds.", "evidence_segment_ids": ["S003"]},
        {"text": "The rollout is owned by Kumar.", "evidence_segment_ids": ["S005"]},
    ],
    "minutes": [{"topic": "Release", "points": [
        {"text": "The CUDA kernels wait for the CI pipeline.", "evidence_segment_ids": ["S002"]},
        {"text": "Costs fell by 30 percent.", "evidence_segment_ids": ["S003"]},
    ]}],
    "decisions": [
        {"decision": "Ship v2 without the reranker", "agreement_evidence": "Agreed.", "evidence_segment_ids": ["S005", "S006"]},
        {"decision": "Move the vector store to Postgres", "agreement_evidence": "Maybe we should move",
         "evidence_segment_ids": ["S004"]},
    ],
    "open_proposals": [],
    "action_items": [
        {"task": "Update the dashboards", "owner_evidence": {"segment_id": "S001", "exact_words": "Priya"},
         "deadline_evidence": {"segment_id": "S007", "exact_words": "Monday"},
         "evidence_quote": "I'll update the dashboards by Friday... no, Monday.", "evidence_segment_ids": ["S007"]},
        {"task": "Clean up the test data", "owner_evidence": None, "deadline_evidence": None,
         "evidence_quote": "Someone needs to clean up the test data.", "evidence_segment_ids": ["S008"]},
    ],
}
