"""VALIDATING / NORMALIZING / SPEECH_CHECK on generated files."""

from __future__ import annotations

import pytest

from app.core.errors import PipelineError
from app.pipeline.normalize import probe, to_wav16k
from app.pipeline.vad import detect_speech
from app.pipeline.validate import validate_file


def test_empty_file(fixtures_dir, settings):
    with pytest.raises(PipelineError) as e:
        validate_file(fixtures_dir / "empty.mp3", "empty.mp3", settings)
    assert e.value.code == "E_EMPTY_FILE"


def test_fake_mp3(fixtures_dir, settings):
    with pytest.raises(PipelineError) as e:
        validate_file(fixtures_dir / "fake.mp3", "fake.mp3", settings)
        probe(fixtures_dir / "fake.mp3", settings)
    assert e.value.code in ("E_UNSUPPORTED_FORMAT", "E_UNREADABLE")


def test_pdf_extension(fixtures_dir, settings):
    with pytest.raises(PipelineError) as e:
        validate_file(fixtures_dir / "notes.pdf", "notes.pdf", settings)
    assert e.value.code == "E_UNSUPPORTED_FORMAT"


def test_silence_has_no_speech(fixtures_dir, settings, tmp_path):
    info = validate_file(fixtures_dir / "silence.wav", "silence.wav", settings)
    assert info["ext"] == "wav"
    probe(fixtures_dir / "silence.wav", settings)
    wav = tmp_path / "a.wav"
    to_wav16k(fixtures_dir / "silence.wav", wav)
    with pytest.raises(PipelineError) as e:
        detect_speech(wav, settings)
    assert e.value.code == "E_NO_SPEECH"


def test_too_long(fixtures_dir, settings):
    settings.MAX_DURATION_MIN = 0
    with pytest.raises(PipelineError) as e:
        probe(fixtures_dir / "tone.wav", settings)
    assert e.value.code == "E_TOO_LONG"
