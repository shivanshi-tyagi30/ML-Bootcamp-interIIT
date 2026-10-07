"""Unit tests for ECAPA-TDNN speaker identification and clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.pipeline.assemble import assemble_segments, assign_speakers
from app.pipeline.ecapa_diarize import EcapaSpeakerIdentifier, get_ecapa_identifier


def test_ecapa_embedding_shape_and_norm():
    """Verify ECAPA-TDNN outputs 192-dimensional unit-normalized embeddings."""
    identifier = get_ecapa_identifier()
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sine_wave = np.sin(2 * np.pi * 300 * t)

    emb = identifier.extract_embedding(sine_wave, sr)
    assert emb.shape == (192,)
    assert emb.dtype == np.float32
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4


def test_ecapa_clustering_distinct_speakers():
    """Verify clustering groups similar embeddings and separates distinct ones."""
    identifier = get_ecapa_identifier()

    # Create synthetic embeddings: 2 for Speaker A, 2 for Speaker B
    np.random.seed(42)
    vec_a = np.random.randn(192).astype(np.float32)
    vec_a = vec_a / np.linalg.norm(vec_a)

    vec_b = np.random.randn(192).astype(np.float32)
    vec_b = vec_b / np.linalg.norm(vec_b)

    # Slight perturbations for same speaker
    vec_a2 = vec_a + 0.05 * np.random.randn(192).astype(np.float32)
    vec_a2 = vec_a2 / np.linalg.norm(vec_a2)

    vec_b2 = vec_b + 0.05 * np.random.randn(192).astype(np.float32)
    vec_b2 = vec_b2 / np.linalg.norm(vec_b2)

    embeddings = [vec_a, vec_a2, vec_b, vec_b2]
    clusters = identifier.cluster_embeddings(embeddings, distance_threshold=0.35)

    assert len(clusters) == 4
    # First two should share a cluster, last two should share a different cluster
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_assign_speakers_fallback_without_audio():
    """Verify assign_speakers works gracefully even if audio is not provided."""
    words = [
        {"w": "Hi", "start": 0.0, "end": 0.5, "conf": 0.9, "seg": 0},
        {"w": "there", "start": 0.6, "end": 1.0, "conf": 0.9, "seg": 0},
        {"w": "Hello", "start": 2.0, "end": 2.5, "conf": 0.9, "seg": 1},  # >0.42s pause
    ]
    assign_speakers(words, wav_path=None)
    assert words[0]["speaker"] == "Speaker 1"
    assert words[1]["speaker"] == "Speaker 1"
    assert words[2]["speaker"] == "Speaker 2"


def test_assemble_segments_with_ecapa(fixtures_dir: Path):
    """Verify assemble_segments runs with real wav fixture and assigns speakers."""
    whisper_segs = [{
        "start": 0.0,
        "end": 3.0,
        "text": "Hello world this is a test.",
        "words": [
            {"w": "Hello", "start": 0.0, "end": 0.5, "conf": 0.9},
            {"w": "world", "start": 0.6, "end": 1.0, "conf": 0.9},
            {"w": "testing", "start": 2.0, "end": 2.8, "conf": 0.9},
        ],
    }]
    tone_wav = fixtures_dir / "tone.wav"
    segments = assemble_segments(whisper_segs, {}, turns=[], wav_path=tone_wav)
    assert len(segments) >= 1
    assert all(s.speaker is not None for s in segments)
    assert any(s.speaker.startswith("Speaker") for s in segments)
