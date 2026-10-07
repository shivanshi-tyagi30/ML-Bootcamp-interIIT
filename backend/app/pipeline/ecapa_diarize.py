"""ECAPA-TDNN speaker embedding extraction and clustering.

Uses SpeechBrain's ECAPA-TDNN model (spkrec-ecapa-voxceleb) to extract 192-dimensional
speaker embeddings for speech bursts/segments and cluster them with Agglomerative Clustering.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_DISTANCE_THRESHOLD = 0.35  # Cosine distance (1 - cos_sim); ~0.65 similarity
MIN_AUDIO_DURATION_SEC = 0.4       # Minimum audio duration for stable embeddings


class EcapaSpeakerIdentifier:
    """Extracts speaker embeddings using ECAPA-TDNN and clusters them into distinct speakers."""

    def __init__(self, model_source: str = DEFAULT_ECAPA_MODEL, device: str = "auto") -> None:
        """Initialize the identifier (lazy loading)."""
        self.model_source = model_source
        self.device_str = device
        self._classifier: Any = None
        self._lock = threading.Lock()
        self._device: str | None = None

    @property
    def loaded(self) -> bool:
        """Whether the model is already loaded in memory."""
        return self._classifier is not None

    def _ensure_loaded(self) -> Any:
        """Load the SpeechBrain ECAPA-TDNN model thread-safely."""
        if self._classifier is not None:
            return self._classifier

        with self._lock:
            if self._classifier is None:
                import torch
                from speechbrain.inference.speaker import EncoderClassifier

                if self.device_str == "auto":
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self._device = self.device_str

                log.info("Loading ECAPA-TDNN model '%s' on %s", self.model_source, self._device)
                self._classifier = EncoderClassifier.from_hparams(
                    source=self.model_source,
                    run_opts={"device": self._device},
                )
        return self._classifier

    def extract_embedding(self, wav_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 192-dimensional normalized speaker embedding from audio waveform."""
        import torch

        classifier = self._ensure_loaded()

        # Ensure single channel 1D float32
        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=-1)
        wav_data = wav_data.astype(np.float32)

        # Pad if shorter than minimum required length
        min_samples = int(MIN_AUDIO_DURATION_SEC * sample_rate)
        if len(wav_data) < min_samples:
            if len(wav_data) == 0:
                return np.zeros(192, dtype=np.float32)
            reps = int(np.ceil(min_samples / len(wav_data)))
            wav_data = np.tile(wav_data, reps)[:min_samples]

        tensor = torch.from_numpy(wav_data).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = classifier.encode_batch(tensor)

        emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb_np)
        if norm > 1e-6:
            emb_np = emb_np / norm
        return emb_np

    def cluster_embeddings(
        self,
        embeddings: list[np.ndarray],
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    ) -> list[int]:
        """Cluster speaker embeddings using Agglomerative Clustering with cosine distance.
        
        Returns cluster indices (0, 1, 0, 2...) corresponding to each input embedding.
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            return []
        if n_samples == 1:
            return [0]

        from sklearn.cluster import AgglomerativeClustering

        embs = np.stack(embeddings)
        clustering = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
        labels = clustering.fit_predict(embs)
        return labels.tolist()

    def identify_speakers(
        self,
        audio_path: str | Path,
        bursts: list[list[dict[str, Any]]],
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    ) -> list[str]:
        """Extract embeddings for each speech burst from audio and assign 'Speaker 1', 'Speaker 2'... labels.
        
        Preserves speaker consistency when a speaker re-enters the conversation later.
        """
        import soundfile as sf

        if not bursts:
            return []

        path = Path(audio_path)
        if not path.exists():
            log.warning("Audio file %s does not exist for ECAPA-TDNN speaker identification", path)
            return []

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        embeddings: list[np.ndarray] = []
        for b in bursts:
            start_sec = max(0.0, float(b[0]["start"]))
            end_sec = max(start_sec + 0.1, float(b[-1]["end"]))

            start_sample = int(start_sec * sr)
            end_sample = min(len(audio), int(end_sec * sr))
            slice_data = audio[start_sample:end_sample]

            emb = self.extract_embedding(slice_data, sr)
            embeddings.append(emb)

        cluster_labels = self.cluster_embeddings(embeddings, distance_threshold=distance_threshold)

        # Map cluster labels to display names 'Speaker 1', 'Speaker 2' in order of appearance
        speaker_map: dict[int, str] = {}
        result_speakers: list[str] = []
        for cluster_id in cluster_labels:
            if cluster_id not in speaker_map:
                speaker_map[cluster_id] = f"Speaker {len(speaker_map) + 1}"
            result_speakers.append(speaker_map[cluster_id])

        return result_speakers


_default_identifier: EcapaSpeakerIdentifier | None = None
_identifier_lock = threading.Lock()


def get_ecapa_identifier(model_source: str = DEFAULT_ECAPA_MODEL, device: str = "auto") -> EcapaSpeakerIdentifier:
    """Get the process-wide ECAPA-TDNN speaker identifier singleton."""
    global _default_identifier
    if _default_identifier is None:
        with _identifier_lock:
            if _default_identifier is None:
                _default_identifier = EcapaSpeakerIdentifier(model_source=model_source, device=device)
    return _default_identifier
