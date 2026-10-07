"""Speaker labels from ECAPA-TDNN voice embeddings (SpeechBrain), clustered per meeting.

How it works:
1. Whisper's words are grouped into speech bursts (split at pauses and every MAX_BURST_SEC).
2. Each burst gets a 192-d ECAPA embedding (a voice fingerprint) on the CPU.
3. Bursts of at least MIN_CLUSTER_SEC are clustered (agglomerative, cosine distance, average linkage).
   Short bursts ("Yes.", "Agreed.") give noisy embeddings, so they are not allowed to start a speaker;
   they join the closest cluster afterwards.
4. Clusters with very little speech are merged into their nearest neighbour, so noise does not
   become "Speaker 5".

No Hugging Face token is needed; `python -m app.prefetch` downloads the model (~80 MB) at deploy time.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_DISTANCE_THRESHOLD = 0.6  # cosine distance; same voice is usually < 0.5 on 2-3 s clips
BURST_GAP_SEC = 0.42  # a pause this long may be a turn change
MAX_BURST_SEC = 8.0  # long monologues are cut so a quick reply without a pause can still be seen
MIN_CLUSTER_SEC = 1.0  # shorter bursts never create a speaker on their own
MIN_SPEAKER_SEC = 3.0  # clusters with less speech than this are merged into the nearest one
MIN_EMBED_SEC = 0.4  # ECAPA needs a little audio; shorter clips are repeated to this length


def make_bursts(words: list[dict[str, Any]], gap: float = BURST_GAP_SEC,
                max_sec: float = MAX_BURST_SEC) -> list[tuple[float, float]]:
    """(start, end) of speech bursts from Whisper words."""
    out: list[tuple[float, float]] = []
    start = end = None
    for w in words:
        if start is None:
            start, end = w["start"], w["end"]
        elif w["start"] - end >= gap or w["end"] - start > max_sec:
            out.append((float(start), float(end)))
            start, end = w["start"], w["end"]
        else:
            end = max(end, w["end"])
    if start is not None:
        out.append((float(start), float(end)))
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def cluster_bursts(
    embeddings: np.ndarray, durations: list[float], threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    n_speakers: int = 0, min_cluster_sec: float = MIN_CLUSTER_SEC, min_speaker_sec: float = MIN_SPEAKER_SEC,
) -> list[int]:
    """Speaker index per burst (0, 1, 0, 2...) in order of first appearance.

    `n_speakers` > 0 fixes the number of speakers (most accurate when known); otherwise the
    cosine-distance `threshold` decides.
    """
    n = len(durations)
    if n == 0:
        return []
    embs = _unit(np.asarray(embeddings, dtype=np.float32))
    dur = np.asarray(durations, dtype=np.float32)
    anchors = np.where(dur >= min_cluster_sec)[0]
    if len(anchors) == 0:  # all bursts short: use them all
        anchors = np.arange(n)

    if len(anchors) == 1:
        anchor_labels = np.zeros(1, dtype=int)
    else:
        from sklearn.cluster import AgglomerativeClustering

        k = min(n_speakers, len(anchors)) if n_speakers > 0 else None
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average", distance_threshold=None if k else threshold,
        )
        anchor_labels = model.fit_predict(embs[anchors])

    def centroids(lbls: np.ndarray, idx: np.ndarray) -> dict[int, np.ndarray]:
        return {c: _unit(embs[idx[lbls == c]].mean(axis=0)) for c in np.unique(lbls)}

    # Merge clusters with too little speech into the closest bigger one (unless the count is fixed).
    if not n_speakers:
        while True:
            cents = centroids(anchor_labels, anchors)
            if len(cents) < 2:
                break
            speech = {c: float(dur[anchors[anchor_labels == c]].sum()) for c in cents}
            small = min(speech, key=speech.get)
            if speech[small] >= min_speaker_sec:
                break
            others = [c for c in cents if c != small]
            target = max(others, key=lambda c: float(cents[small] @ cents[c]))
            anchor_labels = np.where(anchor_labels == small, target, anchor_labels)

    # Every burst (short ones included) takes the closest speaker centroid; anchors keep their cluster.
    cents = centroids(anchor_labels, anchors)
    keys = list(cents)
    mat = np.stack([cents[c] for c in keys])
    labels = [keys[int(np.argmax(mat @ embs[i]))] for i in range(n)]
    for a, lbl in zip(anchors, anchor_labels):
        labels[int(a)] = int(lbl)

    order: dict[int, int] = {}
    return [order.setdefault(lbl, len(order)) for lbl in labels]


class EcapaSpeakerIdentifier:
    """ECAPA-TDNN embedder, loaded on first use."""

    def __init__(self, model_source: str = DEFAULT_ECAPA_MODEL, savedir: str | Path | None = None) -> None:
        """Remember where to load the model from; nothing is loaded yet."""
        self.model_source = model_source
        self.savedir = Path(savedir) if savedir else Path("data") / "models" / "ecapa"
        self._classifier: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the model is in memory."""
        return self._classifier is not None

    def _load(self) -> Any:
        """Load SpeechBrain's EncoderClassifier once (CPU unless CUDA is available)."""
        with self._lock:
            if self._classifier is None:
                import torch
                from speechbrain.inference.speaker import EncoderClassifier

                device = "cuda" if torch.cuda.is_available() else "cpu"
                kwargs: dict[str, Any] = {"source": self.model_source, "savedir": str(self.savedir),
                                          "run_opts": {"device": device}}
                try:  # copy files instead of symlinking: symlinks need admin rights on Windows
                    from speechbrain.utils.fetching import LocalStrategy

                    kwargs["local_strategy"] = LocalStrategy.COPY
                except ImportError:
                    pass
                log.info("loading ECAPA-TDNN %s on %s", self.model_source, device)
                self._classifier = EncoderClassifier.from_hparams(**kwargs)
                self._device = device
        return self._classifier

    def embed(self, clip: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Unit-length 192-d embedding of a mono clip."""
        import torch

        clf = self._load()
        clip = np.asarray(clip, dtype=np.float32)
        if clip.ndim > 1:
            clip = clip.mean(axis=-1)
        need = int(MIN_EMBED_SEC * sr)
        if len(clip) == 0:
            return np.zeros(192, dtype=np.float32)
        if len(clip) < need:
            clip = np.tile(clip, int(np.ceil(need / len(clip))))[:need]
        with torch.no_grad():
            emb = clf.encode_batch(torch.from_numpy(clip).unsqueeze(0).to(self._device))
        return _unit(emb.squeeze().cpu().numpy().astype(np.float32))

    def label_bursts(
        self, wav_path: str | Path, bursts: list[tuple[float, float]], threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        n_speakers: int = 0, should_stop: Any = None,
    ) -> list[int]:
        """Speaker index per burst."""
        import soundfile as sf

        if not bursts:
            return []
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        embs = []
        for a, b in bursts:
            if should_stop and should_stop():
                from app.core.errors import JobCancelled

                raise JobCancelled()
            embs.append(self.embed(audio[int(a * sr) : max(int(a * sr) + 1, int(b * sr))], sr))
        return cluster_bursts(np.stack(embs), [b - a for a, b in bursts], threshold, n_speakers)


_default: EcapaSpeakerIdentifier | None = None
_default_lock = threading.Lock()


def get_ecapa_identifier(model_source: str = DEFAULT_ECAPA_MODEL, savedir: str | Path | None = None,
                         ) -> EcapaSpeakerIdentifier:
    """Process-wide identifier (raises ImportError on first use if speechbrain is missing)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = EcapaSpeakerIdentifier(model_source, savedir)
    return _default
