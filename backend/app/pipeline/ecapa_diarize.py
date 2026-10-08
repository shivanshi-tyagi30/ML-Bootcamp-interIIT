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
DEFAULT_DISTANCE_THRESHOLD = 0.55  # cosine distance, used only when there are very few long bursts
BURST_GAP_SEC = 0.28  # a pause this long indicates a potential turn change
MAX_BURST_SEC = 8.0  # long monologues are cut so a quick reply without a pause can still be seen
MIN_CLUSTER_SEC = 1.0  # shorter bursts give unreliable fingerprints: they join a speaker, never start one
MIN_SPEAKER_SEC = 3.0  # clusters with less speech than this are merged into the nearest one
SHORT_BURST_SEC = 1.5  # bursts shorter than this get a little surrounding audio
SHORT_PAD_SEC = 0.25
MIN_EMBED_SEC = 0.4  # ECAPA needs a little audio; shorter clips are repeated to this length


def make_bursts(words: list[dict[str, Any]], gap: float = BURST_GAP_SEC,
                max_sec: float = MAX_BURST_SEC) -> list[tuple[float, float]]:
    """(start, end) of speech bursts from Whisper words."""
    out: list[tuple[float, float]] = []
    start = end = None
    prev_seg = None
    for w in words:
        seg = w.get("seg")
        seg_split = (prev_seg is not None and seg is not None and seg != prev_seg
                     and end is not None and (w["start"] - end >= 0.18))
        if start is None:
            start, end = w["start"], w["end"]
        elif w["start"] - end >= gap or w["end"] - start > max_sec or seg_split:
            out.append((float(start), float(end)))
            start, end = w["start"], w["end"]
        else:
            end = max(end, w["end"])
        prev_seg = seg
    if start is not None:
        out.append((float(start), float(end)))
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


MAX_SPEAKERS = 8
SPECTRAL_MIN_LINES = 12  # below this many reliable lines the eigengap is unstable; k-means + separation is used
# How distinct two groups of lines are: (within-group likeness - across-group likeness) / (1 - across).
# Simulated meetings: real different speakers score ~0.7-0.9 even with very similar voices on one phone;
# one person split in two scores below 0.1. Below this value the two groups are merged.
MIN_SEPARATION = 0.2
REFINE_ROUNDS = 5


def separation(x: np.ndarray, y: np.ndarray) -> float:
    """Separation of two groups of unit fingerprints (0 = same distribution, 1 = fully distinct)."""
    def within(g: np.ndarray) -> float | None:
        return float((g @ g.T)[np.triu_indices(len(g), 1)].mean()) if len(g) >= 2 else None

    wx, wy = within(x), within(y)
    ws = [v for v in (wx, wy) if v is not None]
    if not ws:
        return 0.0  # two single lines cannot show they are different people: let them merge
    across = float((x @ y.T).mean())
    return (float(np.mean(ws)) - across) / max(1.0 - across, 1e-6)


def _small_labels(centered: np.ndarray, raw: np.ndarray, n_speakers: int = 0) -> np.ndarray:
    """Few lines: try 1, 2, 3... speakers with k-means and keep the largest count whose groups are all
    clearly distinct (see `separation`)."""
    from sklearn.cluster import KMeans

    n = len(centered)
    if n_speakers > 0:
        k = min(n_speakers, n)
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(centered) if k > 1 else np.zeros(n, int)
    best = np.zeros(n, dtype=int)
    for k in range(2, min(MAX_SPEAKERS, n) + 1):
        lbl = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(centered)
        groups = [raw[lbl == c] for c in range(k)]
        if all(separation(groups[i], groups[j]) >= MIN_SEPARATION for i in range(k) for j in range(i + 1, k)):
            best = lbl
        else:
            break
    return best


def _spectral_labels(x: np.ndarray, n_speakers: int = 0, max_speakers: int = MAX_SPEAKERS) -> np.ndarray:
    """Spectral clustering on cosine affinities; the number of speakers comes from the largest eigengap."""
    n = len(x)
    sim = np.clip(x @ x.T, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    # Keep each burst's strongest neighbours only (pruning makes the graph robust to noisy pairs).
    keep = int(min(n - 1, max(3, round(0.3 * n))))
    pruned = np.zeros_like(sim)
    for i in range(n):
        top = np.argsort(sim[i])[-keep:]
        pruned[i, top] = sim[i, top]
    aff = (pruned + pruned.T) / 2
    lap = np.diag(aff.sum(axis=1)) - aff
    vals, vecs = np.linalg.eigh(lap)
    if n_speakers > 0:
        k = min(n_speakers, n)
    else:
        upper = min(max_speakers, n - 1)
        gaps = np.diff(vals[: upper + 1])
        k = int(np.argmax(gaps)) + 1 if len(gaps) else 1
    if k <= 1:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import KMeans

    emb = _unit(vecs[:, :k])
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)


def cluster_bursts(
    embeddings: np.ndarray, durations: list[float], threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    n_speakers: int = 0, min_cluster_sec: float = MIN_CLUSTER_SEC, min_speaker_sec: float = MIN_SPEAKER_SEC,
) -> list[int]:
    """Speaker index per burst (0, 1, 0, 2...) in order of first appearance.

    1. Within-recording normalisation: the duration-weighted mean voice of the recording is removed, so
       what every speaker shares (same mic, room, phone codec) no longer makes two people look alike.
    2. Bursts of at least `min_cluster_sec` (reliable fingerprints) are grouped by spectral clustering; the
       number of speakers comes from the eigengap unless `n_speakers` is given. Very few long bursts fall
       back to agglomerative clustering with `threshold`.
    3. Groups whose raw voice profiles are near-identical are one person and are merged; groups with too
       little speech are merged into the closest one.
    4. Refinement: every burst (short ones too) joins its closest voice profile, profiles are recomputed
       from all their speech (weighted by duration), a few rounds. Short bursts never start a speaker.
    """
    n = len(durations)
    if n == 0:
        return []
    raw = _unit(np.asarray(embeddings, dtype=np.float32))
    dur = np.asarray(durations, dtype=np.float32)
    anchors = np.where(dur >= min_cluster_sec)[0]
    if len(anchors) == 0:  # all bursts short: use them all
        anchors = np.arange(n)
    w = np.minimum(dur, 8.0)[:, None]
    mean = (raw[anchors] * w[anchors]).sum(axis=0) / max(float(w[anchors].sum()), 1e-6)
    embs = _unit(raw - mean)

    if len(anchors) < 6:  # a short recording: let medium-length lines help define the voices
        anchors = np.where(dur >= min(min_cluster_sec, 0.5))[0]
        if len(anchors) == 0:
            anchors = np.arange(n)
    if len(anchors) == 1:
        anchor_labels = np.zeros(1, dtype=int)
    elif len(anchors) < SPECTRAL_MIN_LINES:
        anchor_labels = _small_labels(embs[anchors], raw[anchors], n_speakers)
    else:
        anchor_labels = _spectral_labels(embs[anchors], n_speakers)

    def profiles(lbls: np.ndarray, idx: np.ndarray, space: np.ndarray) -> dict[int, np.ndarray]:
        return {int(c): _unit((space[idx[lbls == c]] * w[idx[lbls == c]]).sum(axis=0)) for c in np.unique(lbls)}

    if not n_speakers:
        # One person split in two: lines in the two groups are about as alike across groups as within them.
        while True:
            keys = [int(c) for c in np.unique(anchor_labels)]
            pairs = [(separation(raw[anchors[anchor_labels == a]], raw[anchors[anchor_labels == b]]), a, b)
                     for i, a in enumerate(keys) for b in keys[i + 1:]]
            if not pairs:
                break
            sep, a, b = min(pairs)
            if sep >= MIN_SEPARATION:
                break
            anchor_labels = np.where(anchor_labels == b, a, anchor_labels)
        # Noise or a cough is not a speaker: merge groups with too little speech.
        total = float(dur[anchors].sum())
        while True:
            prof = profiles(anchor_labels, anchors, embs)
            if len(prof) < 2:
                break
            speech = {c: float(dur[anchors[anchor_labels == c]].sum()) for c in prof}
            small = min(speech, key=speech.get)
            if speech[small] >= min(min_speaker_sec, max(0.5, total * 0.08)):
                break
            target = max((c for c in prof if c != small), key=lambda c: float(prof[small] @ prof[c]))
            anchor_labels = np.where(anchor_labels == small, target, anchor_labels)

    # Refinement over all bursts; the set of speakers is fixed by the anchors.
    labels = np.full(n, -1, dtype=int)
    labels[anchors] = anchor_labels
    prof = profiles(anchor_labels, anchors, embs)
    keys = list(prof)
    for _ in range(REFINE_ROUNDS):
        mat = np.stack([prof[c] for c in keys])
        new = np.array([keys[int(np.argmax(mat @ embs[i]))] for i in range(n)])
        if np.array_equal(new, labels):
            break
        labels = new
        prof = {c: (profiles(labels[labels == c], np.where(labels == c)[0], embs)[c] if (labels == c).any()
                    else prof[c]) for c in keys}

    order: dict[int, int] = {}
    return [order.setdefault(int(lbl), len(order)) for lbl in labels]


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
        for i, (a, b) in enumerate(bursts):
            if should_stop and should_stop():
                from app.core.errors import JobCancelled

                raise JobCancelled()
            if b - a < SHORT_BURST_SEC:
                # A short reply carries little voice: add a little of the surrounding audio (never reaching
                # into the neighbouring bursts, which may be another person).
                lo = bursts[i - 1][1] if i > 0 else 0.0
                hi = bursts[i + 1][0] if i + 1 < len(bursts) else len(audio) / sr
                a, b = max(a - SHORT_PAD_SEC, lo, 0.0), min(b + SHORT_PAD_SEC, hi)
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
