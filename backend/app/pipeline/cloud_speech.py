"""Cloud speech for the hosted website: Whisper large-v3 on Groq for the words, AssemblyAI for who spoke when.

The server only coordinates (no PyTorch, no local models), so it fits a small free instance. Both services get
a compressed copy of the normalised recording (Opus, 24 kbit/s mono: ~11 MB per hour, under Groq's upload
limit). AssemblyAI is started during TRANSCRIBING and collected in DIARIZING, so the two run at the same time.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from app.config import Settings
from app.core.errors import JobCancelled

log = logging.getLogger(__name__)

# Groq returns the language's English name; the pipeline uses ISO codes ("en" for its English-only warnings).
LANGUAGE_CODES = {
    "english": "en", "hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te", "marathi": "mr",
    "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa", "urdu": "ur", "french": "fr",
    "german": "de", "spanish": "es", "portuguese": "pt", "italian": "it", "japanese": "ja", "chinese": "zh",
    "korean": "ko", "russian": "ru", "arabic": "ar", "dutch": "nl",
}
RETRY_STATUS = (429, 500, 502, 503, 504)
_encode_lock = threading.Lock()  # Groq and AssemblyAI ask for the same copy at the same time


def compressed_copy(wav: Path) -> Path:
    """Opus copy of the normalised WAV for upload (made once per job, next to the WAV)."""
    out = wav.with_suffix(".upload.ogg")
    with _encode_lock:
        if not out.exists() or out.stat().st_mtime < wav.stat().st_mtime:
            tmp = out.with_suffix(".tmp.ogg")
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-ac", "1", "-c:a", "libopus", "-b:a", "24k",
                   "-application", "voip", str(tmp)]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            tmp.replace(out)
    return out


def _request(method: str, url: str, *, attempts: int = 4, **kw: Any) -> httpx.Response:
    """HTTP call with a short wait-and-retry on rate limits and server errors."""
    for attempt in range(attempts):
        try:
            r = httpx.request(method, url, **kw)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in RETRY_STATUS and attempt < attempts - 1:
            wait = min(float(r.headers.get("retry-after") or 0) or 2.0 * (attempt + 1), 20.0)
            log.warning("%s %s -> HTTP %d; retrying in %.0f s", method, url.split("?")[0], r.status_code, wait)
            time.sleep(wait)
            continue
        return r
    return r


def _check(r: httpx.Response, service: str, key_name: str) -> None:
    """Raise a readable error for a failed call."""
    if r.status_code in (401, 403):
        raise RuntimeError(f"{service} rejected the API key ({key_name}). Check it in the server settings.")
    if r.status_code == 429:
        raise RuntimeError(f"{service}'s free-tier limit was reached. Wait a minute and press Retry.")
    if r.status_code >= 400:
        raise RuntimeError(f"{service} returned HTTP {r.status_code}: {r.text[:300]}")


class GroqWhisperSTT:
    """Whisper large-v3 through Groq's OpenAI-compatible transcription API (same interface as WhisperSTT)."""

    loaded = True  # nothing to load

    def __init__(self, settings: Settings) -> None:
        """Remember the key and model."""
        self.settings = settings
        self.name = f"{settings.GROQ_STT_MODEL} (Groq cloud)"

    def transcribe(
        self, path: str, initial_prompt: str | None, progress: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Segments with word timestamps, in the same shape faster-whisper produces."""
        s = self.settings
        if not s.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set on the server.")
        audio = compressed_copy(Path(path))
        if progress:
            progress(0.1)
        data: dict[str, Any] = {"model": s.GROQ_STT_MODEL, "response_format": "verbose_json", "temperature": "0"}
        if initial_prompt:
            data["prompt"] = initial_prompt[:800]  # Groq allows ~224 tokens of prompt
        with audio.open("rb") as f:
            r = _request(
                "POST", f"{s.GROQ_BASE_URL.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {s.GROQ_API_KEY}"},
                data={**data, "timestamp_granularities[]": ["word", "segment"]},
                files={"file": (audio.name, f, "audio/ogg")}, timeout=s.CLOUD_STT_TIMEOUT_SEC,
            )
        _check(r, "Groq", "GROQ_API_KEY")
        if should_stop and should_stop():
            raise JobCancelled()
        if progress:
            progress(1.0)
        return groq_to_segments(r.json())


def groq_to_segments(body: dict[str, Any]) -> dict[str, Any]:
    """Groq verbose_json -> {language, language_probability, segments:[{..., words:[{w,start,end,conf}]}]}.

    Groq gives a confidence per segment (avg_logprob), not per word: each word gets its segment's
    probability, so a badly heard stretch is still marked for the checks downstream.
    """
    words = [w for w in body.get("words") or [] if str(w.get("word", "")).strip()]
    segments = []
    i = 0
    for seg in body.get("segments") or []:
        start, end = float(seg.get("start", 0)), float(seg.get("end", 0))
        conf = round(math.exp(min(0.0, float(seg.get("avg_logprob", 0.0)))), 4)
        mine = []
        while i < len(words) and float(words[i]["start"]) < end - 1e-3:
            w = words[i]
            mine.append({"w": str(w["word"]).strip(), "start": float(w["start"]), "end": float(w["end"]), "conf": conf})
            i += 1
        if not mine:  # no word timings for this segment: spread its words evenly
            toks = str(seg.get("text", "")).split()
            step = (end - start) / max(1, len(toks))
            mine = [{"w": t, "start": start + k * step, "end": start + (k + 1) * step, "conf": conf}
                    for k, t in enumerate(toks)]
        segments.append({
            "start": start, "end": end, "text": str(seg.get("text", "")).strip(),
            "avg_logprob": float(seg.get("avg_logprob", 0.0)), "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
            "compression_ratio": float(seg.get("compression_ratio", 1.0)), "words": mine,
        })
    if segments and i < len(words):  # words after the last segment's end belong to it
        segments[-1]["words"] += [{"w": str(w["word"]).strip(), "start": float(w["start"]), "end": float(w["end"]),
                                   "conf": segments[-1]["words"][-1]["conf"] if segments[-1]["words"] else 1.0}
                                  for w in words[i:]]
    lang = str(body.get("language") or "en").lower()
    return {"language": LANGUAGE_CODES.get(lang, lang[:2]), "language_probability": 1.0, "segments": segments}


class AssemblyAIDiarizer:
    """Speaker turns ("who spoke when") from AssemblyAI; the words still come from Whisper."""

    name = "AssemblyAI speaker labels (cloud)"
    loaded = True

    def __init__(self, settings: Settings) -> None:
        """Remember the key."""
        self.settings = settings
        self.base = settings.ASSEMBLYAI_BASE_URL.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.settings.ASSEMBLYAI_API_KEY:
            raise RuntimeError("ASSEMBLYAI_API_KEY is not set on the server.")
        return {"authorization": self.settings.ASSEMBLYAI_API_KEY}

    def submit(self, wav: Path, speakers: int = 0) -> str:
        """Upload the recording and start a speaker-labelled transcript; returns its id."""
        audio = compressed_copy(wav)
        with audio.open("rb") as f:
            r = _request("POST", f"{self.base}/v2/upload", headers=self._headers(), content=f.read(), timeout=300)
        _check(r, "AssemblyAI", "ASSEMBLYAI_API_KEY")
        body: dict[str, Any] = {"audio_url": r.json()["upload_url"], "speaker_labels": True}
        if self.settings.ASSEMBLYAI_SPEECH_MODEL:
            body["speech_model"] = self.settings.ASSEMBLYAI_SPEECH_MODEL
        if speakers:
            body["speakers_expected"] = speakers
        r = _request("POST", f"{self.base}/v2/transcript", headers=self._headers(), json=body, timeout=60)
        _check(r, "AssemblyAI", "ASSEMBLYAI_API_KEY")
        return str(r.json()["id"])

    def collect(self, transcript_id: str, should_stop: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
        """Wait for the transcript and return turns [{start, end, label}] in seconds."""
        deadline = time.monotonic() + self.settings.CLOUD_STT_TIMEOUT_SEC * 4
        while True:
            if should_stop and should_stop():
                raise JobCancelled()
            r = _request("GET", f"{self.base}/v2/transcript/{transcript_id}", headers=self._headers(), timeout=60)
            _check(r, "AssemblyAI", "ASSEMBLYAI_API_KEY")
            body = r.json()
            if body.get("status") == "completed":
                return [{"start": u["start"] / 1000, "end": u["end"] / 1000, "label": str(u.get("speaker", "A"))}
                        for u in body.get("utterances") or []]
            if body.get("status") == "error":
                raise RuntimeError(f"AssemblyAI could not label speakers: {body.get('error', 'unknown error')}")
            if time.monotonic() > deadline:
                raise RuntimeError("AssemblyAI took too long to label speakers.")
            time.sleep(3)

    def run(self, path: str) -> list[dict[str, Any]]:
        """Submit and wait (used when nothing was started earlier)."""
        return self.collect(self.submit(Path(path), self.settings.DIARIZATION_NUM_SPEAKERS))
