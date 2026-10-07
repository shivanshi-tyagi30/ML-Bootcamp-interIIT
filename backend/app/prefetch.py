"""Download every model once, at deploy time, so no user ever waits for a download.

    python -m app.prefetch

Uses the same settings (.env) as the server: the Whisper model the server will pick for this machine, the
speaker model, Silero VAD, and both Ollama models. Safe to re-run; anything already present is skipped.
Exits with status 1 if something required could not be fetched.
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from app.config import Settings, get_settings


def _step(name: str, fn) -> bool:
    """Run one fetch step and print the outcome."""
    print(f"- {name} ...", flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {e!r}", flush=True)
        return False
    print("  ok", flush=True)
    return True


def fetch_whisper(s: Settings) -> None:
    """Download the Whisper model the server will use on this machine."""
    from faster_whisper import download_model

    download_model(str(s.whisper_runtime()["model"]))


def fetch_silero() -> None:
    """Silero VAD ships inside its pip package; loading it once checks it works."""
    from silero_vad import load_silero_vad

    load_silero_vad()


def fetch_ecapa(s: Settings) -> None:
    """Download the ECAPA-TDNN speaker model into DATA_DIR/models/ecapa."""
    from app.pipeline.ecapa_diarize import get_ecapa_identifier

    get_ecapa_identifier(s.ECAPA_MODEL, s.DATA_DIR / "models" / "ecapa")._load()


def fetch_pyannote(s: Settings) -> None:
    """Download the pyannote pipeline (only when it is the configured backend)."""
    from pyannote.audio import Pipeline

    from app.pipeline.diarize import pipeline_id

    try:
        Pipeline.from_pretrained(pipeline_id(), token=s.HF_TOKEN)
    except TypeError:
        Pipeline.from_pretrained(pipeline_id(), use_auth_token=s.HF_TOKEN)


async def pull_ollama(s: Settings, model: str) -> None:
    """`ollama pull <model>` through the API (no-op if already downloaded)."""
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{s.ollama_host}/api/pull", json={"model": model, "stream": False})
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")


def main() -> int:
    """Fetch everything; return the process exit code."""
    s = get_settings()
    ok = True
    ok &= _step(f"Whisper {s.whisper_runtime()['model']}", lambda: fetch_whisper(s))
    ok &= _step("Silero VAD", fetch_silero)
    if s.DIARIZATION_ENABLED:
        if s.DIARIZATION_BACKEND == "pyannote" or (s.DIARIZATION_BACKEND == "auto" and s.HF_TOKEN):
            ok &= _step("pyannote speaker model", lambda: fetch_pyannote(s))
        else:
            ok &= _step(f"speaker model {s.ECAPA_MODEL}", lambda: fetch_ecapa(s))
    if s.LLM_BACKEND == "ollama":
        for m in dict.fromkeys([s.LM1_MODEL, s.LM2_MODEL]):
            ok &= _step(f"Ollama {m}", lambda m=m: asyncio.run(pull_ollama(s, m)))
    print("All models are ready." if ok else "Some models are missing; see FAILED above.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
