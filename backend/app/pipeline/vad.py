"""SPEECH_CHECK: is there enough speech to transcribe? (spec Section 10.3)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from app.config import Settings
from app.core.errors import PipelineError
from app.core.stages import Stage

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

log = logging.getLogger(__name__)

# Frames quieter than this are certainly not speech (about -45 dBFS).
ENERGY_FLOOR = 10 ** (-45 / 20)
FRAME_SEC = 0.03

SpeechFn = Callable[[np.ndarray, int], list[dict[str, float]]]
_silero: Any = None


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 mono samples."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def energy_regions(samples: np.ndarray, sr: int) -> list[dict[str, float]]:
    """Regions whose frame energy is above the floor. Cheap gate that catches silence."""
    n = int(sr * FRAME_SEC)
    if n == 0 or len(samples) < n:
        return []
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    loud = np.sqrt((frames**2).mean(axis=1)) > ENERGY_FLOOR
    regions: list[dict[str, float]] = []
    start = None
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append({"start": start * FRAME_SEC, "end": i * FRAME_SEC})
            start = None
    if start is not None:
        regions.append({"start": start * FRAME_SEC, "end": len(loud) * FRAME_SEC})
    return regions


def energy_regions_file(path: Path) -> list[dict[str, float]]:
    """energy_regions() reading the WAV in blocks: memory stays small even for a two-hour recording."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        n = int(sr * FRAME_SEC)
        loud: list[bool] = []
        for block in f.blocks(blocksize=n * 2000, dtype="float32", always_2d=True):
            mono = block.mean(axis=1)
            frames = mono[: len(mono) // n * n].reshape(-1, n)
            loud.extend((np.sqrt((frames**2).mean(axis=1)) > ENERGY_FLOOR).tolist())
    regions: list[dict[str, float]] = []
    start = None
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append({"start": start * FRAME_SEC, "end": i * FRAME_SEC})
            start = None
    if start is not None:
        regions.append({"start": start * FRAME_SEC, "end": len(loud) * FRAME_SEC})
    return regions


def silero_available() -> bool:
    """Whether Silero VAD (and PyTorch) are installed; the hosted website runs without them."""
    import importlib.util

    return importlib.util.find_spec("silero_vad") is not None and importlib.util.find_spec("torch") is not None


def silero_regions(samples: np.ndarray, sr: int) -> list[dict[str, float]]:
    """Silero VAD speech timestamps in seconds (threshold 0.5, 250 ms speech, 500 ms silence)."""
    global _silero
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _silero is None:
        _silero = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(samples), _silero, sampling_rate=sr, threshold=0.5,
        min_speech_duration_ms=250, min_silence_duration_ms=500, return_seconds=True,
    )
    return [{"start": float(t["start"]), "end": float(t["end"])} for t in ts]


def detect_speech(wav: Path, settings: Settings, vad_fn: SpeechFn | None = None) -> dict[str, Any]:
    """Speech regions and total seconds; raises E_NO_SPEECH below MIN_SPEECH_SEC."""
    total = lambda rs: round(sum(r["end"] - r["start"] for r in rs), 3)  # noqa: E731
    loud = energy_regions_file(wav)
    if total(loud) < settings.MIN_SPEECH_SEC:
        raise PipelineError("E_NO_SPEECH", Stage.SPEECH_CHECK, f"energy {total(loud)}s")
    method = "silero"
    if vad_fn is None and not silero_available():  # no full copy of the audio in memory
        regions, method = loud, "energy"
    else:
        samples, sr = load_wav(wav)
        try:
            regions = (vad_fn or silero_regions)(samples, sr)
        except ImportError:
            log.warning("silero-vad unavailable; using the energy gate only")
            regions, method = loud, "energy"
    speech = total(regions)
    if speech < settings.MIN_SPEECH_SEC:
        raise PipelineError("E_NO_SPEECH", Stage.SPEECH_CHECK, f"{method} {speech}s")
    return {"method": method, "regions": regions, "speech_seconds": speech}


async def speech_check(ctx: "JobContext") -> None:
    """Stage function: write 02_vad.json."""
    result = await asyncio.to_thread(detect_speech, ctx.wav_path, ctx.settings, ctx.services.vad)
    ctx.write(ctx.output_name(Stage.SPEECH_CHECK), result)
