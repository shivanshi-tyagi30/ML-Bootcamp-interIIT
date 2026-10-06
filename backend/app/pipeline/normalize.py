"""NORMALIZING: probe the file and convert to 16 kHz mono WAV (spec Section 10.2)."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.core.errors import PipelineError
from app.core.stages import Stage

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext


def probe(path: Path, settings: Settings) -> dict[str, Any]:
    """ffprobe the file; raises E_UNREADABLE (no audio) or E_TOO_LONG."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(out.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        raise PipelineError("E_UNREADABLE", Stage.NORMALIZING, repr(e)) from e
    audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if out.returncode != 0 or not audio:
        raise PipelineError("E_UNREADABLE", Stage.NORMALIZING, out.stderr.strip()[:500])
    a = audio[0]
    duration = float(data.get("format", {}).get("duration") or a.get("duration") or 0)
    if duration > settings.MAX_DURATION_MIN * 60:
        raise PipelineError("E_TOO_LONG", Stage.NORMALIZING, f"{duration:.0f}s")
    return {
        "codec": a.get("codec_name"),
        "duration_s": round(duration, 3),
        "sample_rate": int(a.get("sample_rate") or 0),
        "channels": int(a.get("channels") or 0),
        "format": data.get("format", {}).get("format_name"),
    }


def to_wav16k(src: Path, dst: Path) -> None:
    """Convert to 16 kHz mono 16-bit PCM with loudness normalisation; raises E_UNREADABLE."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "pcm_s16le", str(dst),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        raise PipelineError("E_UNREADABLE", Stage.NORMALIZING, repr(e)) from e
    if out.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        raise PipelineError("E_UNREADABLE", Stage.NORMALIZING, out.stderr.strip()[:500])


async def normalize(ctx: "JobContext") -> None:
    """Stage function: write audio_16k.wav and 01_probe.json."""
    info = await asyncio.to_thread(probe, ctx.original_path, ctx.settings)
    await asyncio.to_thread(to_wav16k, ctx.original_path, ctx.wav_path)
    ctx.write(ctx.output_name(Stage.NORMALIZING), info)
