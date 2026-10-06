"""RECHECKING: second opinion from Parakeet on risky spans (spec Section 10.5)."""

from __future__ import annotations

import asyncio
import difflib
import logging
import random
import re
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.config import Settings
from app.core.stages import Stage
from app.core.text import MODAL, NEG, NUM, FILLERS, capitalized_non_initial, normalize_number

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

log = logging.getLogger(__name__)

TIME_WORDS = re.compile(
    r"\b(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|"
    r"quarter|eod|eow|morning|evening|deadline|by|before|until|next|end of)\b", re.I,
)
COMMITMENT = re.compile(r"\b(?:i'll|i will|we'll|can you|could you|will you|take that|on it|assign)\b", re.I)

RISK_HIGH_STAKES, RISK_LOW_CONF, RISK_AUDIT = 2, 1, 0


class RecheckASR(Protocol):
    """Second ASR model; tests provide a fake."""

    name: str

    def transcribe_clip(self, path: str) -> list[dict[str, Any]]:
        """Words [{w, start, end}] with times relative to the clip start."""
        ...


class ParakeetASR:
    """NVIDIA Parakeet-TDT via NeMo, loaded on first use."""

    def __init__(self, model_name: str) -> None:
        """Import NeMo now (raises ImportError if missing); load the model lazily."""
        import nemo.collections.asr  # noqa: F401  (fail fast when NeMo is absent)

        self.name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the model is in memory."""
        return self._model is not None

    def transcribe_clip(self, path: str) -> list[dict[str, Any]]:
        """Word timestamps for one clip."""
        import nemo.collections.asr as nemo_asr

        with self._lock:
            if self._model is None:
                self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.name)
        out = self._model.transcribe([path], timestamps=True)
        hyp = out[0][0] if isinstance(out, tuple) else out[0]
        stamps = getattr(hyp, "timestamp", None) or getattr(hyp, "timestep", None) or {}
        words = stamps.get("word", []) if isinstance(stamps, dict) else []
        return [
            {"w": w.get("word", ""), "start": float(w.get("start", 0)), "end": float(w.get("end", 0))}
            for w in words if w.get("word")
        ]


_default: ParakeetASR | None = None


def default_asr(settings: Settings) -> ParakeetASR:
    """Process-wide Parakeet singleton (raises ImportError without NeMo)."""
    global _default
    if _default is None:
        _default = ParakeetASR(settings.PARAKEET_MODEL)
    return _default


def flatten_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All Whisper words in order, each tagged with its Whisper segment index."""
    return [{**w, "seg": i} for i, s in enumerate(segments) for w in s["words"]]


def is_high_stakes(text: str) -> bool:
    """Numbers, negations, modals, time words, commitments or a likely name."""
    return any(p.search(text) for p in (NUM, NEG, MODAL, TIME_WORDS, COMMITMENT)) or bool(capitalized_non_initial(text))


def flag_spans(segments: list[dict[str, Any]], settings: Settings, seed: str) -> list[dict[str, Any]]:
    """Risky spans [{start, end, risk}] per the spec's three rules, merged and budgeted."""
    words = flatten_words(segments)
    spans: list[dict[str, Any]] = []
    flagged_segs: set[int] = set()
    for i, s in enumerate(segments):
        if s["words"] and is_high_stakes(s["text"]):
            spans.append({"start": s["start"], "end": s["end"], "risk": RISK_HIGH_STAKES})
            flagged_segs.add(i)
    for j, w in enumerate(words):
        win = words[max(0, j - 2) : j + 1]
        low_window = len(win) == 3 and sum(x["conf"] for x in win) / 3 < settings.RECHECK_WINDOW_CONF
        if w["conf"] < settings.RECHECK_CONF_THRESHOLD or low_window:
            first = win[0] if low_window else w
            spans.append({"start": first["start"], "end": w["end"], "risk": RISK_LOW_CONF})
            flagged_segs.add(w["seg"])
    rest = [i for i, s in enumerate(segments) if i not in flagged_segs and s["words"]]
    rng = random.Random(seed)
    for i in rng.sample(rest, k=round(len(rest) * settings.RECHECK_AUDIT_SHARE)):
        spans.append({"start": segments[i]["start"], "end": segments[i]["end"], "risk": RISK_AUDIT})

    # Merge overlaps, keeping the highest risk.
    spans.sort(key=lambda x: x["start"])
    merged: list[dict[str, Any]] = []
    for sp in spans:
        if merged and sp["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], sp["end"])
            merged[-1]["risk"] = max(merged[-1]["risk"], sp["risk"])
        else:
            merged.append(dict(sp))

    # Budget: highest risk first, stop once the share of audio is used up.
    duration = max((s["end"] for s in segments), default=0.0)
    budget = settings.RECHECK_MAX_SHARE * duration
    chosen, used = [], 0.0
    for sp in sorted(merged, key=lambda x: (-x["risk"], x["start"])):
        length = sp["end"] - sp["start"]
        if used + length > budget:
            break
        chosen.append(sp)
        used += length
    return sorted(chosen, key=lambda x: x["start"])


def align(whisper: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Word-level arbitration. Returns {index into `whisper`: update}.

    Equal words gain confidence (max with 0.9); replaced or deleted words are
    marked disputed with the other model's text. Insertions mark the previous
    word only when the inserted words are not fillers.
    """
    norm = lambda ws: [normalize_number(w["w"]) for w in ws]  # noqa: E731
    a, b = norm(whisper), norm(other)
    updates: dict[int, dict[str, Any]] = {}
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        alt = " ".join(w["w"] for w in other[j1:j2])
        if tag == "equal":
            for i in range(i1, i2):
                updates[i] = {"conf": max(whisper[i]["conf"], 0.9)}
        elif tag in ("replace", "delete"):
            for i in range(i1, i2):
                updates[i] = {"disputed": True, "alt": alt or ""}
        elif tag == "insert" and i1 > 0 and not set(b[j1:j2]) <= FILLERS:
            prev = i1 - 1
            updates[prev] = {"disputed": True, "alt": f"{whisper[prev]['w']} {alt}".strip()}
    return updates


def run_recheck(
    wav: Path, segments: list[dict[str, Any]], asr: RecheckASR, settings: Settings, seed: str, tmp_dir: Path,
) -> dict[str, Any]:
    """Re-decode flagged spans and return per-word updates keyed by global word index."""
    import soundfile as sf

    samples, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    words = flatten_words(segments)
    spans = flag_spans(segments, settings, seed)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    updates: dict[int, dict[str, Any]] = {}
    report = []
    try:
        for k, sp in enumerate(spans):
            c0 = max(0.0, sp["start"] - settings.RECHECK_PAD_SEC)
            c1 = sp["end"] + settings.RECHECK_PAD_SEC
            clip = tmp_dir / f"span_{k:04d}.wav"
            sf.write(str(clip), samples[int(c0 * sr) : int(c1 * sr)], sr)
            heard = [
                {**w, "start": w["start"] + c0, "end": w["end"] + c0} for w in asr.transcribe_clip(str(clip))
            ]
            inside = lambda w: sp["start"] <= (w["start"] + w["end"]) / 2 <= sp["end"]  # noqa: E731
            idx = [i for i, w in enumerate(words) if inside(w)]
            other = [w for w in heard if inside(w)]
            local = align([words[i] for i in idx], other)
            for li, upd in local.items():
                updates[idx[li]] = upd
            report.append({**sp, "whisper": " ".join(words[i]["w"] for i in idx),
                           "parakeet": " ".join(w["w"] for w in other),
                           "disputed": sum(1 for u in local.values() if u.get("disputed"))})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {
        "skipped": False, "model": asr.name, "spans": report,
        "word_updates": {str(i): u for i, u in sorted(updates.items())},
        "disputed_count": sum(1 for u in updates.values() if u.get("disputed")),
    }


async def recheck(ctx: "JobContext") -> None:
    """Stage function: write 04_recheck.json (never fails; skips when unavailable)."""
    out_name = ctx.output_name(Stage.RECHECKING)
    if not ctx.settings.RECHECK_ENABLED:
        ctx.write(out_name, {"skipped": True, "reason": "disabled", "word_updates": {}})
        return
    try:
        asr = ctx.services.recheck_asr or default_asr(ctx.settings)
    except ImportError:
        ctx.log.warning("NeMo not installed; skipping the Parakeet re-check")
        ctx.write(out_name, {"skipped": True, "reason": "nemo_unavailable", "word_updates": {}})
        return
    segments = ctx.read(ctx.output_name(Stage.TRANSCRIBING))["segments"]
    try:
        result = await asyncio.to_thread(
            run_recheck, ctx.wav_path, segments, asr, ctx.settings, ctx.job_id, ctx.dir / "tmp"
        )
    except Exception as e:  # noqa: BLE001 - this stage never fails the job
        ctx.log.exception("re-check failed; continuing without it")
        result = {"skipped": True, "reason": f"error: {e!r}", "word_updates": {}}
    ctx.write(out_name, result)
