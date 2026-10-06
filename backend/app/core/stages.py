"""Pipeline stages, their output files and progress weights (spec Sections 2 and 9.1)."""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """Job states. Values are what the API and the UI see."""

    UPLOADED = "uploaded"
    QUEUED = "queued"
    VALIDATING = "validating"
    NORMALIZING = "normalizing"
    SPEECH_CHECK = "speech_check"
    TRANSCRIBING = "transcribing"
    RECHECKING = "rechecking"
    DIARIZING = "diarizing"
    RAW_SAVED = "raw_saved"
    VOCABULARY = "vocabulary"
    REFINING = "refining"
    GUARDING = "guarding"
    DOCUMENTING = "documenting"
    VERIFYING = "verifying"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


# Stages run by the pipeline, in order.
PIPELINE_STAGES: list[Stage] = [
    Stage.VALIDATING,
    Stage.NORMALIZING,
    Stage.SPEECH_CHECK,
    Stage.TRANSCRIBING,
    Stage.RECHECKING,
    Stage.DIARIZING,
    Stage.RAW_SAVED,
    Stage.VOCABULARY,
    Stage.REFINING,
    Stage.GUARDING,
    Stage.DOCUMENTING,
    Stage.VERIFYING,
    Stage.RENDERING,
]

# The file whose presence (as valid JSON) means the stage is done (resume rule).
STAGE_OUTPUT: dict[Stage, str] = {
    Stage.VALIDATING: "00_upload.json",
    Stage.NORMALIZING: "01_probe.json",
    Stage.SPEECH_CHECK: "02_vad.json",
    Stage.TRANSCRIBING: "03_whisper.json",
    Stage.RECHECKING: "04_recheck.json",
    Stage.DIARIZING: "05_diarization.json",
    Stage.RAW_SAVED: "raw_transcript.json",
    Stage.VOCABULARY: "06_vocabulary.json",
    Stage.REFINING: "07_lm1_edits.json",
    Stage.GUARDING: "08_refinement.json",
    Stage.DOCUMENTING: "09_lm2_raw.json",
    Stage.VERIFYING: "record.json",
    Stage.RENDERING: "exports/manifest.json",
}

# Relative weights of each stage in the overall progress bar.
STAGE_WEIGHTS: dict[Stage, float] = {
    Stage.VALIDATING: 2,
    Stage.NORMALIZING: 3,
    Stage.SPEECH_CHECK: 3,
    Stage.TRANSCRIBING: 35,
    Stage.RECHECKING: 10,
    Stage.DIARIZING: 7,
    Stage.RAW_SAVED: 0,
    Stage.VOCABULARY: 5,
    Stage.REFINING: 15,
    Stage.GUARDING: 1,
    Stage.DOCUMENTING: 15,
    Stage.VERIFYING: 2,
    Stage.RENDERING: 2,
}

STAGE_MESSAGES: dict[Stage, str] = {
    Stage.QUEUED: "Waiting for the GPU",
    Stage.VALIDATING: "Checking the file",
    Stage.NORMALIZING: "Converting audio",
    Stage.SPEECH_CHECK: "Detecting speech",
    Stage.TRANSCRIBING: "Transcribing",
    Stage.RECHECKING: "Double-checking unclear words",
    Stage.DIARIZING: "Labelling speakers",
    Stage.RAW_SAVED: "Saving the raw transcript",
    Stage.VOCABULARY: "Learning the meeting's vocabulary",
    Stage.REFINING: "Refining terminology",
    Stage.GUARDING: "Checking every edit",
    Stage.DOCUMENTING: "Writing the meeting record",
    Stage.VERIFYING: "Verifying the record against the transcript",
    Stage.RENDERING: "Preparing downloads",
    Stage.COMPLETED: "Done",
}


def overall_progress(stage: Stage, within: float = 0.0) -> float:
    """Overall progress in [0, 1] when `stage` is `within` (0..1) complete."""
    if stage == Stage.COMPLETED:
        return 1.0
    if stage not in PIPELINE_STAGES:
        return 0.0
    total = sum(STAGE_WEIGHTS.values())
    done = sum(STAGE_WEIGHTS[s] for s in PIPELINE_STAGES[: PIPELINE_STAGES.index(stage)])
    return round(min(1.0, (done + STAGE_WEIGHTS[stage] * max(0.0, min(1.0, within))) / total), 3)
