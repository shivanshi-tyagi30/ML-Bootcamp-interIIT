# TRACE Backend

FastAPI service implementing deterministic acoustic verification, guarded transcript refinement, and symbolic meeting documentation.

---

## Architecture & Service Boundary

The backend is engineered as a lightweight orchestration and verification engine designed to deploy on minimal cloud footprints (512 MB RAM on Render) by offloading heavy inference to specialized cloud APIs while executing all verification locally in pure Python:

- **Audio Ingestion & Gating:** FFmpeg standardizes audio to 16 kHz mono 16-bit PCM. Silero VAD v5.1 rejects digital silence or clips under 2.0s (`MIN_SPEECH_SEC`).
- **Speech Recognition:** Groq LPU Cloud running `whisper-large-v3` returns word-level timestamps and confidence scores in under 8 seconds.
- **Acoustic Diarization:** AssemblyAI Cloud API detects speaker turns, followed by local conversational name-binding heuristics (self-introductions + direct address).
- **Terminology Refiner (LM1):** Google Gemini 2.0 Flash proposes atomic vocabulary edits over sliding windows (40 segments + 5 context segments).
- **Deterministic Guardrail Engine:** Pure Python invariants (numbers, negations, modals, length caps, and Jaro-Winkler + Metaphone phonetic distance >= 0.75) validate every candidate edit.
- **Meeting Documenter (LM2):** Google Gemini 2.0 Flash generates structured meeting candidates under strict Pydantic JSON schemas with pointer-only task constraints.
- **Symbolic Fact Verifier:** RapidFuzz quote matching (>= 90%), regex agreement cue gates, deadline proximity (<= 2 segments), and biometric voice attribution audit every claim.
- **Persistence & Isolation:** SQLite tracks job states with `x-device-id` multi-device scoping; intermediate stage outputs are saved to disk for instant recovery on retry.

---

## 13-Stage Pipeline State Machine

Every meeting job progresses through a linear, resumable state machine in `app/pipeline/runner.py`. Each stage writes its output to `data/jobs/{job_id}/`:

```
UPLOADED -> QUEUED -> VALIDATING -> NORMALIZING -> SPEECH_CHECK -> TRANSCRIBING
  -> RECHECKING -> DIARIZING -> RAW_SAVED -> VOCABULARY (LM1 Pass A)
  -> REFINING (LM1 Pass B) -> GUARDING -> DOCUMENTING (LM2) -> VERIFYING
  -> RENDERING -> COMPLETED
```

- Stage Checkpoints: If an external API call hits a rate limit or network glitch, the job enters `FAILED`. Calling `POST /api/jobs/{id}/retry` resumes directly from the last saved stage without re-running earlier compute-heavy transcription.
- Multi-Device Scoping: Jobs are tagged with the requester's `x-device-id` UUID header, isolating meeting histories across separate browser devices.

---

## Core Invariant Logic

### 1. Guardrail Invariants (`app/pipeline/guard.py`)
```python
# Every LM1 edit (original -> replacement) must satisfy:
1. Substring exists in target segment text
2. Numbers(orig) == Numbers(repl)
3. Negations(orig) == Negations(repl)
4. Modals(orig) == Modals(repl)
5. Person names cannot be modified
6. len(repl) <= 3 * max(len(orig), 4)
7. Phonetic similarity: max(JW(squashed), JW(Metaphone)) >= 0.75
8. Cannot touch words flagged as disputed (confidence < 0.45)
```

### 2. Pointer Extraction Protocol (`app/pipeline/lm2_document.py`)
Language models are forbidden from generating freeform owner names or calendar dates. They must emit pointers:
```json
{
  "task": "Build the upload endpoint",
  "owner_evidence": {"segment_id": "S004", "exact_words": "Shivanshi"},
  "deadline_evidence": {"segment_id": "S004", "exact_words": "by Friday"},
  "evidence_quote": "I'll build the upload endpoint by Friday.",
  "evidence_segment_ids": ["S004"]
}
```

### 3. Symbolic Verifier (`app/pipeline/verify.py`)
- Quote Gate: Quotes must match transcript text with RapidFuzz `partial_ratio >= 90`.
- Agreement Gate: Decisions require explicit confirmation cues (`agree`, `decided`, `let's go with`, `approved`). Unconfirmed decisions are demoted to open proposals.
- Self-Assignment Gate: Claiming tasks via first-person pronouns ("I will do this") requires the task speaker's biometric cluster to match the speaker who introduced themselves.
- Deadline Adjacency: Cross-segment deadlines must lie within 2 segments of the task quote.

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/jobs` | Upload audio (multipart `file`, optional `title`, `glossary`, `x-device-id`) |
| `GET` | `/api/jobs` | List jobs filtered by `x-device-id` header |
| `GET` | `/api/jobs/{id}` | Retrieve job status, metadata, and final or partial record |
| `GET` | `/api/jobs/{id}/events` | Server-Sent Events (SSE) streaming live progress (0-100%) and stage updates |
| `POST` | `/api/jobs/{id}/retry` | Resume a failed job from its last completed checkpoint |
| `POST` | `/api/jobs/{id}/cancel` | Abort a running job gracefully |
| `PATCH` | `/api/jobs/{id}` | Rename meeting or persist corrected speaker labels across records |
| `DELETE` | `/api/jobs/{id}` | Remove job artifacts and database entry |
| `GET` | `/api/jobs/{id}/audio` | Stream original audio with HTTP Range support for seeking |
| `GET` | `/api/jobs/{id}/export` | Export record (`?fmt=json|md|docx|txt_raw|txt_refined`) |
| `GET` | `/api/health` | Service health check (FFmpeg, API readiness) |

---

## Configuration & Environment Variables

Key runtime settings in `app/config.py`:

```env
# Cloud Service Backends
STT_BACKEND=groq
GROQ_API_KEY=your_groq_api_key
GROQ_STT_MODEL=whisper-large-v3

DIARIZATION_BACKEND=assemblyai
ASSEMBLYAI_API_KEY=your_assemblyai_api_key

LOCAL_LLM_ENABLED=false
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

# Thresholds & Limits
MAX_FILE_MB=200
MAX_DURATION_MIN=120
MIN_SPEECH_SEC=2.0
LOWCONF_DISPUTE_THRESHOLD=0.45
LM1_WINDOW_SEGMENTS=40
LM1_CONTEXT_SEGMENTS=5
LM1_MIN_CONFIDENCE=0.70
SOUND_ALIKE_THRESHOLD=0.75
QUOTE_MATCH_THRESHOLD=90
DEADLINE_ADJACENCY=2
```

---

## Local Setup

```bash
cd backend
python -m venv .venv

# Activate:
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements-windows.txt   # Windows
pip install -r requirements.txt           # Linux

# Run:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
