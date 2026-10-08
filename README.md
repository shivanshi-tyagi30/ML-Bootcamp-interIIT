# TRACE: Transparent Record & Acoustic Cross-Verification Engine

Evidence-grounded meeting transcription, guarded terminology refinement, and symbolic fact verification.

---

## What is TRACE?

Commercial meeting tools (Otter, Teams, Zoom AI) suffer from a common failure mode: they treat meeting summarization as an unconstrained generative text problem. When an automatic speech recognition (ASR) model mishears technical jargon, non-native accents, or Indian names ("PyTorch" as "pie torch", "Guwahati" as "Guhati"), downstream language models amplify the error. Even worse, generative models routinely invent task deadlines from relative phrases ("by next sprint" becomes an arbitrary calendar date), assign ownership to whoever brought up an idea rather than who accepted it, or turn tentative proposals into binding decisions.

TRACE was built to stop generative drift by treating language models as untrusted candidate generators constrained by deterministic checks:

- Speech-to-text and diarization run via fast cloud APIs (Groq Whisper large-v3 and AssemblyAI).
- Terminology corrections are proposed by Google Gemini 2.0 Flash, but applied only if they pass strict phonetic similarity tests (Jaro-Winkler plus Metaphone) and preserve numbers, negations, and modal verbs verbatim.
- Meeting records (minutes, decisions, action items) are extracted using structured JSON schemas with pointer-only constraints. The LLM cannot type an owner name or deadline date as freeform text; it must provide an exact pointer to a transcript segment.
- A deterministic symbolic verifier cross-checks quotes (RapidFuzz >= 90), regex agreement cues, deadline proximity, and speaker voice consistency before an item enters the final output.
- The entire stack is deployed live on a free-tier Render container (512 MB RAM, 0.1 CPU, zero GPU), delivering full-meeting processing in roughly 35 seconds at $0 infrastructure cost.

---

## Live Deployment (Render Free Tier)

TRACE runs live as a Docker container on Render's free tier without local model weights:

- Speech Recognition: Groq LPU Cloud running `whisper-large-v3` via API (transcribes 10 minutes of audio in under 8 seconds).
- Speaker Diarization: AssemblyAI Cloud API for speaker-turn detection.
- Language Models (LM1 Refiner & LM2 Documenter): Google Gemini 2.0 Flash (`gemini-2.0-flash`) using structured JSON schema decoding.
- Host Orchestration & Verification: FastAPI backend running on Python 3.11 with FFmpeg, Silero VAD, RapidFuzz, and jellyfish for deterministic checks.

All external credentials can be configured server-side or supplied directly through the web UI. Keys entered in the browser stay in client `localStorage` and are sent in-memory over HTTPS per upload; they are never written to disk or the database.

---

## System Architecture

```
                                  [ Audio Upload ]
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │    Stage 1: Ingest      │ 16 kHz Mono PCM (FFmpeg)
                            │   & Energy / VAD Gate   │ Floor >= 2.0s speech
                            └────────────┬────────────┘
                                         │
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
   ┌──────────────────────────┐                      ┌──────────────────────────┐
   │    Stage 2: Speech ASR   │                      │   Stage 4: Diarization   │
   │ Groq: whisper-large-v3   │                      │ AssemblyAI: Speaker Turns│
   │ Word timings & p_conf    │                      │ Timed anonymous labels   │
   └────────────┬─────────────┘                      └────────────┬─────────────┘
                │                                                 │
                ▼                                                 ▼
   ┌──────────────────────────┐                      ┌──────────────────────────┐
   │  Stage 3: Uncertainty    │                      │ Stage 5: Name Resolution │
   │ Freeze low-conf tokens   │                      │ Self-intro (W=3.0)       │
   │ p_conf < 0.45            │                      │ Direct address (W=1.0)   │
   └────────────┬─────────────┘                      └────────────┬─────────────┘
                │                                                 │
                └────────────────────────┬────────────────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │ Stage 6A: Vocabulary LM │ Gemini 2.0 Flash
                            │ Global domain extraction│ Technical glossary merge
                            └────────────┬────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │ Stage 6B: Refinement LM │ Sliding windows (N=40)
                            │ Proposed atomic edits   │ Read-only context (C=5)
                            └────────────┬────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │  Stage 7: Guard Engine  │ Invariant checks:
                            │ Deterministic rejection │ Numbers, negations, modals,
                            │ of unfaithful edits     │ Jaro-Winkler + Metaphone >= 0.75
                            └────────────┬────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │ Stage 8: Documenter LM  │ Gemini 2.0 Flash
                            │ Schema-guided extraction│ Pointer-only task evidence
                            └────────────┬────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │  Stage 9: Fact Verifier │ RapidFuzz quote gate (>= 90)
                            │ Demote unagreed items   │ Agreement regex cues
                            │ & prune alien facts     │ Deadline adjacency (<= 2 segs)
                            └────────────┬────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │ Stage 10: Fidelity Eval │ Preservation metrics,
                            │ & Export Generation     │ Markdown, DOCX, JSON
                            └────────────┬────────────┘
                                         ▼
                            ┌─────────────────────────┐
                            │ Synchronized Workspace  │ React 19 + TypeScript
                            │ 60 FPS audio-word sync  │ Citation jump (2s pre-roll)
                            └─────────────────────────┘
```

---

## Pipeline Execution Stages

Every job runs through a 13-stage state machine managed by `backend/app/pipeline/runner.py`. Each stage writes its intermediate output to `data/jobs/{job_id}/`:

| Index | State Name | Artifact | Description |
|---|---|---|---|
| 01 | `VALIDATING` | In-memory | Checks container integrity, audio stream, and MIME type |
| 02 | `NORMALIZING` | `01_audio.wav` | Standardizes audio to 16 kHz mono 16-bit linear PCM via FFmpeg |
| 03 | `SPEECH_CHECK`| In-memory | RMS gate plus Silero VAD v5.1; rejects files with under 2.0s speech (`E_NO_SPEECH`) |
| 04 | `TRANSCRIBING`| `02_whisper.json` | Groq Whisper large-v3 generates word-level timestamps and confidence scores |
| 05 | `RECHECKING`  | `03_recheck.json` | Tokens with `p_conf < 0.45` are marked disputed and frozen against editing |
| 06 | `DIARIZING`   | `04_diarization.json` | AssemblyAI extracts speaker intervals; fallback assigns pseudo-speakers on pauses |
| 07 | `RAW_SAVED`   | `raw_transcript.json` | Speaker intervals merged with words; self-introductions map clusters to names |
| 08 | `VOCABULARY`  | `05_vocab.json` | LM1 Pass A extracts technical acronyms and merges custom upload glossaries |
| 09 | `REFINING`    | `06_refine.json` | LM1 Pass B processes 40-segment windows and proposes atomic string replacements |
| 10 | `GUARDING`    | `refined_transcript.json`, `refinement.json` | Deterministic rules reject edits that alter numbers, polarity, or acoustics |
| 11 | `DOCUMENTING` | `07_document.json` | LM2 extracts candidate summary, minutes, decisions, and action items |
| 12 | `VERIFYING`   | `record.json` | Symbolic verifier audits quotes, demotes unagreed decisions, and verifies pointers |
| 13 | `RENDERING`   | `record.md`, `record.docx` | Calculates fidelity scorecard and builds downloadable deliverables |

Stage Caching & Recovery: If an external API call times out or hits a rate limit, the job enters `FAILED`. Retrying resumes execution directly from the last saved disk artifact without re-running transcription or diarization.

---

## Core Engineering Invariants

### 1. The Guardrail Invariant Engine (`guard.py`)
Language models frequently alter semantic meaning when attempting to correct transcription typos. TRACE evaluates every proposed edit `(orig, repl)` against strict invariant checks:

```python
# Every proposed edit must satisfy all conditions or it is rejected:
1. Substring Existence: orig must be an exact substring in the target segment text.
2. Number Conservation: Numbers(orig) == Numbers(repl)
3. Negation Conservation: Negations(orig) == Negations(repl)
4. Modal Conservation: Modals(orig) == Modals(repl)
5. Person Name Protection: If orig is detected as a person name, orig == repl
6. Length Cap: len(repl) <= 3 * max(len(orig), 4)
7. Phonetic Sound-Alike: Similarity(orig, repl) >= 0.75
8. Disputed Span Protection: Edits cannot touch words marked disputed by acoustic checks.
```

Phonetic similarity is computed as the maximum of squashed string similarity and spoken Metaphone encoding similarity:

$$\text{Sim}_{\text{phonetic}}(\text{orig}, \text{repl}) = \max\Big(\text{JW}(\text{orig}_{\text{squash}}, \text{repl}_{\text{squash}}),\ \text{JW}\big(\text{Metaphone}(\text{orig}),\ \text{Metaphone}(\text{repl}_{\text{spoken}})\big)\Big)$$

Acronyms are expanded into spoken letter names (`CUDA` expands to "see you dee ay"), allowing valid technical acronym fixes while blocking semantic drift (e.g., changing "test data" to "production cluster").

### 2. Pointer-Only Documentation Protocol (`lm2_document.py`)
To prevent hallucinated assignees and fabricated deadlines, the documenter LLM cannot emit freeform strings for owners or deadlines. It must return pointers:

```json
{
  "task": "Build the upload endpoint",
  "owner_evidence": {
    "segment_id": "S004",
    "exact_words": "Shivanshi"
  },
  "deadline_evidence": {
    "segment_id": "S004",
    "exact_words": "by Friday"
  },
  "evidence_quote": "I'll build the upload endpoint by Friday.",
  "evidence_segment_ids": ["S004"]
}
```

The backend code copies the string directly from the pointed segment text. If the pointer points to a disputed word, the field is downgraded to `Unspecified` with flag `audio_unclear`. If the pointer cannot be found verbatim, it becomes `Unspecified` with flag `pointer_invalid`.

### 3. Symbolic Fact Verification (`verify.py`)
The verification layer has zero generative capabilities; it can only remove or downgrade:

- Fuzzy Quote Matching: Evidence quotes cited for decisions and tasks must match the underlying transcript text with RapidFuzz `partial_ratio >= 90`. If the quote cannot be verified, the item is dropped.
- Decision Agreement Gate: Decisions require explicit conversational agreement cues matching:
  `agree | agreed | decided | let's go with | approved | confirmed | settled | fine by me | makes sense`
  If a candidate decision lacks supporting agreement evidence, it is automatically demoted to an open proposal (`demoted_from_decision = True`).
- Cross-Segment Self-Assignment: If a task owner is claimed via first-person pronoun ("I will take that"), the speaker label of the task segment must match the speaker label of the segment where that person introduced themselves. If unverified, the owner is set to `Unspecified` with flag `self_assignment_unverified`.
- Deadline Proximity: Cross-segment deadlines must occur within 2 segments of the task discussion (`DEADLINE_ADJACENCY = 2`).
- Unsupported Fact Pruning: Summary and minutes sentences are scanned for numbers and capitalized non-initial words. If an entity does not exist in the cited segment or its immediate neighbors (+/- 1 segment), the sentence is removed.

---

## Interactive Workspace & Frontend

The frontend is built with React 19, TypeScript, Tailwind CSS, and Vite:

- 60 FPS Playback Synchronization: The `useAudio` hook uses a `requestAnimationFrame` loop binding HTML5 audio `currentTime` to word boundary intervals. Active spoken words dynamically scale (`scale-106`) with real-time accent highlighting, while automatic container scrolling keeps active speech centered.
- Bidirectional Citation Chips: Every decision, proposal, and action item carries clickable citation chips (e.g., `[S014]`). Clicking any chip seeks the audio player to that exact segment with a 2-second lead-in pre-roll for acoustic context.
- Inline Transcript Diff View: Visualizes character- and word-level diffs, color-coding accepted technical replacements (green), rejected candidates (red strikethrough), and disputed audio words (amber dotted underline).
- Global Speaker Rename Propagation: Editing a speaker label cascades through the transcript, summary, minutes topics, and task assignments in client memory, persisting changes to the backend via `PATCH /api/jobs/{id}`.
- Multi-Device Scoping: Client sessions generate a unique UUID stored in `localStorage` and sent via `x-device-id`. The backend's SQLite database filters recent-meeting lists by device ID, ensuring users sharing a single hosted Render instance cannot see each other's recordings.

---

## Technical Configuration & Thresholds

All thresholds are defined in `backend/app/config.py`:

| Parameter | Value | Purpose |
|---|---|---|
| `MAX_FILE_MB` | `200` | Maximum upload size in megabytes |
| `MAX_DURATION_MIN` | `120` | Maximum audio duration in minutes |
| `MIN_SPEECH_SEC` | `2.0` | Minimum speech duration required to pass VAD |
| `LOWCONF_DISPUTE_THRESHOLD` | `0.45` | Token confidence floor; below this, words are marked disputed |
| `LM1_WINDOW_SEGMENTS` | `40` | Number of segments per LM1 refinement window |
| `LM1_CONTEXT_SEGMENTS` | `5` | Read-only context segments on each side of the refinement window |
| `LM1_MIN_CONFIDENCE` | `0.70` | Model confidence required to accept a terminology edit proposal |
| `SOUND_ALIKE_THRESHOLD` | `0.75` | Jaro-Winkler phonetic similarity threshold for terminology edits |
| `QUOTE_MATCH_THRESHOLD` | `90` | RapidFuzz partial token ratio threshold for cited quotes |
| `DEADLINE_ADJACENCY` | `2` | Maximum segment distance between a task quote and a deadline pointer |
| `LEAD_IN_S` | `2.0` | Seconds of audio pre-roll when jumping to a citation chip |
| `GEMINI_MODEL` | `gemini-2.0-flash`| Default cloud model for both LM1 and LM2 passes |

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/             # REST routes for jobs, health, and schemas
│   │   ├── core/            # Database, error types, stages, text utilities
│   │   ├── data/            # Pre-seeded tech terms and places dictionaries
│   │   ├── llm/             # Gemini client with retry and backoff logic
│   │   ├── models/          # Pydantic schemas for records and LLM inputs
│   │   ├── pipeline/        # 13-stage pipeline implementations
│   │   ├── config.py        # Environment settings and threshold constants
│   │   └── main.py          # FastAPI application entrypoint
│   ├── Dockerfile           # Production container build
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # Workspace, PlayerBar, TranscriptPane, RecordPane
│   │   ├── hooks/           # useAudio synchronization hook
│   │   └── lib/             # API client, diff annotations, speaker rename
│   └── package.json         # React 19 frontend dependencies
└── render.yaml              # Render Blueprint deployment configuration
```

---

## Local Development Setup

### Prerequisites
- Python 3.11 or newer
- Node.js 18 or newer
- FFmpeg and FFprobe installed on your system PATH

### 1. Backend Setup
```bash
cd backend
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements-windows.txt   # or requirements.txt on Linux
```

Create a `.env` file in `backend/` with your API keys:
```env
STT_BACKEND=groq
GROQ_API_KEY=your_groq_api_key
DIARIZATION_BACKEND=assemblyai
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
LOCAL_LLM_ENABLED=false
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
SERVE_FRONTEND=false
```

Start the backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The web interface will be available at `http://localhost:5173`. In development, Vite automatically proxies API requests to `http://localhost:8000`.

---

## API Summary

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/jobs` | Multipart audio upload with optional `title`, `glossary`, and `x-device-id` |
| `GET` | `/api/jobs` | Lists jobs filtered by the requester's `x-device-id` header |
| `GET` | `/api/jobs/{id}` | Returns job status, metadata, and final or partial records |
| `GET` | `/api/jobs/{id}/events` | Server-Sent Events (SSE) streaming live stage progress and errors |
| `POST` | `/api/jobs/{id}/retry` | Resumes a failed or cancelled job from its last completed checkpoint |
| `POST` | `/api/jobs/{id}/cancel` | Aborts a running job gracefully |
| `PATCH` | `/api/jobs/{id}` | Updates job title or persists corrected speaker names |
| `DELETE` | `/api/jobs/{id}` | Deletes job directory and database record |
| `GET` | `/api/jobs/{id}/audio` | Streams uploaded audio with HTTP Range support for seeking |
| `GET` | `/api/jobs/{id}/export` | Downloads record as `json`, `md`, `docx`, `txt_raw`, or `txt_refined` |
| `GET` | `/api/health` | Returns readiness status for FFmpeg, Groq, AssemblyAI, and Gemini |
