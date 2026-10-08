# TRACE: Transparent Record & Acoustic Cross-Verification Engine

Grounded meeting transcription and structured documentation. LLMs propose candidates; deterministic Python rules prove, constrain, or reject them.

---

## The Problem & Core Differentiating Point

Commercial meeting tools treat summarization as an unconstrained text generation problem. They hallucinate dates from vague conversational remarks ("by next sprint" becomes an arbitrary date), attribute tasks to whoever spoke rather than who committed, and mishear technical jargon.

TRACE enforces deterministic grounding through three hard invariants:

1. **Guarded Terminology Refinement (`guard.py`):**
   LM1 proposes corrections for misheard domain terms, but an invariant engine checks every candidate. Edits are rejected if they alter numbers, negate sentences, change modal verbs (e.g., "might" to "will"), or fail phonetic similarity checks (Jaro-Winkler plus Metaphone >= 0.75).

2. **Pointer-Only Evidence Protocol (`lm2_document.py`):**
   The documenter LLM cannot emit freeform strings for task owners or deadlines. It must supply an exact pointer to a transcript segment (`{"segment_id": "S004", "exact_words": "by Friday"}`). The backend resolves the text verbatim from the transcript. If the pointed audio is ambiguous or disputed, it downgrades to `Unspecified`.

3. **Symbolic Fact Verification (`verify.py`):**
   A non-generative audit engine runs after extraction. Quotes cited for decisions and tasks must achieve a RapidFuzz token match >= 90%. Decisions without explicit agreement cues ("agreed", "let's go with", "approved") are demoted to open proposals. Uncited claims are excised.

---

## Live Architecture (Render Free Tier)

TRACE runs live on a 512 MB RAM Render container at $0 infrastructure cost by pairing remote inference APIs with local verification:

```
Audio Input ──► Groq Cloud (whisper-large-v3) ──► 10-min audio transcribed in <8s
            ──► AssemblyAI Cloud               ──► Speaker turn diarization
            ──► Gemini 2.0 Flash (LM1)         ──► Proposes terminology edits
            ──► Python Guard Engine            ──► Invariant & phonetic check
            ──► Gemini 2.0 Flash (LM2)         ──► Structured JSON candidates
            ──► Python Fact Verifier           ──► Quotes, agreement, pointers
            ──► React 19 Workspace             ──► 60 FPS audio-word playback
```

- **Groq LPU Cloud:** High-speed `whisper-large-v3` transcription without host GPU requirements.
- **AssemblyAI:** Acoustic speaker turns merged with conversational name-binding heuristics (self-introductions + direct address).
- **Google Gemini 2.0 Flash:** Structured JSON schema decoding for both terminology refinement (LM1) and documentation (LM2).
- **FastAPI Host (512 MB RAM):** Media normalization, VAD gating, invariant validation, and export rendering running on CPU in milliseconds.

---

## Interactive Review Workspace

- **60 FPS Audio-Word Sync:** High-frequency playback tracking highlights active words (`scale-106`) and keeps spoken sentences centered in real time.
- **Bi-Directional Provenance:** Every decision, proposal, and action item carries citation chips (e.g. `[S003]`). Clicking any chip seeks the audio player to the exact moment with a 2-second lead-in.
- **Inspectable Diffs:** Shows accepted terminology fixes (green), rejected hallucinations (red strikethrough), and disputed low-confidence words (amber underline).
- **Device Privacy:** Job access and history are isolated per device using client UUIDs (`x-device-id`). User-supplied API keys stay in browser `localStorage` and are never written to the host disk.

---

## Key Thresholds

| Parameter | Setting | Purpose |
|---|---|---|
| `LOWCONF_DISPUTE_THRESHOLD` | `< 0.45` | Freezes uncertain ASR tokens against LLM rewrites |
| `LM1_WINDOW_SEGMENTS` | `40` | Sliding window size for localized terminology refinement |
| `SOUND_ALIKE_THRESHOLD` | `0.75` | Minimum Jaro-Winkler phonetic similarity for edit acceptance |
| `QUOTE_MATCH_THRESHOLD` | `90` | RapidFuzz partial token ratio required to validate quotes |
| `DEADLINE_ADJACENCY` | `<= 2` | Maximum segment distance between a task and a deadline pointer |
| `LEAD_IN_S` | `2.0 s` | Audio offset when jumping to a citation chip |

---

## Quick Start

### 1. Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt  # or requirements.txt on Linux
```

Configure your `.env`:
```env
STT_BACKEND=groq
GROQ_API_KEY=your_key
DIARIZATION_BACKEND=assemblyai
ASSEMBLYAI_API_KEY=your_key
GEMINI_API_KEY=your_key
LOCAL_LLM_ENABLED=false
```

Run the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
