# TRACE: System Architecture, Technical Specification & Engineering Context
**Deterministic Grounding, Acoustic Cross-Verification, and Invariant-Preserving Meeting Intelligence**

> **Note for LLM / Report Generators:** This document contains the complete, unadulterated engineering ground-truth of the TRACE platform. Use all technical parameters, algorithmic formulations, pipeline state machines, code structures, and design decisions detailed here to generate technical reports, competition submissions, or system documentation.

---

## 1. Executive Summary & Core Engineering Philosophy

### 1.1 The Failure Modes of Commercial Meeting AI
Commercial meeting assistants (e.g., Otter.ai, Zoom AI Companion, Microsoft Copilot) and naive LLM pipelines exhibit three fundamental failure modes:
1. **Phonetic & Domain Drift:** Standard Automatic Speech Recognition (ASR) engines hallucinate colloquial approximations when encountering technical jargon, non-native accents, or Indian names and terminology (e.g., transcribing *"PyTorch"* as *"pie torch"*, *"Kubernetes"* as *"cooper netties"*, *"Guwahati"* as *"Guhati"*, or *"FastAPI"* as *"fast a p i"*).
2. **Generative Fact Hallucination:** Downstream Large Language Models (LLMs) hallucinate commitments from passing conversational remarks, fabricate concrete calendar deadlines from relative expressions (*"by next sprint"* $\to$ invented dates), attribute tasks to passive listeners rather than committed assignees, or formalize unagreed proposals as binding decisions.
3. **Black-Box Opacity:** Meeting summaries lack verifiable bi-directional provenance. There is no cryptographic or millisecond-accurate acoustic trace linking an extracted decision or action item back to the exact speech segment in the audio waveform.

### 1.2 The TRACE Solution: Deterministic Grounding & Invariants
**TRACE (Transparent Record & Acoustic Cross-Verification Engine)** re-architects meeting intelligence from an acoustic-symbolic perspective:
- **LLMs are Untrusted Candidate Generators:** Language models are strictly prohibited from generating freeform output without evidence. They propose candidate vocabulary terms, candidate transcript edits, and candidate action items.
- **Deterministic Symbolic Verifiers:** Mathematical rules, phonetic distance metrics (Jaro-Winkler + Metaphone), regex agreement gates, and acoustic biometrics prove, constrain, or reject candidates.
- **Pointer-Only Extraction:** The language model cannot type an owner name or deadline date as a freeform string. It must supply an exact pointer `{"segment_id": "S012", "exact_words": "Monday"}`. Pointers are resolved by deterministic code directly from the transcript text.
- **Bi-Directional Provenance:** Every sentence, decision, proposal, and action item carries citation segment IDs (`evidence_segment_ids`), allowing the frontend UI to seek to the exact audio timestamp with a single click.

---

## 2. Complete Repository & Codebase Layout

```
ML-Bootcamp-interIIT/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes_jobs.py       # REST API: Job upload, status, cancel, retry, export, audio stream
│   │   │   ├── routes_schema.py     # OpenAPI / JSON Schema endpoints for MeetingRecord
│   │   │   └── routes_health.py     # System health, model readiness, ffmpeg checks
│   │   ├── core/
│   │   │   ├── db.py                # SQLite persistence with multi-device isolation (x-device-id)
│   │   │   ├── stages.py            # Pipeline state machine definitions (13 sequential stages)
│   │   │   ├── text.py              # Linguistic primitives: regex for numbers, negations, modals, cues
│   │   │   └── errors.py            # Error hierarchy (E_NO_SPEECH, E_LM1_FAILED, E_BUSY, etc.)
│   │   ├── data/
│   │   │   ├── tech_terms.txt       # Seed vocabulary of 500+ technical terms and frameworks
│   │   │   └── places.txt           # Standard spellings of Indian cities, institutions & tech hubs
│   │   ├── llm/
│   │   │   ├── client.py            # Multi-backend LLM client (Ollama native, vLLM, Google Gemini)
│   │   │   └── prompts/
│   │   │       ├── lm1_vocabulary.txt    # Pass A: Domain vocabulary & acronym extraction prompt
│   │   │       ├── lm1_refine.txt        # Pass B: Context-windowed terminology correction prompt
│   │   │       ├── lm2_document.txt      # LM2: Chain-of-evidence scratchpad documenter prompt
│   │   │       └── lm2_document_cloud.txt# Extended brief for cloud reasoning models
│   │   ├── models/
│   │   │   ├── record.py            # Pydantic schema: Segment, Word, Edit, Decision, ActionItem, MeetingRecord
│   │   │   └── llm_io.py            # Intermediate Pydantic schemas for structured LLM decoding
│   │   ├── pipeline/
│   │   │   ├── normalize.py         # Audio transcoding to 16 kHz mono PCM & ffprobe validation
│   │   │   ├── vad.py               # Silero VAD v5.1 speech activity detection + RMS energy gate
│   │   │   ├── stt_whisper.py       # faster-whisper (large-v3 / large-v3-turbo) local transcription
│   │   │   ├── stt_groq.py          # Groq Cloud Whisper large-v3 for serverless 512MB RAM mode
│   │   │   ├── recheck.py           # NVIDIA Parakeet-TDT 0.6B acoustic cross-check & dispute marking
│   │   │   ├── ecapa_diarize.py     # SpeechBrain ECAPA-TDNN 192-d speaker clustering
│   │   │   ├── speaker_names.py     # Conversational name resolution (self-intro + turn-taking address)
│   │   │   ├── assemble.py          # Merge word timings with speaker turns + pseudo-speaker fallback
│   │   │   ├── lm1_vocabulary.py    # Pass A: Global vocabulary induction & custom glossary merge
│   │   │   ├── lm1_refine.py        # Pass B: Sliding-window transcript refinement
│   │   │   ├── guard.py             # Deterministic guardrail engine: invariant checks on proposed edits
│   │   │   ├── lm2_document.py      # LM2 meeting documenter orchestration
│   │   │   ├── verify.py            # Deterministic symbolic verifier & item demotion engine
│   │   │   ├── fidelity.py          # Deterministic fidelity scorecard calculation
│   │   │   ├── render.py            # Markdown and DOCX report generator
│   │   │   └── runner.py            # Pipeline state machine runner with stage caching & resume
│   │   ├── config.py                # Pydantic Settings: environment variables and hardware profiles
│   │   └── main.py                  # FastAPI application entrypoint with CORS and static UI mount
│   ├── tests/                       # Pytest test suite with fake STT/LLM pipeline mocking
│   ├── requirements.txt             # Full GPU requirements (PyTorch, NeMo, SpeechBrain, faster-whisper)
│   ├── requirements-windows.txt     # CPU laptop profile (Torch CPU, faster-whisper int8, Ollama)
│   └── requirements-dev.txt         # Lightweight API & test runner requirements (no ML weights)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Workspace.tsx        # Main synchronized 3-pane layout
│   │   │   ├── PlayerBar.tsx        # Audio scrubber, waveform visualization, 60 FPS playhead
│   │   │   ├── TranscriptPane.tsx   # Word-level animated pop-up playback and inline editor
│   │   │   ├── RecordPane.tsx       # Grounded meeting record, citation chips, Diff view
│   │   │   ├── RecentMeetings.tsx   # Multi-device isolated meeting list
│   │   │   └── UploadModal.tsx      # Audio dropzone, glossary input, and cloud API key config
│   │   ├── hooks/
│   │   │   └── useAudio.ts          # requestAnimationFrame audio synchronization engine
│   │   └── lib/
│   │       ├── api.ts               # Backend REST & Server-Sent Events client
│   │       ├── annotate.ts          # Segment diff generator (accepted, rejected, disputed)
│   │       └── renameSpeaker.ts     # Global in-memory speaker rename cascade
│   └── package.json                 # React 19, TypeScript, Tailwind CSS, Vite
└── render.yaml                      # Render Blueprint for zero-GPU 512MB RAM cloud deployment
```

---

## 3. The 13-Stage Pipeline State Machine

Every processing job progresses through a linear, resumable state machine. Each stage writes intermediate JSON artifacts to `data/jobs/{job_id}/`:

```
[UPLOADED] (00_upload.json)
    │
    ▼
[QUEUED] ── (GPU / Worker Lock Acquisition)
    │
    ▼
[VALIDATING] ── Probes file container, audio stream health, mime-type
    │
    ▼
[NORMALIZING] (01_audio.wav) ── 16 kHz Mono 16-bit PCM via ffmpeg
    │
    ▼
[SPEECH_CHECK] ── RMS Energy Gate + Silero VAD v5.1 (floor >= 2.0s speech)
    │
    ▼
[TRANSCRIBING] (02_whisper.json) ── faster-whisper / Groq (word timestamps + p_conf)
    │
    ▼
[RECHECKING] (03_recheck.json) ── Parakeet-TDT 0.6B cross-check (freezes disputed spans)
    │
    ▼
[DIARIZING] (04_diarization.json) ── ECAPA-TDNN 192-d embeddings + Cosine Clustering
    │
    ▼
[RAW_SAVED] (raw_transcript.json) ── Speaker turns merged with words & conversational names
    │
    ▼
[VOCABULARY] (05_vocab.json) ── LM1 Pass A: Technical dictionary & acronym induction
    │
    ▼
[REFINING] (06_refine.json) ── LM1 Pass B: Sliding-window atomic edit proposals
    │
    ▼
[GUARDING] (refined_transcript.json, refinement.json) ── Invariant guardrail checks
    │
    ▼
[DOCUMENTING] (07_document.json) ── LM2: Structured extraction with evidence pointers
    │
    ▼
[VERIFYING] (record.json) ── Fuzzy quote match (>=90%), agreement gate, pointer resolution
    │
    ▼
[RENDERING] (record.md, record.docx) ── Export generation & fidelity metrics
    │
    ▼
[COMPLETED] ── Final result cached and served via SSE & REST API
```

*Fault Recovery:* If any stage fails (e.g., LM2 rate-limited), the job enters `FAILED`. The user can press **Retry**, which automatically resumes execution from the last saved disk artifact without re-running earlier compute-heavy stages (like ASR or Diarization).

---

## 4. Deep-Dive: Stage-by-Stage Engineering & Mathematical Invariants

### 4.1 Audio Ingestion & Normalization (`normalize.py`, `vad.py`)
- **Container Ingestion:** Validates against `ALLOWED_EXT = "mp3,wav,m4a,ogg,flac,webm,mp4,aac"`. Payload caps: 200 MB (`MAX_FILE_MB`), 120 minutes (`MAX_DURATION_MIN`).
- **Resampling:** Converts arbitrary bitrates and channel layouts to standardized 16 kHz mono 16-bit linear PCM:
  $$\text{PCM Audio} = \text{FFmpeg}(-i\ \text{input}, -ac\ 1, -ar\ 16000, -c:a\ \text{pcm\_s16le})$$
- **Two-Stage VAD Gate:**
  1. *RMS Energy Filter:* Scans for digital zero or silence ($E_{\text{rms}} < 10^{-4}$).
  2. *Silero VAD v5.1:* Evaluates 512-sample frames (32 ms). Voice probability threshold $p_t \ge 0.5$. Merges speech bursts across inter-speech pauses $< 300\text{ ms}$.
  - Invariant: Cumulative speech duration must satisfy $\sum t_{\text{speech}} \ge 2.0\text{ s}$ (`MIN_SPEECH_SEC`), else raises `E_NO_SPEECH`.

### 4.2 Primary ASR Transcription (`stt_whisper.py`, `stt_groq.py`)
- **Hardware-Adaptive Quantization:**
  - *CUDA GPU Profile:* `large-v3`, `float16`, beam size 5.
  - *CPU Laptop Profile:* `large-v3-turbo`, `int8`, beam size 1, parallelized across all cores (`cpu_threads = 0`).
  - *Cloud Serverless Profile:* Groq Cloud `whisper-large-v3` (executes in $< 10\text{ s}$ via streaming HTTP multipart upload).
- **Word Timings & Probabilities:** Generates per-word objects:
  $$w_i = (text_i, t_{\text{start}}, t_{\text{end}}, p_{\text{conf}}), \quad p_{\text{conf}} = \exp(\text{avg\_logprob})$$

### 4.3 Second-Opinion Acoustic Cross-Validation (`recheck.py`)
To prevent single-model acoustic blind spots, TRACE audits high-stakes spans using **NVIDIA Parakeet-TDT 0.6B** (NeMo FastConformer RNN-T):
1. **Candidate Span Selection:**
   - *High-Stakes Spans ($R=2$):* Tokens containing numbers, negations (*not, never, neither*), modal auxiliary verbs (*will, must, should, could*), or temporal markers with confidence $p_{\text{conf}} < 0.70$.
   - *Low-Confidence Spans ($R=1$):* Any token with $p_{\text{conf}} < 0.60$.
   - *Audit Spans ($R=0$):* Deterministic 5% pseudo-random sample of high-stakes terms regardless of reported confidence.
2. **Context Padding:** Spans are padded with 2.0 s pre/post acoustic context buffers, merged across overlaps, and capped at 35% of total audio length (`RECHECK_MAX_SHARE`).
3. **Dispute Invariant:** If Parakeet hypothesis conflicts with Whisper:
   $$w_{\text{primary}} \ne w_{\text{second}} \implies w_i.\text{disputed} = \text{True}, \quad w_i.\text{alt} = w_{\text{second}}$$
4. **CPU Fallback:** On machines without NeMo/Parakeet, any Whisper token with $p_{\text{conf}} < 0.45$ (`LOWCONF_DISPUTE_THRESHOLD`) is automatically flagged as disputed. **Disputed tokens are frozen; downstream LLMs cannot modify them.**

### 4.4 Voice Biometrics & Diarization (`ecapa_diarize.py`)
- **Local Biometrics:** SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`) extracts 192-dimensional embeddings.
- **Conversational Pause Splitting:** Speech segments are partitioned at conversational pauses ($\Delta t \ge 0.28\text{ s}$) or capped at $8.0\text{ s}$ max burst length. Short interjections $< 1.0\text{ s}$ (*"Yeah", "Sure"*) are excluded from cluster centroids.
- **Hypersphere Cosine Clustering:** Agglomerative clustering with average linkage:
  $$d_{\text{cosine}}(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}, \quad \text{threshold} = 0.60$$
  Clusters with $< 3.0\text{ s}$ of total speech are merged into their nearest neighbor to avoid microphone noise inflation.
- **Cloud / Pyannote Fallback:** Supports AssemblyAI API (used on Render) or local Pyannote 3.1 via `HF_TOKEN`. When diarization is completely unavailable, TRACE deploys a conversational pause heuristic (`assign_pseudo_speakers`) cycling speaker labels across pauses $> 1.5\text{ s}$.

### 4.5 Conversational Identity Resolution (`speaker_names.py`)
Binds anonymous acoustic clusters (`Speaker 1`, `Speaker 2`) to real human names spoken in the conversation:
1. **Self-Introduction Heuristic ($\text{Weight} = +3.0$):**
   Matches patterns: `/(?:this is|i am|i'm|my name is)\s+([A-Z][a-z]+)/`. The speaking cluster receives $+3.0$ votes for the declared identity.
2. **Direct Conversational Address ($\text{Weight} = +1.0$):**
   Matches patterns: `/(?:thanks|can you|over to|ask)\s+([A-Z][a-z]+)/`. If Speaker A addresses Name $N$ and Speaker B ($B \ne A$) speaks within a 2-segment reply window (`REPLY_WINDOW = 2`), Speaker B receives $+1.0$ vote for Name $N$, while Speaker A receives $-1.0$.
3. **Negative Lexicon Filter (`NOT_NAMES`):**
   Strictly filters out false-positive names:
   - Modals and determiners: *Can, Would, Should, Could, We, Let's*
   - Honorifics and conversational fillers: *ji, bhai, didi, yaar, sir, ma'am, boss, folks*
   - Known technical terms and places from `tech_terms.txt` and `places.txt`.
4. **Bipartite Assignment:** Solved greedily in descending order; each name and speaker cluster is bound at most once.

### 4.6 Two-Pass LLM Refinement & Deterministic Guardrails (`lm1_vocabulary.py`, `lm1_refine.py`, `guard.py`)
- **Pass A (Vocabulary Induction):** Scans the transcript using `LM1_MODEL` (`qwen3:14b` or `gemini-2.0-flash`) in 20k-token chunks. Identifies domain and extracts technical acronyms and misheard jargon:
  $$\text{Term} = \{\text{term}: \text{"FastAPI"}, \text{heard\_as}: [\text{"fast a p i"}, \text{"fast api"}]\}$$
  Merges with user-supplied glossary from the upload dialog.
- **Pass B (Sliding-Window Refinement):** Partitions the transcript into sliding windows (`LM1_WINDOW_SEGMENTS = 40`) with read-only context boundaries (`LM1_CONTEXT_SEGMENTS = 5`). Proposes discrete atomic edits:
  $$\text{Edit} = \{\text{segment\_id}, \text{original}, \text{replacement}, \text{category}, \text{confidence}\}$$
- **The Deterministic Guard Engine (`guard.py`):**
  Every proposed edit must pass through invariant checks before touching the transcript:
  $$\text{Check}(\text{Edit}) = \begin{cases}
  \text{REJECT}(\text{"low\_confidence"}) & \text{if } \text{conf} < 0.70 \\
  \text{REJECT}(\text{"not\_found"}) & \text{if } \text{orig} \notin \text{seg.text} \\
  \text{REJECT}(\text{"number\_changed"}) & \text{if } \text{Nums}(\text{orig}) \ne \text{Nums}(\text{repl}) \\
  \text{REJECT}(\text{"negation\_changed"}) & \text{if } \text{Neg}(\text{orig}) \ne \text{Neg}(\text{repl}) \\
  \text{REJECT}(\text{"modal\_changed"}) & \text{if } \text{Modals}(\text{orig}) \ne \text{Modals}(\text{repl}) \\
  \text{REJECT}(\text{"name\_changed"}) & \text{if } \text{IsPersonName}(\text{orig}) \land \text{orig} \ne \text{repl} \\
  \text{REJECT}(\text{"over\_rewrite"}) & \text{if } \text{len}(\text{repl}) > 3 \times \max(\text{len}(\text{orig}), 4) \\
  \text{REJECT}(\text{"not\_sound\_alike"}) & \text{if } \text{Sim}_{\text{phonetic}}(\text{orig}, \text{repl}) < 0.75 \\
  \text{REJECT}(\text{"touches\_disputed\_frozen"}) & \text{if } \text{OverlapsDisputedHighStakes}(\text{orig}) \\
  \text{ACCEPT} & \text{otherwise}
  \end{cases}$$
- **Phonetic Distance Formula:**
  $$\text{Sim}_{\text{phonetic}}(\text{orig}, \text{repl}) = \max\Big(\text{JW}(\text{orig}_{\text{squash}}, \text{repl}_{\text{squash}}),\ \text{JW}\big(\text{Metaphone}(\text{orig}),\ \text{Metaphone}(\text{repl}_{\text{spoken}})\big)\Big)$$
  - $\text{JW}$: Jaro-Winkler string similarity.
  - Spelled acronyms expand into spoken letter names (e.g. `CUDA` $\to$ *"see you dee ay"*).

### 4.7 Meeting Documentation & Symbolic Fact Verification (`lm2_document.py`, `verify.py`)
- **Constrained Schema Decoding:** `LM2_MODEL` (`gemma3:27b` or `gemini-2.0-flash`) operates under strict Pydantic JSON schema constraints.
- **Pointers-Only Protocol:** The model is prohibited from outputting raw strings for task owners or deadlines. It must supply an evidence pointer:
  $$\text{Pointer} = \{\text{segment\_id}: \text{"S015"}, \text{exact\_words}: \text{"by Friday"}\}$$
- **The Deterministic Symbolic Verifier (`verify.py`):**
  The verifier can **only remove or downgrade**; it has zero generative abilities:
  1. *Fuzzy Quote Match:* Quotes cited for decisions and tasks must fuzzy-match the cited segments:
     $$\text{fuzz.partial\_ratio}(\text{quote}, \text{cited\_segments}) \ge 90$$
     Failing quotes cause the item to be dropped.
  2. *Decision Agreement Regex Gate:* Decisions must match explicit agreement markers:
     $$\text{regex}(\text{"agree|agreed|decided|let's go with|approved|confirmed|settled|fine by me|makes sense"})$$
     If agreement evidence is missing or fails verification, the decision is demoted to an open proposal (`demoted_from_decision = True`).
  3. *Pointer Resolution & Acoustic Cross-Check:*
     - `exact_words` must appear verbatim in the referenced segment.
     - If the pointed words overlap a disputed word, the field is downgraded to `"Unspecified"` with flag `audio_unclear`.
     - *Cross-Segment Self-Assignment Check:* If an action item is claimed via first-person pronoun (*"I'll handle that"*), the task speaker's biometric cluster must match the speaker who introduced themselves. If unverified, the owner is downgraded to `"Unspecified"` with flag `self_assignment_unverified`.
     - *Deadline Adjacency Check:* Deadlines referenced in another segment must lie within $\le 2$ segments of the task discussion (`DEADLINE_ADJACENCY = 2`).
  4. *Unsupported Fact Pruning:*
     Scans summary and minutes sentences for numbers and capitalized non-initial words (potential proper nouns). If an extracted entity cannot be found in the cited segment or its immediate neighbor segments ($\pm 1$), the sentence is excised.

### 4.8 Deterministic Fidelity Scorecard (`fidelity.py`)
Every record includes a mathematical fidelity scorecard:
- **Numbers Preserved:** Ratio of numeric entities in the raw transcript preserved verbatim after refinement.
- **Negations Preserved:** Semantic polarity preservation ratio across all segments.
- **Items Downgraded / Dropped:** Cumulative audit log of hallucinated items suppressed.
- **Transcript Coverage:**
  $$\text{Coverage} = \frac{|\text{Cited Segments} \cap \text{Raw Segments}|}{|\text{Raw Segments}|} \times 100\%$$

---

## 5. Multi-Profile Deployment Runtime Architecture

TRACE runs across three distinct deployment environments using the exact same codebase:

| Parameter | GPU Server Profile (`.env.example`) | CPU Laptop Profile (`.env.cpu`) | Cloud Free-Tier Profile (`render.yaml`) |
|---|---|---|---|
| **Target Hardware** | NVIDIA GPU ($\ge 16\text{ GB VRAM}$) | 16 GB RAM Laptop (Windows/Mac/Linux) | Render Free Tier (512 MB RAM, 0.1 CPU) |
| **STT Engine** | `faster-whisper large-v3` (float16, beam 5) | `faster-whisper large-v3-turbo` (int8, beam 1) | Groq Cloud `whisper-large-v3` API |
| **Diarization** | ECAPA-TDNN (CUDA) or Pyannote | ECAPA-TDNN (CPU, PyTorch) | AssemblyAI Speaker-Turn API |
| **LM1 Refiner** | Ollama / vLLM `qwen3:14b` | Ollama `qwen3:8b` (thinking off) | Google Gemini `gemini-2.0-flash` |
| **LM2 Documenter** | Ollama / vLLM `gemma3:27b` | Ollama `gemma3:12b` (scratchpad on) | Google Gemini `gemini-2.0-flash` |
| **Memory Strategy** | Both models kept in VRAM (`keep_alive: -1`) | Sequential swapping (`LM1_UNLOAD_BEFORE_LM2`) | Zero local models; 512 MB memory limit |
| **Total 10-Min Run Time** | ~45 seconds | ~3.5 minutes | ~35 seconds |

### 5.1 Cloud Free-Tier Resilience Engineering (`client.py`)
- **Exponential Backoff on HTTP 429:** Handles Google Gemini's 15 Requests-Per-Minute (RPM) rate limits with automatic progressive backoff (retrying up to 4 times with adaptive cooldowns) while streaming live status alerts to the frontend via Server-Sent Events (SSE).
- **Multi-Model Auto-Rotation:** If a cloud model returns HTTP 503 ("high demand"), the client automatically falls back across available flash and pro models.
- **Multi-Device Privacy & Isolation:** SQLite storage uses device-scoped UUIDs (`x-device-id`), preventing users on separate devices from seeing each other's uploaded recordings while sharing a single backend instance.

---

## 6. Frontend Engineering & Reactive Synchronized Workspace

- **High-Frequency Audio Synchronization (`useAudio.ts`):** Operates a 60 FPS `requestAnimationFrame` loop binding playback `currentTime` to word boundary intervals:
  $$\text{ActiveWord} = \operatorname*{argmin}_i (t_{\text{current}} \ge w_i.\text{start} \land t_{\text{current}} < w_i.\text{end})$$
  Spoken words pop up dynamically (`scale(1.06)`, accent color, subtle elevation), while auto-scroll keeps the active sentence centered.
- **Bi-Directional Provenance Chips (`RecordPane.tsx`):** Every decision, proposal, and task includes clickable citation chips (`[S015]`). Clicking a chip jumps the audio player to the exact moment with a 2.0-second lead-in pre-roll (`LEAD_IN_S = 2.0`) for immediate acoustic context.
- **Global In-Memory Speaker Renaming (`renameSpeaker.ts`):** When a user corrects a speaker's name in the transcript, the change automatically cascades through summary paragraphs, minutes topics, and action item assignments across the client state and persists to `/api/jobs/{id}`.
- **Visual Diff View (`annotate.ts`):** Renders character-level and word-level diffs categorizing edits into accepted technical replacements (green highlight), blocked hallucinations (red strikethrough), and disputed audio words (amber dotted underline).

---

## 7. Complete Parameter & Constant Reference Table

| Constant | Value | Source File | Architectural Purpose |
|---|---|---|---|
| `MIN_SPEECH_SEC` | 2.0 s | `config.py` | Minimum speech duration required to accept an uploaded file |
| `MAX_FILE_MB` | 200 MB | `config.py` | Maximum accepted file upload payload |
| `MAX_DURATION_MIN` | 120 min | `config.py` | Maximum audio duration processed |
| `LOWCONF_DISPUTE_THRESHOLD` | 0.45 | `config.py` | Whisper token probability below which words are marked disputed on CPU |
| `PARAKEET_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | `config.py` | Secondary acoustic model for acoustic cross-validation |
| `RECHECK_CONF_THRESHOLD` | 0.60 | `config.py` | Token confidence threshold triggering secondary recheck |
| `RECHECK_WINDOW_CONF` | 0.70 | `config.py` | Confidence threshold for high-stakes tokens (numbers/negations) |
| `RECHECK_MAX_SHARE` | 0.35 (35%) | `config.py` | Maximum audio duration allocated to Parakeet recheck |
| `ECAPA_DISTANCE_THRESHOLD` | 0.60 | `config.py` | Cosine distance agglomerative clustering threshold |
| `BURST_GAP_SEC` | 0.28 s | `ecapa_diarize.py` | Conversational pause threshold indicating a potential turn change |
| `MAX_BURST_SEC` | 8.0 s | `ecapa_diarize.py` | Maximum speech burst length before forcing a split |
| `MIN_CLUSTER_SEC` | 1.0 s | `ecapa_diarize.py` | Bursts shorter than this cannot initiate a new speaker cluster |
| `SELF_WEIGHT` | 3.0 | `speaker_names.py` | Identity voting weight for self-introduction patterns |
| `ADDRESS_WEIGHT` | 1.0 | `speaker_names.py` | Identity voting weight for direct conversational address |
| `REPLY_WINDOW` | 2 segments | `speaker_names.py` | Conversational reply window checked for turn-taking response |
| `LM1_WINDOW_SEGMENTS` | 40 | `config.py` | Segment count per Pass B terminology refinement window |
| `LM1_CONTEXT_SEGMENTS` | 5 | `config.py` | Boundary context segments per refinement window |
| `LM1_MIN_CONFIDENCE` | 0.70 | `config.py` | Minimum model confidence required to accept a terminology edit |
| `SOUND_ALIKE_THRESHOLD` | 0.75 | `config.py` | Jaro-Winkler phonetic similarity threshold for terminology edits |
| `QUOTE_MATCH_THRESHOLD` | 90 | `config.py` | Fuzzy partial token ratio threshold for evidence quotes |
| `DEADLINE_ADJACENCY` | 2 segments | `verify.py` | Maximum distance between a task quote and a deadline pointer |
| `LEAD_IN_S` | 2.0 s | `Workspace.tsx` | Pre-roll audio offset when jumping to an evidence chip |

---

## 8. Summary for Technical Writing / Competition Submission
When compiling the technical documentation or submission paper from this specification, structure the core narrative around:
1. **The Core Thesis:** TRACE replaces ungrounded generative summarization with an **acoustic-symbolic verification engine**.
2. **Dual-Model Acoustic Audit:** Using Whisper + Parakeet to freeze ambiguous words so LLMs cannot rewrite them.
3. **Deterministic Guardrails:** Rejecting semantic drift via mathematical invariants (numbers, negations, modals, and phonetic Metaphone matching).
4. **Pointer-Only Extraction & Symbolic Verifier:** Prohibiting LLMs from generating ungrounded facts, enforcing 90% fuzzy quote matches, regex agreement gates, and biometric self-assignment checks.
5. **Universal Portability:** Seamless operation across a high-end GPU server, a 16 GB CPU-only laptop, and a free-tier 512 MB cloud container.
