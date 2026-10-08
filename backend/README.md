# TRACE backend

FastAPI service that turns an English meeting recording into a raw transcript, a refined transcript and a
grounded meeting record (summary, minutes, decisions, open proposals, action items). Owners and deadlines are
copied from the transcript by code or set to `Unspecified`; the language models never type them.
Built from `TRACE Backend Build Spec v1.0`; deviations are listed at the end.

## Models

| Role | Model | Version / id | What it does |
|---|---|---|---|
| Speech-to-text | Whisper large-v3 via faster-whisper | `large-v3` (`WHISPER_MODEL`) | Words, timestamps and per-word confidence |
| Speech check | Silero VAD | `silero-vad>=5.1` | Rejects recordings with under 2 s of speech |
| Re-check (optional) | NVIDIA Parakeet-TDT-0.6B | `nvidia/parakeet-tdt-0.6b-v2` | Re-hears risky spans; disagreements become *disputed* words |
| Diarization | ECAPA-TDNN / pyannote | `speechbrain/spkrec-ecapa-voxceleb` or pyannote | 192-d speaker embeddings clustered with cosine distance (Speaker 1, Speaker 2 …) |
| LM1 refiner | Qwen3 14B | `qwen3:14b` (`LM1_MODEL`) | Pass A: meeting vocabulary. Pass B: terminology edits, checked by the guard |
| LM2 documenter | Gemma 3 27B | `gemma3:27b` (`LM2_MODEL`) | Summary, minutes, decisions, proposals, tasks with pointers and citations |

The models actually used are written to every record in `meta.models`.

## How outputs move between stages

```
UPLOADED -> QUEUED -> VALIDATING -> NORMALIZING -> SPEECH_CHECK -> TRANSCRIBING
  -> RECHECKING -> DIARIZING (optional) -> RAW_SAVED -> VOCABULARY (LM1 pass A)
  -> REFINING (LM1 pass B) -> GUARDING -> DOCUMENTING (LM2) -> VERIFYING
  -> RENDERING -> COMPLETED
Any state --(PipelineError)--> FAILED {code, stage, user_message}
```

Every stage writes its output to `data/jobs/{job_id}/` before the next starts (`00_upload.json` …
`record.json`, `exports/`). A restarted job skips stages whose output already exists. If LM2 fails, the
job is `failed` but both transcripts stay viewable and downloadable. Only one job uses the GPU at a time;
others wait as `queued`.

## Setup

System packages: **ffmpeg**, **ffprobe**, **libmagic** (`apt install ffmpeg libmagic1`), Python 3.11+, an NVIDIA GPU
for the full pipeline.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full pipeline (GPU)
# or: pip install -r requirements-dev.txt  # API + tests only, no ML models
cp .env.example .env                     # then edit as needed
```

Language models (pick one):

```bash
# Ollama (default; serves both models and swaps them on one GPU)
ollama pull qwen3:14b
ollama pull gemma3:27b

# vLLM (one model per server): set LLM_BACKEND=vllm, LLM_BASE_URL, LM2_BASE_URL and the model ids
vllm serve Qwen/Qwen3-14B --port 8001
vllm serve google/gemma-3-27b-it --port 8002
```

Diarization uses SpeechBrain's **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) by default without requiring
gated tokens or accounts. It extracts 192-d speaker embeddings from speech bursts and groups them using
Agglomerative Clustering with cosine distance. Bursts under 1 s ("Yes.", "Agreed.") never start a speaker of
their own; they join the closest voice. Speakers with under 3 s of speech are merged into the nearest one. If you
know how many people spoke, set `DIARIZATION_NUM_SPEAKERS` (most accurate); otherwise tune
`ECAPA_DISTANCE_THRESHOLD` (higher = fewer speakers). Needs `speechbrain` and `scikit-learn`; the model (~80 MB)
is fetched into `data/models/ecapa` by `python -m app.prefetch`. If diarization is off or unavailable, speakers are left empty, never
guessed. Alternatively, Pyannote can be enabled with `DIARIZATION_BACKEND=pyannote` and an `HF_TOKEN`.

Place and institution names are corrected to their standard spelling ("Guhati" -> "Guwahati") when the new spelling
is in the meeting vocabulary, the user's expected terms, or `app/data/places.txt`; people's names stay as heard. Without Parakeet, Whisper words below `LOWCONF_DISPUTE_THRESHOLD` are marked disputed.

### Windows laptop, CPU only

Install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) (`winget install Gyan.FFmpeg`, then open a new terminal) and
[Ollama](https://ollama.com/download). In PowerShell:

```powershell
cd backend
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, if Activate.ps1 is blocked
.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
copy .env.cpu .env
python -m app.prefetch        # downloads every model once (Whisper, speaker model, both Ollama models)
uvicorn app.main:app --port 8000
```

Two laptop profiles:

| Setting | `.env.cpu` (best quality, 16 GB RAM) | `.env.cpu-fast` (quicker) | Why |
|---|---|---|---|
| Whisper | `large-v3-turbo`, int8, **beam 5** | `large-v3-turbo`, int8, beam 1 | turbo is close to large-v3 accuracy at a fraction of the CPU time; beam search adds accuracy cheaply |
| Speakers | pyannote when `HF_TOKEN` is set, else ECAPA | ECAPA | pyannote handles short replies and overlap better |
| LM1 | `qwen3:8b`, thinking off | `qwen3:4b`, thinking off | the 8B model knows far more names, places and technical terms |
| LM2 | `gemma3:12b`, scratchpad on | `gemma3:4b`, scratchpad off | the 12B model separates decisions, proposals and tasks much better; the scratchpad makes it gather evidence first |
| Memory | one LLM in memory at a time (`LM1_UNLOAD_BEFORE_LM2=true`) | both stay loaded | 8B + 12B do not fit in 16 GB together; swapping to disk would be far slower than reloading |
| LM1 windows | `LM1_WINDOW_SEGMENTS=120` | same | a 5-10 minute meeting is refined in one call |
| Context | `num_ctx` sized per call (8k/12k/16k/32k) | same | Ollama's 2-4k default silently cut long prompts |
| Re-check | off (NeMo does not install on Windows) | same | low-confidence words are marked disputed instead |

`.env.cpu` takes roughly 2-3x longer than `.env.cpu-fast` on the same laptop; with an NVIDIA GPU the defaults
(`.env.example`) are both faster and higher quality. Stage times are written to `data/jobs/{id}/timings.json`
and shown in the workspace header.

## Cloud model (Gemini or any OpenAI-compatible API)

One cloud model can do both LM1 and LM2: much stronger than a laptop model and far faster on CPU-only machines.
Get a free key at aistudio.google.com, then either `copy .env.gemini .env` and paste it as `GEMINI_API_KEY`, or
paste it in the app's API key box (it is sent only with uploads, kept in server memory, never written to disk).

- The record is written from `app/llm/prompts/lm2_document_cloud.txt`, a full minute-taking brief (executive
  summary, topic-by-topic minutes with positions and reasons, decisions, tasks, speaker checks). Small local
  models keep the shorter `lm2_document.txt`. A test runs the brief's own example through the verifier.
- The request asks for the exact record structure (`json_schema`); providers that reject it get JSON mode with
  the structure in the prompt. The guard, verifier and speaker-name checks still check every edit and item.
- With a key, the cloud model is the only language model: the local Ollama model runs only for uploads without a
  key. A busy model (HTTP 503 "high demand") is retried, then another cloud model the key can use takes over. A bad
  key, no internet or every model busy give a clear error, and Retry continues from the failed step.
- Re-uploading a file reuses the earlier result only if it was made the same way (cloud or local).
- `GEMINI_MODEL`, `CLOUD_BASE_URL` (e.g. `https://api.openai.com/v1` for GPT), `CLOUD_REASONING_EFFORT`,
  `CLOUD_TIMEOUT_SEC` configure it. Free tiers have rate limits, and transcript text is
  sent to the provider.

## Deploying (no user ever waits for a model)

1. **At deploy time** run `python -m app.prefetch`. It downloads every model the server will use on that machine
   (the Whisper size it picks for CPU or GPU, Silero VAD, the speaker model, both Ollama models) and exits with
   status 1 if anything is missing. Run it in your image build or setup script, not at request time.
2. **At startup** the server loads all of them into memory *before* it opens the port (`WARMUP_ON_START=true`),
   using the same context size as real calls so Ollama never reloads. The log says
   `all models loaded in N s; ready`, and `GET /api/health` returns `"ready": true`; point your platform's
   health check at it.
3. **While running** Ollama keeps both models loaded (`OLLAMA_KEEP_ALIVE=-1m`) and LM1 is not unloaded before
   LM2, so every upload, including the first one after a quiet hour, starts with everything already in memory.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then start the frontend (`cd ../frontend && npm run dev`); it proxies `/api` to port 8000.

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/api/jobs` | multipart `file`, optional `title`, `glossary` (comma-separated); `?force=true` skips the dedupe cache. `202 {"job_id","status":"queued","cached":false}`; a file already processed returns `200 {"job_id": <existing>, "cached": true}` |
| GET | `/api/jobs` | `?limit=50&offset=0` → `{"items": [JobSummary], "total"}`, newest first |
| GET | `/api/jobs/{id}` | `{"job", "record" or null, "partial": {"raw_transcript", "refined_transcript", "refinement"}}` |
| GET | `/api/jobs/{id}/events` | SSE, event `progress` (see below) |
| PATCH | `/api/jobs/{id}` | `{"title"}`; updates the record and regenerates exports |
| DELETE | `/api/jobs/{id}` | `204`; `409` while queued or running |
| POST | `/api/jobs/{id}/cancel` | Stops a queued or running job at the next safe point (job becomes `failed` with `E_CANCELLED`); returns the JobSummary |
| POST | `/api/jobs/{id}/retry` | `202`; runs a failed or cancelled job again, resuming after the last saved stage; `409` while running |
| GET | `/api/jobs/{id}/audio` | Original file, supports HTTP Range |
| GET | `/api/jobs/{id}/export` | `?fmt=json\|md\|docx\|txt_raw\|txt_refined`; file name from the title slug |
| GET | `/api/schema/record` | JSON Schema of `MeetingRecord` |
| GET | `/api/health` | `{"status","ffmpeg","llm":{"backend","host","reachable","missing"},"whisper":{...},"models","gpu"}` |

SSE `progress` data: `{"job_id","stage","status","progress","message","raw_ready","refined_ready","record_ready","warnings"}`,
plus `error_code`, `error_message`, `error_detail` (short technical reason, e.g. "Model 'gemma3:12b' is not
downloaded in Ollama") and `failed_stage` on failure. `JobSummary` carries `error_detail` too. A late subscriber immediately receives the latest state.

## Errors

| Code | HTTP | Message |
|---|---|---|
| `E_UNSUPPORTED_FORMAT` | 415 | This file type isn't supported. Please upload an audio file (MP3, WAV, M4A, OGG, FLAC, WEBM, AAC). |
| `E_EMPTY_FILE` | 400 | The file is empty. Please choose a recording that contains audio. |
| `E_TOO_LARGE` | 413 | The file is too large. The maximum size is {MAX_FILE_MB} MB. |
| `E_TOO_LONG` | 413 | The recording is too long. The maximum length is {MAX_DURATION_MIN} minutes. |
| `E_UNREADABLE` | 422 | The file could not be read. It may be corrupt or have no audio track. |
| `E_NO_SPEECH` | 422 | No speech was detected in this recording. |
| `E_STT_FAILED` | 500 | Transcription failed. Please try again. |
| `E_LM1_FAILED` | 500 | Transcript refinement failed. The raw transcript is still available. |
| `E_LM2_FAILED` | 500 | Writing the meeting record failed. Both transcripts are still available. |
| `E_RENDER_FAILED` | 500 | The record was created but the downloads could not be generated. |
| `E_BUSY` | 429 | The server is busy with other uploads. Please try again in a moment. |
| `E_NOT_FOUND` | 404 | This meeting could not be found. |
| `E_JOB_RUNNING` | 409 | This meeting is still being processed. Try again when it has finished. |
| `E_FFMPEG_MISSING` | 500 | ffmpeg is not installed on the server, so the audio can't be converted. Install ffmpeg and restart the backend. |
| `E_CANCELLED` | — | Processing was cancelled. |
| `E_INTERNAL` | 500 | Something unexpected went wrong. Please try again. |
| `W_NON_ENGLISH` | — | (warning) This recording may not be in English; results may be less accurate. |

The first three come back synchronously from `POST /api/jobs` as `{"code","message"}`; later ones arrive through
the job status and SSE.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite needs ffmpeg but no GPU: models are injected through `Services` (see `tests/conftest.py`).
`tests/test_pipeline.py` runs the whole state machine with fake STT/LLM on real audio files, including the LM2
failure and resume paths.

## Time per stage

Filled in from `data/jobs/{id}/timings.json` after the demo run.

| Stage | Seconds (demo recording, GPU: …) |
|---|---|
| normalizing | |
| speech_check | |
| transcribing | |
| rechecking | |
| diarizing | |
| vocabulary | |
| refining | |
| guarding | |
| documenting | |
| verifying | |
| rendering | |

## Sample outputs

Put the demo recording's `record.json`, `record.md`, `record.docx` and both transcripts in `samples/` after the
demo run, so they match the demonstration.

## Deviations from the spec

- **Speech check fallback.** An energy gate runs before Silero, so silent files fail fast; without `silero-vad`
  installed the gate is used alone.
- **Segment boundaries.** Raw segments also end where a Whisper segment ends on sentence punctuation, so short
  replies like "Agreed." keep their own segment id.
- **Re-check insertions.** When Parakeet hears extra words that are only fillers ("um", "uh"), no word is disputed.
- **Verifier "no new facts".** Numbers are compared in canonical form ("15" matches "fifteen"); "I" and words
  after sentence punctuation are not treated as names. A sentence-*initial* invented name still passes, as the
  spec only checks non-initial capitals.
- **Deadline pointers** outside the task's segments (and not within ±2 segments) are flagged `pointer_invalid`.
- **Extra API errors** `E_BUSY`, `E_NOT_FOUND`, `E_JOB_RUNNING`; `DELETE` also refuses queued jobs.
- **`LM2_BASE_URL`** (optional) lets LM2 use a second server, which vLLM needs.
- **Partial results** include `refinement`, and SSE events carry `warnings` and `failed_stage`.
- **CPU defaults.** Whisper settings are `auto` (large-v3/float16/beam 5 on a GPU, large-v3-turbo/int8/beam 1 on
  CPU) instead of fixed large-v3/cuda. Ollama is called through its native `/api/chat` with the JSON schema as
  `format`, a per-call `num_ctx`, and `think: false` for Qwen3, instead of the OpenAI-compatible endpoint.
- **LM2 scratchpad** can be turned off (`LM2_SCRATCHPAD=false`) to halve generation on CPU. LM2 fields the model
  leaves out default to empty, so a missing list never fails the whole record; the verifier still drops anything
  without valid evidence.
- **Cancel and retry** endpoints, `E_CANCELLED`, `E_FFMPEG_MISSING` and the `error_detail` field are additions.
- **Low-confidence fallback.** When Parakeet is unavailable, words below `LOWCONF_DISPUTE_THRESHOLD` are marked
  disputed (with no alternative), as the plan's fallback suggests.
