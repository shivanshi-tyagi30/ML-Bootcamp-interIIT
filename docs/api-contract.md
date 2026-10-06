# Backend API contract (what the frontend expects)

The frontend in `frontend/` is built against this contract. Field names follow
the MeetingRecord schema in plan section 10; the TypeScript mirror is
`frontend/src/lib/types.ts`. If the backend changes a field, update both.

In dev, the UI calls `/api/...` and Vite proxies that to `http://localhost:8000/...`
(override with `TRACE_BACKEND=http://host:port`). In production, set
`VITE_API_URL` to the backend's base URL. CORS must allow the UI origin.

## `POST /jobs`

`multipart/form-data` with:

| field      | required | notes                                                    |
|------------|----------|----------------------------------------------------------|
| `file`     | yes      | the audio file                                           |
| `glossary` | no       | comma-separated expected terms, passed to Whisper as `initial_prompt` |

Response `202 {"job_id": "..."}`.

Errors that can be detected at once (bad type, empty, too large) should
respond with a 4xx and this body:

```json
{"error": {"code": "E_UNSUPPORTED_FORMAT", "user_message": "This file type isn't supported. Upload MP3, WAV, M4A, OGG, FLAC or WEBM."}}
```

FastAPI's `{"detail": {"code": ..., "user_message": ...}}` is also accepted.

## `GET /jobs/{id}/events` (Server-Sent Events)

Send one `data:` line of JSON each time the stage changes:

```
data: {"stage": "transcribing"}

data: {"stage": "refining", "warnings": ["W_NON_ENGLISH"]}

data: {"stage": "failed", "error": {"code": "E_LM2_FAILED", "stage": "documenting", "user_message": "..."}}

data: {"stage": "completed"}
```

`stage` is one of: `uploaded validating normalizing speech_check transcribing
rechecking refining guarding documenting verifying rendering completed failed`.
Close the stream after `completed` or `failed`. If the stream drops, the UI
falls back to polling `GET /jobs/{id}` every 2 s.

## `GET /jobs/{id}`

```json
{
  "job_id": "...",
  "stage": "documenting",
  "error": null,
  "warnings": [],
  "record": { "meta": {...}, "raw_transcript": [...], "...": "fields filled in so far" }
}
```

`record` is the MeetingRecord, **partially filled while the job runs**:

- `raw_transcript` must be present once `rechecking` is done. The processing
  screen shows it as soon as the stage moves past `rechecking`.
- `refined_transcript` and `refinement` must be present once `guarding` is done.
- On a late failure (`E_LM1_FAILED`, `E_LM2_FAILED`), keep whatever exists. The
  UI opens the workspace with the transcripts and shows the error in place of
  the record.
- If the job fails before `raw_transcript` exists, the UI returns to the upload
  screen and shows the error card.

### Things the UI relies on

- Each `refinement.accepted[*].original` / `rejected[*].original` is an exact
  substring of that segment's **raw** text (the guard already enforces this).
  The UI builds the Diff view by finding it there.
- `rejected[*].reject_reason` uses the guard's codes: `not_found`,
  `number_changed`, `negation_changed`, `over_rewrite`, `not_sound_alike`
  (also understood: `low_confidence`, `frozen_token`).
- `raw_transcript[*].words[*]` with `disputed: true` and `alt` draws the
  dotted underline. Words must be in text order.
- `action_items[*].owner` / `deadline` are the final strings (copied text or
  `"Unspecified"`). `verifier_flags` decides the hover text on Unspecified
  chips: `audio_unclear` → "audio unclear", `pointer_invalid` → "pointer
  didn't match", otherwise "not stated in the recording".
- `fidelity.numbers_preserved` / `negations_preserved` are `"x/y"` strings.
  They show green when x = y, red otherwise.
- `meta.models` names appear in the header (the rubric gives marks for naming them).

## `GET /jobs/{id}/export?fmt=json|md|docx|txt_raw|txt_refined`

Returns the file with a `Content-Disposition: attachment` header. All formats
are generated from the same stored JSON. The Markdown layout should match
`frontend/src/lib/exports.ts` (`recordMarkdown`), which mock mode uses.

## Audio playback

The backend does not need to serve audio. The browser plays the user's own
uploaded file through an object URL, so segment `start`/`end` times must be in
seconds relative to the **original** file.
