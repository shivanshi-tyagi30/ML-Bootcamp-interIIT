# Backend API contract

The backend in `backend/` implements the **TRACE Backend Build Spec v1.0**, sections 9 and 21. The full
endpoint and error tables are in [`backend/README.md`](../backend/README.md). This page lists what the frontend
relies on.

- **Base path** `/api`. In dev, Vite proxies `/api/*` unchanged to `http://localhost:8000`
  (override with `TRACE_BACKEND`). In production set `VITE_API_URL` and add the UI origin to `CORS_ORIGINS`.
- **Upload** `POST /api/jobs` (multipart `file`, `title`, `glossary`). Errors are `{"code","message"}`. A cached
  result returns `{"job_id","cached":true}` and the UI opens it directly.
- **Progress** `GET /api/jobs/{id}/events`: SSE event `progress` with `stage`, `status`, `progress` (0–1),
  `message`, `raw_ready`, `refined_ready`, `record_ready`, `warnings`; on failure `error_code`, `error_message`,
  `failed_stage`. When `raw_ready` turns true the UI fetches the job to show the raw transcript early. If the stream
  drops, the UI polls `GET /api/jobs/{id}` every 2 s.
- **Results** `GET /api/jobs/{id}` → `{"job", "record", "partial"}`. With no record (still running, or LM2 failed)
  the UI builds a partial view from `partial.raw_transcript`, `partial.refined_transcript` and `partial.refinement`.
- **History** `GET /api/jobs`, `PATCH /api/jobs/{id}` (rename), `DELETE /api/jobs/{id}` (409 while processing).
- **Audio** `GET /api/jobs/{id}/audio` with HTTP Range. Files picked in the current session play from the browser.
- **Downloads** `GET /api/jobs/{id}/export?fmt=json|md|docx|txt_raw|txt_refined`.

Things the UI depends on in the record:

- Each `refinement.accepted[*].original` / `rejected[*].original` is an exact substring of that segment's
  **raw** text; the Diff view finds it there.
- `raw_transcript[*].words[*]` with `disputed` and `alt` draw the dotted underline; words are in text order.
- `action_items[*].owner` / `deadline` are final strings; `verifier_flags` pick the hover text on
  `Unspecified` chips (`audio_unclear`, `pointer_invalid`, `self_assignment_unverified`).
- `open_proposals[*].demoted_from_decision` shows a "no clear agreement" tag.
- Segment `start`/`end` are seconds from the start of the original file.
