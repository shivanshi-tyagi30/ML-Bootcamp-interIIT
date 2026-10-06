# Trace frontend

React + TypeScript (Vite) + Tailwind v4. This is the "Provenance Lens" interface
from plan section 11.

```bash
cd frontend
npm install
npm run dev:mock   # no backend needed: replays a sample meeting
npm run dev        # talks to the FastAPI backend on :8000 via /api proxy
npm run build      # typecheck + production build into dist/
```

Mock mode is also available on any build by adding `?mock` to the URL. In
mock mode, a file name containing `nospeech` or `fail-lm2` simulates those
failures.

The backend contract is in [`../docs/api-contract.md`](../docs/api-contract.md).

## What's here

| Screen     | Features |
|------------|----------|
| Upload     | Drag and drop; type, empty and size checks in the browser (`E_UNSUPPORTED_FORMAT`, `E_EMPTY_FILE`, `E_TOO_LARGE`); optional glossary; red error card with "Try another file" |
| Processing | Live step bar driven by SSE (polling fallback); raw transcript preview as soon as it's ready; non-English warning |
| Workspace  | Fidelity scorecard and colour legend; Raw / Refined / Diff transcript (virtualized), with hover details for edits, blocked edits and disputed words; coverage dimming; record tabs (Summary, Minutes, Decisions, Action items, Not agreed); evidence chips that scroll the transcript and play from 2 s earlier; honest "Unspecified" chips with the reason; timeline with decision/task pins; speed control; light/dark mode; Space and J/K shortcuts; download menu; partial results when LM2 fails |

## Layout

```
src/
  lib/types.ts          MeetingRecord + job API types (mirror of plan §10 schema)
  lib/api.ts            HTTP client (POST /jobs, SSE events, GET job, export URLs) + mock backend
  lib/sampleRecord.ts   sample meeting covering the trap cases in plan §14
  lib/annotate.ts       splits a raw segment into edit / blocked / disputed pieces
  lib/exports.ts        client-side TXT/Markdown/JSON (mock mode; matches backend layout)
  lib/errors.ts         error codes + browser-side file checks
  lib/useAudio.ts       <audio> state hook
  components/           UploadScreen, ProcessingScreen, Workspace, TranscriptPane,
                        RecordPane, PlayerBar, Scorecard, DownloadMenu, ui
```

Not built yet (plan "Could 6"): a real waveform (wavesurfer.js). The timeline
already shows pins on a plain progress track.
