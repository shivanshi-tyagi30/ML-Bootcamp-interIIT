# Trace 
## An AI Meeting Assistant

## Repository layout

- [`backend/`](backend/): FastAPI pipeline (Whisper → Parakeet re-check → LM1 Qwen → LM2 Gemma → verifier).
  See [`backend/README.md`](backend/README.md) for models, setup and the API.
- [`frontend/`](frontend/): React interface. `npm run dev` talks to the backend on port 8000;
  `npm run dev:mock` runs without one.
- [`docs/api-contract.md`](docs/api-contract.md): what the frontend relies on from the backend.

## Quick start

```bash
# Backend (GPU machine, Ollama running): download every model once, then start
cd backend && pip install -r requirements.txt && python -m app.prefetch && uvicorn app.main:app --port 8000

# Windows laptop without a GPU: see "Windows laptop, CPU only" in backend/README.md
#   (pip install -r requirements-windows.txt, copy .env.cpu .env, python -m app.prefetch)

# Frontend
cd frontend && npm install && npm run dev   # open http://localhost:5173
```
