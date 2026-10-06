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
# Backend (GPU machine, Ollama running with qwen3:14b and gemma3:27b)
cd backend && pip install -r requirements.txt && uvicorn app.main:app --port 8000

# Frontend
cd frontend && npm install && npm run dev   # open http://localhost:5173
```
