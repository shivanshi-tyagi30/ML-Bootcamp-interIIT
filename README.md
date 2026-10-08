# Trace 
## An AI Meeting Assistant

## Repository layout

- [`backend/`](backend/): FastAPI pipeline (Whisper → Parakeet re-check → LM1 Qwen → LM2 Gemma → verifier).
  See [`backend/README.md`](backend/README.md) for models, setup and the API.
- [`frontend/`](frontend/): React interface. `npm run dev` talks to the backend on port 8000;
  `npm run dev:mock` runs without one.
- [`docs/api-contract.md`](docs/api-contract.md): what the frontend relies on from the backend.

## Run on Google Colab (free GPU, no install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivanshi-tyagi30/ML-Bootcamp-interIIT/blob/main/colab/Trace_on_Colab.ipynb)

Open the notebook, choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Setup takes
about 10-15 minutes; the last cell prints a public link to the full app running on Colab's GPU
(Whisper large-v3, qwen3:8b, gemma3:12b). Keep the tab open while using it.

## Quick start

```bash
# Backend (GPU machine, Ollama running): download every model once, then start
cd backend && pip install -r requirements.txt && python -m app.prefetch && uvicorn app.main:app --port 8000

# Windows laptop without a GPU: see "Windows laptop, CPU only" in backend/README.md
#   (pip install -r requirements-windows.txt, copy .env.cpu .env, python -m app.prefetch)

# Frontend
cd frontend && npm install && npm run dev   # open http://localhost:5173
```
