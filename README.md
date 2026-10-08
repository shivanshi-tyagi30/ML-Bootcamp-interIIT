# Trace 
## An AI Meeting Assistant

## Repository layout

- [`backend/`](backend/): FastAPI pipeline (Whisper → Parakeet re-check → LM1 Qwen → LM2 Gemma → verifier).
  See [`backend/README.md`](backend/README.md) for models, setup and the API.
- [`frontend/`](frontend/): React interface. `npm run dev` talks to the backend on port 8000;
  `npm run dev:mock` runs without one.
- [`docs/api-contract.md`](docs/api-contract.md): what the frontend relies on from the backend.

## Website (free, permanent link): Hugging Face Space

The whole app runs as one website on a free Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM). Whisper and
speaker labels run on the Space; each visitor pastes their own free Gemini API key, which writes the minutes.

1. Sign up at [huggingface.co](https://huggingface.co) (free, no card).
2. **New Space**: any name (e.g. `trace`), **SDK: Docker**, template **Blank**, hardware **CPU basic (free)**,
   visibility **Public**.
3. In the Space, open **Files -> Add file -> Upload files** and upload the two files from
   [`deploy/huggingface/`](deploy/huggingface/): `Dockerfile` and `README.md` (replace the Space's README).
4. The Space builds by itself (about 15-20 minutes the first time; watch the **Logs** tab). When it says
   `all models loaded ... ready`, open `https://<your-username>-<space-name>.hf.space`.

To deploy a newer version after merging to `main`: in the Space, **Settings -> Factory rebuild**.
Good to know: meetings are kept until the Space restarts (free Spaces have no lasting disk); a Space with no
visitors for 48 hours goes to sleep and the next visit wakes it up in about a minute; transcription on the
free CPU takes about 1-2x the recording's length. Everyone with the link sees the same meeting list.

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
