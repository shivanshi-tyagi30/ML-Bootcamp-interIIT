# Trace 
## An AI Meeting Assistant

## Repository layout

- [`backend/`](backend/): FastAPI pipeline (Whisper → Parakeet re-check → LM1 Qwen → LM2 Gemma → verifier).
  See [`backend/README.md`](backend/README.md) for models, setup and the API.
- [`frontend/`](frontend/): React interface. `npm run dev` talks to the backend on port 8000;
  `npm run dev:mock` runs without one.
- [`docs/api-contract.md`](docs/api-contract.md): what the frontend relies on from the backend.

## Website (free, permanent link): Render

The hosted website runs on Render's free plan with no local models, so it fits in 512 MB:
**Whisper large-v3 on Groq** writes the transcript, **AssemblyAI** tells the speakers apart (only "who spoke
when"; the words still come from Whisper), and **Gemini** refines it and writes the minutes. All three have free
tiers without a card. The keys are stored in Render's environment settings, so visitors paste nothing.

1. Get three free API keys:
   - Gemini: [aistudio.google.com](https://aistudio.google.com) -> Get API key
   - Groq: [console.groq.com](https://console.groq.com) -> API Keys -> Create API key
   - AssemblyAI: [assemblyai.com](https://www.assemblyai.com) -> Sign up -> copy the API key from the dashboard
2. Sign in at [render.com](https://render.com) with GitHub.
3. **New -> Blueprint**, pick this repository. Render reads [`render.yaml`](render.yaml) and asks for the three
   keys; paste them and click **Apply**.
4. The first build takes about 5-10 minutes. The link is `https://trace-xxxx.onrender.com`
   (shown at the top of the service page).

Every push to `main` deploys again by itself. Good to know: a free service sleeps after 15 minutes without
visitors (the next visit takes about a minute to wake it), and its files are wiped when it sleeps or redeploys,
so download the meetings you want to keep. Recordings can be up to 90 minutes. Everyone with the link shares
the same meeting list and the same free-tier quotas.

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
