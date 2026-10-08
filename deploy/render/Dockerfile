# Trace as one website on Render's free plan: the built frontend and the API on one port.
# No local models: Whisper large-v3 runs on Groq, speaker labels come from AssemblyAI, Gemini writes the minutes.
# Render builds this from the repository (see render.yaml at the repository root).

# 1. The frontend.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# 2. The server.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libmagic1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements-dev.txt backend/requirements-render.txt backend/
RUN pip install --no-cache-dir -r backend/requirements-render.txt
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist

# The hosted website's settings (each can be overridden in Render: Environment).
ENV PYTHONUNBUFFERED=1 \
    SERVE_FRONTEND=true \
    DATA_DIR=/app/backend/data \
    STT_BACKEND=groq \
    DIARIZATION_ENABLED=true DIARIZATION_BACKEND=assemblyai \
    RECHECK_ENABLED=false \
    LOCAL_LLM_ENABLED=false \
    SPEAKER_NAMES_FROM_TRANSCRIPT=true \
    WARMUP_ON_START=true \
    MAX_DURATION_MIN=90 \
    MAX_CONCURRENT_UPLOADS=2

WORKDIR /app/backend
# Render sets PORT.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
