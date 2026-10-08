# Trace as one website on a free Hugging Face Space (Docker SDK, CPU basic: 2 vCPU, 16 GB RAM).
# The Space holds only this file and README.md: the code is downloaded from GitHub at build time, so
# "Settings -> Factory rebuild" in the Space deploys the latest main branch.
# Whisper and speaker labels run on the Space's CPU; the minutes are written by Gemini with each user's key.
# Local test: docker build -t trace -f deploy/huggingface/Dockerfile . && docker run -p 7860:7860 trace

ARG REPO=shivanshi-tyagi30/ML-Bootcamp-interIIT
ARG BRANCH=main

# 1. The code (GitHub's archive of the branch; Python's standard library is all it needs).
FROM python:3.11-slim AS src
ARG REPO
ARG BRANCH
RUN python -c "import io, sys, tarfile, urllib.request; \
url = 'https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH'; \
data = urllib.request.urlopen(url, timeout=120).read(); \
tarfile.open(fileobj=io.BytesIO(data)).extractall('/tmp/src'); print('downloaded', url)" \
    && mv /tmp/src/* /src

# 2. The frontend.
FROM node:22-slim AS web
WORKDIR /web
COPY --from=src /src/frontend/ ./
RUN npm ci --no-audit --no-fund && npm run build

# 3. The server.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libmagic1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user --from=src /src/backend/requirements-dev.txt /src/backend/requirements-space.txt backend/
# CPU-only PyTorch (much smaller than the default CUDA build).
RUN pip install --no-cache-dir --user torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r backend/requirements-space.txt

COPY --chown=user --from=src /src/backend/ backend/
COPY --chown=user --from=web /web/dist frontend/dist

# Settings for the hosted website (any of them can be overridden in the Space: Settings -> Variables).
ENV SERVE_FRONTEND=true \
    LOCAL_LLM_ENABLED=false \
    DATA_DIR=/home/user/app/backend/data \
    WHISPER_MODEL=large-v3-turbo WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8 WHISPER_CPU_THREADS=2 \
    OMP_NUM_THREADS=2 \
    RECHECK_ENABLED=false \
    DIARIZATION_ENABLED=true DIARIZATION_BACKEND=ecapa \
    SPEAKER_NAMES_FROM_TRANSCRIPT=true \
    WARMUP_ON_START=true

WORKDIR /home/user/app/backend
# Download every model into the image now, so no visitor waits for a download.
RUN python -m app.prefetch

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
