FROM python:3.11-slim

# CPU build of PyTorch by default. For a GPU use docker-compose.gpu.yml, which passes a CUDA wheel index here.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TQDM_DISABLE=1 ARTIFACTS_DIR=/artifacts

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY scripts ./scripts
RUN useradd -m cutis && chown -R cutis /srv
USER cutis
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fs http://localhost:8000/health | grep -q '"status":"ready"' || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
