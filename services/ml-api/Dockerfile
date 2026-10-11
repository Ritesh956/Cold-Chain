# Build context = repository root:  docker build -f services/ml-api/Dockerfile .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# workspace manifests first, so dependency layers cache across code edits
COPY pyproject.toml uv.lock ./
COPY libs/coldchain_schemas/pyproject.toml libs/coldchain_schemas/pyproject.toml
COPY services/simulator/pyproject.toml services/simulator/pyproject.toml
COPY services/ml-api/pyproject.toml services/ml-api/pyproject.toml
COPY ml/pyproject.toml ml/pyproject.toml
RUN mkdir -p libs/coldchain_schemas/src/coldchain_schemas services/simulator/src/coldchain_simulator \
    services/ml-api/src/coldchain_ml_api ml/src/coldchain_ml \
 && touch libs/coldchain_schemas/src/coldchain_schemas/__init__.py \
    services/simulator/src/coldchain_simulator/__init__.py \
    services/ml-api/src/coldchain_ml_api/__init__.py ml/src/coldchain_ml/__init__.py \
 && uv sync --frozen --no-dev --package coldchain-ml-api --no-editable

# real sources (only what the service needs; the simulator and training extras are not installed)
COPY ml/src ml/src
COPY services/ml-api/src services/ml-api/src
RUN uv sync --frozen --no-dev --package coldchain-ml-api --no-editable

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
WORKDIR /home/app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz').status == 200 else 1)"
# one worker per container: scale with replicas (or WEB_CONCURRENCY) because scoring is CPU/GIL-bound
CMD ["uvicorn", "coldchain_ml_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
