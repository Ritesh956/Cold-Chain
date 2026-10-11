# Build context = repository root:  docker build -f services/monitor/Dockerfile .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# every workspace manifest is needed for the lockfile to resolve; sources are stubbed so the
# dependency layer caches across code edits
COPY pyproject.toml uv.lock ./
COPY libs/coldchain_schemas/pyproject.toml libs/coldchain_schemas/pyproject.toml
COPY services/simulator/pyproject.toml services/simulator/pyproject.toml
COPY services/ml-api/pyproject.toml services/ml-api/pyproject.toml
COPY services/core/pyproject.toml services/core/pyproject.toml
COPY services/monitor/pyproject.toml services/monitor/pyproject.toml
COPY ml/pyproject.toml ml/pyproject.toml
RUN mkdir -p libs/coldchain_schemas/src/coldchain_schemas services/simulator/src/coldchain_simulator \
    services/ml-api/src/coldchain_ml_api services/core/src/coldchain_core \
    services/monitor/src/coldchain_monitor ml/src/coldchain_ml \
 && touch libs/coldchain_schemas/src/coldchain_schemas/__init__.py \
    services/simulator/src/coldchain_simulator/__init__.py \
    services/ml-api/src/coldchain_ml_api/__init__.py \
    services/core/src/coldchain_core/__init__.py \
    services/monitor/src/coldchain_monitor/__init__.py ml/src/coldchain_ml/__init__.py \
 && uv sync --frozen --no-dev --package coldchain-monitor --no-editable

COPY services/monitor/src services/monitor/src
RUN uv sync --frozen --no-dev --package coldchain-monitor --no-editable

FROM python:3.12-slim
RUN useradd --system --uid 10001 app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
WORKDIR /home/app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status == 200 else 1)"
CMD ["uvicorn", "coldchain_monitor.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080"]
