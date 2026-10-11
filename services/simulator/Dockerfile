# Live sensor simulator (publishes SYNTHETIC readings to NATS).
# Build context = repository root:  docker build -f services/simulator/Dockerfile .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY libs/coldchain_schemas/pyproject.toml libs/coldchain_schemas/pyproject.toml
COPY services/simulator/pyproject.toml services/simulator/pyproject.toml
COPY services/ml-api/pyproject.toml services/ml-api/pyproject.toml
COPY services/core/pyproject.toml services/core/pyproject.toml
COPY ml/pyproject.toml ml/pyproject.toml
RUN mkdir -p libs/coldchain_schemas/src/coldchain_schemas services/simulator/src/coldchain_simulator \
    services/ml-api/src/coldchain_ml_api services/core/src/coldchain_core ml/src/coldchain_ml \
 && touch libs/coldchain_schemas/src/coldchain_schemas/__init__.py \
    services/simulator/src/coldchain_simulator/__init__.py \
    services/ml-api/src/coldchain_ml_api/__init__.py \
    services/core/src/coldchain_core/__init__.py ml/src/coldchain_ml/__init__.py \
 && uv sync --frozen --no-dev --package coldchain-simulator --extra live --no-editable

COPY libs libs
COPY services/simulator/src services/simulator/src
RUN uv sync --frozen --no-dev --package coldchain-simulator --extra live --no-editable

FROM python:3.12-slim
RUN useradd --system --uid 10001 app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 NATS_URL=nats://nats:4222 SPEEDUP=60
USER app
WORKDIR /home/app
# 60x: one simulated minute per real second. SEED / DAYS can be overridden.
CMD ["sh", "-c", "python -m coldchain_simulator run --nats $NATS_URL --speedup $SPEEDUP"]
