# Thin wrappers. On Windows without make, run the commands on the right directly.
.PHONY: install test lint up down data nlp-setup

install:  ## uv sync --all-packages
	uv sync --all-packages

test:
	uv run pytest

lint:
	uv run ruff check .

up:
	docker compose --profile core up -d --build

down:
	docker compose --profile core down

data:  ## SYNTHETIC freezer history + ground truth
	uv run python -m coldchain_simulator generate --out data/synthetic/freezer

nlp-setup:  ## opt-in PyTorch group (about 1 GB); see docs/model-cards/incident-nlp.md to reproduce
	uv sync --all-packages --group nlp
