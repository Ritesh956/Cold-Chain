"""MLflow wrapper. Local file store by default; set MLFLOW_TRACKING_URI to use the Compose server."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow


@contextmanager
def start_run(experiment: str, run_name: str) -> Iterator[mlflow.ActiveRun]:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run
