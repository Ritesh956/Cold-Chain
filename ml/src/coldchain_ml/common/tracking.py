"""MLflow wrapper. Local file store by default; set MLFLOW_TRACKING_URI to use the Compose server."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow


def _utf8_console() -> None:
    """MLflow prints an emoji line when it talks to a tracking server over HTTP; a cp1252 Windows console
    then raises UnicodeEncodeError at the end of the run, after the model is already registered."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


@contextmanager
def start_run(experiment: str, run_name: str) -> Iterator[mlflow.ActiveRun]:
    _utf8_console()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run
