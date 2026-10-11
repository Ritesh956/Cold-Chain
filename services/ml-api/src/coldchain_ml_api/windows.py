"""Turn a request into the DataFrame the shared feature code expects, with strict validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from coldchain_ml.contract import STEP_MIN

from coldchain_ml_api.errors import InputError
from coldchain_ml_api.schemas import WindowRequest


def to_frame(req: WindowRequest, min_history: int) -> pd.DataFrame:
    n = len(req.readings)
    needed = min_history + req.return_last - 1
    if n < needed:
        raise InputError(
            "insufficient_history",
            f"need at least {needed} readings ({min_history} of history for the first scored minute, "
            f"return_last={req.return_last}), got {n}",
        )
    # Column-wise construction: ~10x faster than model_dump() per reading on a 1500-row window.
    rs = req.readings
    ts = pd.to_datetime([r.ts for r in rs], utc=True)
    if n > 1 and not ((ts[1:] - ts[:-1]) == pd.Timedelta(minutes=STEP_MIN)).all():
        raise InputError(
            "irregular_timestamps",
            "readings must be consecutive 1-minute samples, oldest first; send gaps as temp_c=null",
        )
    return pd.DataFrame(
        {
            "ts": ts,
            "zone": req.zone,
            "temp_c": np.array([np.nan if r.temp_c is None else r.temp_c for r in rs], dtype=float),
            "compressor_on": np.array([r.compressor_on for r in rs], dtype=bool),
            "compressor_current_a": np.array([r.compressor_current_a for r in rs], dtype=float),
        }
    )
