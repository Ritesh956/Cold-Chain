"""Latency benchmark for the ml-api on real 1500-minute windows from a cached simulator history.

    uv run uvicorn coldchain_ml_api.app:create_app --factory --port 8081
    uv run python scripts/bench_ml_api.py --url http://127.0.0.1:8081 --requests 200 --concurrency 1

Reports client-side latency (includes JSON serialisation of the request body), so it is what a caller
such as the core service would see on the same machine.

Windows gotchas that inflated early numbers ~50x and are NOT service latency: use 127.0.0.1, not
`localhost` (IPv6 connect timeout before the IPv4 fallback), and reuse one httpx client per thread
(a new client builds an SSL context every time).
"""

from __future__ import annotations

import argparse
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HISTORY = Path("data/synthetic/anomaly_eval/seed102/readings.parquet")


def window(readings: pd.DataFrame, end: int, n: int) -> dict:
    w = readings.iloc[end - n + 1 : end + 1]
    return {
        "zone": "A",
        "return_last": 1,
        "readings": [
            {
                "ts": r.ts.isoformat(),
                "temp_c": None if pd.isna(r.temp_c) else float(r.temp_c),
                "compressor_on": bool(r.compressor_on),
                "compressor_current_a": float(r.compressor_current_a),
            }
            for r in w.itertuples()
        ],
    }


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    r = pd.read_parquet(HISTORY)
    r = r[r["zone"] == "A"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    ends = rng.integers(3000, len(r), size=args.requests)
    bodies = {"/v1/anomaly/score": [window(r, int(e), 500) for e in ends[:50]],
              "/v1/breach/predict": [window(r, int(e), 1500) for e in ends[:50]]}

    print(f"{platform.processor()} | python {platform.python_version()} | concurrency {args.concurrency}")
    with httpx.Client(base_url=args.url, timeout=30) as probe:
        probe.get("/readyz").raise_for_status()
        print("models:", {k: v["version"] for k, v in probe.get("/v1/models").json().items()})

    for path, pool in bodies.items():
        size_kb = len(httpx.Request("POST", "http://x", json=pool[0]).content) / 1024

        local = threading.local()

        def call(i: int, path: str = path, pool: list = pool, local: threading.local = local) -> float:
            if not hasattr(local, "client"):
                local.client = httpx.Client(base_url=args.url, timeout=30)
            t0 = time.perf_counter()
            resp = local.client.post(path, json=pool[i % len(pool)])
            resp.raise_for_status()
            return (time.perf_counter() - t0) * 1000

        for i in range(5):  # warm-up
            call(i)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as ex:
            lat = list(ex.map(call, range(args.requests)))
        wall = time.perf_counter() - t0
        print(
            f"{path:20s} body {size_kb:5.0f} KB | p50 {pct(lat, 50):6.1f} ms  p95 {pct(lat, 95):6.1f} ms  "
            f"p99 {pct(lat, 99):6.1f} ms  mean {statistics.mean(lat):6.1f} ms | "
            f"{args.requests / wall:5.1f} req/s"
        )


if __name__ == "__main__":
    main()
