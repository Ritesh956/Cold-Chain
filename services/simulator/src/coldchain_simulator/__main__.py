"""CLI: `python -m coldchain_simulator generate|run`."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from coldchain_simulator.config import SimConfig
from coldchain_simulator.generate import write_dataset


def main() -> None:
    p = argparse.ArgumentParser(prog="coldchain_simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic history + ground truth to parquet")
    g.add_argument("--out", type=Path, default=Path("data/synthetic/freezer"))
    g.add_argument("--seed", type=int, default=SimConfig.seed)
    g.add_argument("--days", type=int, default=SimConfig.days)

    r = sub.add_parser("run", help="publish readings to NATS at an accelerated clock")
    r.add_argument("--nats", default="nats://localhost:4222")
    r.add_argument("--speedup", type=float, default=60.0, help="1 = real time, 60 = 1 sim-hour/min")
    r.add_argument("--seed", type=int, default=SimConfig.seed)
    r.add_argument("--days", type=int, default=SimConfig.days)

    args = p.parse_args()
    cfg = replace(SimConfig(), seed=args.seed, days=args.days)

    if args.cmd == "generate":
        r_path, e_path = write_dataset(cfg, args.out)
        print(f"wrote {r_path} and {e_path} (SYNTHETIC data, seed={cfg.seed})")
    else:
        from coldchain_simulator.publish import run

        asyncio.run(run(cfg, args.nats, args.speedup))


if __name__ == "__main__":
    main()
