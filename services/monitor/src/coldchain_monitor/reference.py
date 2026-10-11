"""Reference profiles: what each input looked like in the data the models were built on.

`reference.json` ships inside the package. Rebuild it from the (SYNTHETIC) training readings with
    python -m coldchain_monitor.reference --readings data/synthetic/freezer/readings.parquet
It includes the fault episodes that are in the training data, on purpose: the reference is
"normal operation as the models saw it", faults included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coldchain_monitor.drift import Profile, make_profile

DEFAULT_PATH = Path(__file__).with_name("reference.json")


def load(path: Path | None = None) -> dict[str, Profile]:
    raw = json.loads((path or DEFAULT_PATH).read_text())
    return {d["feature"]: Profile.from_dict(d) for d in raw["profiles"]}


def build(readings_path: Path, out: Path) -> None:
    import pandas as pd  # dev-time only; the service image does not need it

    df = pd.read_parquet(readings_path)
    profiles = [
        make_profile("temp_c", df["temp_c"].dropna().to_numpy()),
        make_profile("compressor_current_a", df["compressor_current_a"].to_numpy()),
        # a boolean: two bins, "off" and "on"
        make_profile("compressor_on", df["compressor_on"].astype(float).to_numpy(), n_bins=2),
    ]
    out.write_text(
        json.dumps(
            {
                "source": readings_path.as_posix(),
                "note": "SYNTHETIC training readings; includes injected fault episodes",
                "profiles": [p.to_dict() for p in profiles],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    a = ap.parse_args()
    build(a.readings, a.out)
    print(f"wrote {a.out}")
