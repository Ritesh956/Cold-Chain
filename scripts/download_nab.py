"""Download the NAB machine-temperature series and its labels, pinned to one commit.

Source: https://github.com/numenta/NAB (Lavin & Ahmad, 2015). Licence: see LICENSE.txt written
next to the data. Files are NOT committed to this repo; run this script to fetch them.

    uv run python scripts/download_nab.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
BASE = f"https://raw.githubusercontent.com/numenta/NAB/{COMMIT}"
FILES = {
    "data/realKnownCause/machine_temperature_system_failure.csv": "machine_temperature_system_failure.csv",
    "labels/combined_windows.json": "combined_windows.json",
    "labels/combined_labels.json": "combined_labels.json",
    "LICENSE.txt": "LICENSE.txt",
}
OUT = Path("data/real/nab")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        dest = OUT / local
        with urllib.request.urlopen(f"{BASE}/{remote}", timeout=60) as r:  # noqa: S310 (pinned https URL)
            data = r.read()
        dest.write_bytes(data)
        print(f"{local:48s} {len(data):>8,d} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"NAB commit {COMMIT} -> {OUT}")


if __name__ == "__main__":
    main()
