"""Download the three M5 files this project uses (about 325 MB) into data/real/m5/.

Source: Kaggle competition "M5 Forecasting - Accuracy" (Walmart sales). Not redistributed: the files are
git-ignored and every user fetches them under their own Kaggle account.

Before the first run:
  1. Create a free Kaggle account, then Settings > API > Create New Token.
  2. Open https://www.kaggle.com/competitions/m5-forecasting-accuracy and accept the competition rules
     (otherwise the download fails with HTTP 403).
  3. Put the credentials in `.env.kaggle` (git-ignored): KAGGLE_USERNAME=... and KAGGLE_KEY=...
     (a new-style token starts with KGAT_ and is handed to the Kaggle CLI as KAGGLE_API_TOKEN, which this
     script does for you).
  4. Install the CLI somewhere (it is not a project dependency):  pip install kaggle

    python scripts/download_m5.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "m5-forecasting-accuracy"
FILES = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]  # validation file is a subset
OUT = Path("data/real/m5")


def _load_env(path: Path = Path(".env.kaggle")) -> None:
    """Read KAGGLE_* lines from .env.kaggle without overriding variables already set in the environment.
    Not .env: Docker Compose hands .env to containers, which must not receive this credential."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.startswith("KAGGLE_") and value.strip():
            os.environ.setdefault(key, value.strip())


def main() -> int:
    _load_env()
    cli = shutil.which("kaggle")
    if cli is None:
        print("The Kaggle CLI is not installed: pip install kaggle", file=sys.stderr)
        return 1
    if os.environ.get("KAGGLE_KEY", "").startswith("KGAT_"):
        os.environ.setdefault("KAGGLE_API_TOKEN", os.environ["KAGGLE_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (OUT / name).exists():
            print(f"{name}: already there")
            continue
        subprocess.run([cli, "competitions", "download", COMPETITION, "-f", name, "-p", str(OUT)], check=True)
        archive = OUT / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                if any(Path(m).name != m for m in z.namelist()):
                    raise RuntimeError(f"{archive} contains paths; refusing to extract")
                z.extractall(OUT)
            archive.unlink()
        print(f"{name}: {(OUT / name).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
