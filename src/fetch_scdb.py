"""Download and cache the SCDB Justice-Centered Citation file (release 2025_01).

SCDB serves over HTTP only (HTTPS is misconfigured at scdb.wustl.edu, but the
data is public read-only — not a security concern). The CSV is Latin-1 /
Windows-1252 encoded; UTF-8 read produces mojibake (e.g., `Â§`).
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

SCDB_URL = (
    "http://scdb.wustl.edu/_brickFiles/2025_01/"
    "SCDB_2025_01_justiceCentered_Citation.csv.zip"
)
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "scdb_justice.csv.zip"
CSV_PATH = RAW_DIR / "SCDB_2025_01_justiceCentered_Citation.csv"

USER_AGENT = "JusticeCast/0.1 (academic; contact saurav.kanegaonkar@gmail.com)"

logger = logging.getLogger(__name__)


def download_scdb(force: bool = False) -> Path:
    """Download SCDB zip and extract the CSV. Idempotent. Returns CSV path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not force:
        logger.info("SCDB zip cached at %s — skipping download", ZIP_PATH)
    else:
        logger.info("Downloading SCDB from %s", SCDB_URL)
        r = requests.get(SCDB_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        ZIP_PATH.write_bytes(r.content)
        logger.info("Saved %d bytes to %s", len(r.content), ZIP_PATH)

    if CSV_PATH.exists() and not force:
        logger.info("SCDB CSV already extracted at %s", CSV_PATH)
    else:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
        logger.info("Extracted CSV to %s", CSV_PATH)

    return CSV_PATH


def load_scdb(force: bool = False) -> pd.DataFrame:
    """Load SCDB Justice-Centered file as a DataFrame.

    Always reads with `encoding='latin1'` — SCDB ships in Windows-1252.
    """
    csv_path = download_scdb(force=force)
    return pd.read_csv(csv_path, encoding="latin1", low_memory=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    df = load_scdb()
    print(f"SCDB loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Term range: {df['term'].min()} – {df['term'].max()}")
