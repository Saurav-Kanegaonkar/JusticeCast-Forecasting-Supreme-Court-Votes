"""Phase 2: build the final modeling table from `justice_case_rows.parquet`.

Applies the cleanup decisions documented in `project-state.md`:
  1. Drop rows with no derivable binary label (partyWinning ∈ {2, NaN} or
     majority NaN). This sweeps up the 151 NaN-label rows from Phase 1
     including the 45 unmatched (case, Justice) rows from OT2015.
  2. Drop rows from original-jurisdiction cases (docket like "* ORIG", "22O*",
     "*, Orig.", or matching SCDB jurisdiction codes for original jurisdiction).
     Belt-and-suspenders: most ORIG cases didn't return Oyez transcripts
     anyway, but a handful might slip through.
  3. Drop rows where `word_count < WORD_COUNT_FLOOR` (default 30) — these are
     truncated half-utterances ("What --", "Counsel --") with no stance signal.

Inputs:  data/processed/justice_case_rows.parquet
Outputs: data/processed/modeling_table.parquet
         reports/results/modeling_table_audit.csv (per-stage row counts)

Usage:
    python -m src.build_modeling_table
    python -m src.build_modeling_table --word-count-floor 50
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import pandas as pd

from src.text_clean import preprocess_text

IN_PATH = Path("data/processed/justice_case_rows.parquet")
OUT_PATH = Path("data/processed/modeling_table.parquet")
AUDIT_PATH = Path("reports/results/modeling_table_audit.csv")

WORD_COUNT_FLOOR_DEFAULT = 30

# Original-jurisdiction docket patterns: "128 ORIG", "8 ORIG", "138, Orig.",
# "No. 137, Orig.", "22O141", "22O65", etc.
_ORIG_RE = re.compile(r"\b(orig|22O\d+)\b", re.I)

logger = logging.getLogger(__name__)


def is_original_jurisdiction(docket: str) -> bool:
    return bool(_ORIG_RE.search(str(docket)))


def build_modeling_table(word_count_floor: int = WORD_COUNT_FLOOR_DEFAULT) -> pd.DataFrame:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"{IN_PATH} missing — run `python -m src.build_dataset` first."
        )
    df = pd.read_parquet(IN_PATH)
    audit: list[tuple[str, int, int]] = []

    def _step(name: str, after: pd.DataFrame) -> None:
        audit.append((name, len(df) if not audit else audit[-1][2], len(after)))

    _step("input (justice_case_rows.parquet)", df)

    # 1. Drop rows with no derivable label
    labeled = df[df["voted_petitioner"].notna()].copy()
    _step("after drop NaN-label rows", labeled)

    # 2. Drop original-jurisdiction cases
    no_orig = labeled[~labeled["docket"].apply(is_original_jurisdiction)].copy()
    _step("after drop original-jurisdiction cases", no_orig)

    # 3. Drop low-word-count rows (uses pre-cleanup word_count from Phase 1
    #    parquet; rough enough since preprocessing only strips bracket
    #    annotations which are typically 1-2 tokens each)
    final = no_orig[no_orig["word_count"] >= word_count_floor].copy()
    _step(f"after drop word_count < {word_count_floor}", final)

    # 4. Apply preprocess_text to the modeling-table text column. Strips
    #    transcription artifacts ([Laughter], [Crosstalk], etc.) so they
    #    don't become Phase 3 vocabulary tokens. Idempotent.
    final["text"] = final["text"].fillna("").map(preprocess_text)
    # Recompute word_count on the cleaned text so downstream code sees the
    # post-preprocessing length.
    final["word_count"] = final["text"].str.split().str.len()

    final["voted_petitioner"] = final["voted_petitioner"].astype("int8")
    final["unanimous"] = final["unanimous"].astype("int8")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "rows_in", "rows_out", "rows_dropped"])
        for name, inn, out in audit:
            w.writerow([name, inn, out, inn - out])

    logger.info("Wrote %s — %d rows × %d cols", OUT_PATH, *final.shape)
    logger.info("Audit:")
    for name, inn, out in audit:
        logger.info("  %-50s  %6d -> %6d  (Δ %+d)", name, inn, out, out - inn)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-count-floor", type=int, default=WORD_COUNT_FLOOR_DEFAULT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    df = build_modeling_table(word_count_floor=args.word_count_floor)
    print()
    print(f"Final modeling table: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Distinct cases:  {df['caseId'].nunique():,}")
    print(f"Petitioner-win rate: {(df['voted_petitioner'] == 1).mean():.1%}")
    print(f"Unanimous-case rows: {(df['unanimous'] == 1).sum():,} "
          f"({(df['unanimous'] == 1).mean():.1%})")


if __name__ == "__main__":
    main()
