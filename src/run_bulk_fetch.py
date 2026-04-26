"""Bulk-fetch driver for Phase 1 Stop B.

Reads SCDB, filters to the 2005–2024 term window, dedupes (term, docket)
pairs, and runs `fetch_oyez.fetch_case_full` on each one sequentially. The
fetcher's global rate limiter handles politeness; this driver just iterates
and logs progress.

Outputs:
- data/raw/oyez/cases/{term}_{docket}.json        (Step 1 cache)
- data/raw/oyez/transcripts/{audio_id}.json       (Step 2 cache)
- reports/results/bulk_fetch_log.csv              (per-case fetch result)

Usage:
    python -m src.run_bulk_fetch                  # 2005–2024 window
    python -m src.run_bulk_fetch --start-term 2018 --end-term 2024
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from src import fetch_oyez, fetch_scdb

LOG_PATH = Path("reports/results/bulk_fetch_log.csv")

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-term", type=int, default=2005)
    parser.add_argument("--end-term", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: stop after N cases (for partial runs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scdb = fetch_scdb.load_scdb()
    window = scdb[(scdb.term >= args.start_term) & (scdb.term <= args.end_term)]

    case_keys = sorted(
        {(int(t), str(d)) for t, d in window[["term", "docket"]].itertuples(index=False)}
    )
    if args.limit:
        case_keys = case_keys[:args.limit]

    total = len(case_keys)
    logger.info("Bulk-fetching %d unique (term, docket) pairs for %d–%d",
                total, args.start_term, args.end_term)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(LOG_PATH, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["term", "docket", "case_fetched", "n_audio_sessions",
                     "transcripts_fetched", "elapsed_sec", "error"])

    t_start = time.monotonic()
    n_success = n_no_audio = n_failed = 0
    n_transcripts = 0

    for i, (term, docket) in enumerate(case_keys, start=1):
        t_case = time.monotonic()
        try:
            result = fetch_oyez.fetch_case_full(term, docket)
        except Exception as e:
            logger.exception("Unhandled exception for %s/%s — continuing", term, docket)
            result = fetch_oyez.FetchResult(term, docket, False, 0, 0,
                                            error=f"unhandled: {e}")
        elapsed = time.monotonic() - t_case

        writer.writerow([term, docket, result.case_fetched,
                         result.n_audio_sessions, result.transcripts_fetched,
                         f"{elapsed:.2f}", result.error or ""])
        fout.flush()

        if result.error:
            n_failed += 1
        elif result.n_audio_sessions == 0:
            n_no_audio += 1
        else:
            n_success += 1
            n_transcripts += result.transcripts_fetched

        if i % 50 == 0 or i == total:
            elapsed_total = time.monotonic() - t_start
            rate = i / elapsed_total
            eta_sec = (total - i) / rate if rate > 0 else 0
            logger.info(
                "Progress %d/%d (%.1f%%) — success=%d no_audio=%d failed=%d "
                "transcripts=%d — elapsed=%.0fs ETA=%.0fs",
                i, total, 100 * i / total,
                n_success, n_no_audio, n_failed, n_transcripts,
                elapsed_total, eta_sec,
            )

    fout.close()
    elapsed_total = time.monotonic() - t_start
    logger.info(
        "Bulk fetch DONE in %.0fs (%.1f min). "
        "success=%d no_audio=%d failed=%d transcripts=%d. "
        "Log: %s",
        elapsed_total, elapsed_total / 60,
        n_success, n_no_audio, n_failed, n_transcripts, LOG_PATH,
    )


if __name__ == "__main__":
    main()
