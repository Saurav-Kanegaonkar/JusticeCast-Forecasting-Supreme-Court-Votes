"""Stop C: rescue standard-format docket failures from the bulk fetch.

Many SCDB cases decided in OT_N were originally argued in OT_{N-1} or
OT_{N+1} (re-arguments). Oyez files cases under the term they were first
argued in, while SCDB indexes by decision term — so the same docket can
appear under different terms in the two systems.

Strategy per failed (term, docket):
  1. Try term - 1 (Oyez often holds the case under the prior term)
  2. Try term + 1 (less common but possible)
  3. Verify the rescued case actually matches by checking docket_number

When a rescue succeeds, write the case JSON to the cache under the ORIGINAL
SCDB (term, docket) key so `build_dataset.py` picks it up unchanged.

Outputs:
  reports/results/rescue_log.csv — one row per attempted rescue
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd

from src import fetch_oyez, fetch_scdb

LOG_PATH = Path("reports/results/rescue_log.csv")

logger = logging.getLogger(__name__)

# Standard format: NN-NNNNN with hyphens, no letters/spaces/punctuation
_STANDARD_DOCKET_RE = re.compile(r"^\d+-\d+$")


def _is_standard(docket: str) -> bool:
    return bool(_STANDARD_DOCKET_RE.match(str(docket)))


def _try_alt_term(orig_term: int, docket: str, alt_term: int) -> tuple[bool, str, dict | None]:
    """Try fetching at (alt_term, docket). Return (success, info, raw_json)."""
    try:
        case = fetch_oyez.fetch_case(alt_term, docket)
    except fetch_oyez.CaseNotFound as e:
        return False, f"alt_term={alt_term}: not found ({e.__class__.__name__})", None
    except Exception as e:
        return False, f"alt_term={alt_term}: error {e}", None

    if str(case.get("docket_number")) != str(docket):
        return False, (
            f"alt_term={alt_term}: docket mismatch "
            f"(got {case.get('docket_number')!r})"
        ), None
    return True, f"alt_term={alt_term}: matched {case.get('name','?')[:60]}", case


def _persist_at_original_key(orig_term: int, docket: str, case: dict) -> Path:
    """Write the rescued case JSON to the cache under the SCDB (term, docket)."""
    fetch_oyez.CASES_DIR.mkdir(parents=True, exist_ok=True)
    safe = str(docket).replace("/", "_")
    p = fetch_oyez.CASES_DIR / f"{orig_term}_{safe}.json"
    p.write_text(json.dumps(case))
    return p


def _fetch_audio_for_rescue(case: dict) -> int:
    """Fetch transcripts for all oral_argument_audio entries; return count."""
    audios = case.get("oral_argument_audio") or []
    fetched = 0
    for entry in audios:
        audio_id = (entry or {}).get("id")
        if audio_id is None:
            continue
        try:
            fetch_oyez.fetch_transcript(audio_id)
            fetched += 1
        except Exception as e:
            logger.error("Transcript fetch failed for audio_id=%s: %s", audio_id, e)
    return fetched


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    log_df = pd.read_csv("reports/results/bulk_fetch_log.csv")
    failures = log_df[log_df["error"].fillna("").astype(str).str.len() > 0].copy()
    candidates = failures[failures["docket"].apply(_is_standard)].copy()

    scdb = fetch_scdb.load_scdb()
    scdb["docket"] = scdb["docket"].astype(str)
    case_names = (
        scdb[["term", "docket", "caseName"]]
        .drop_duplicates(subset=["term", "docket"])
        .set_index(["term", "docket"])["caseName"]
        .to_dict()
    )

    logger.info("Attempting rescue on %d standard-format failures", len(candidates))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(LOG_PATH, "w", newline="")
    w = csv.writer(fout)
    w.writerow([
        "scdb_term", "docket", "scdb_case_name",
        "rescued", "rescue_strategy", "rescued_oyez_term",
        "n_audio_sessions", "transcripts_fetched", "notes",
    ])

    n_rescued = n_no_audio = n_failed = 0
    t_start = time.monotonic()

    for _, row in candidates.iterrows():
        orig_term = int(row["term"])
        docket = str(row["docket"])
        scdb_name = case_names.get((orig_term, docket), "")

        rescue_strategy = "none"
        rescued_term = None
        case = None

        for alt in (orig_term - 1, orig_term + 1):
            ok, info, candidate_case = _try_alt_term(orig_term, docket, alt)
            logger.info("%s/%s -> %s", orig_term, docket, info)
            if ok:
                rescue_strategy = f"term{alt - orig_term:+d}"
                rescued_term = alt
                case = candidate_case
                break

        if case is None:
            n_failed += 1
            w.writerow([orig_term, docket, scdb_name, False, "none", None,
                        0, 0, "no candidate matched on term-1 or term+1"])
            fout.flush()
            continue

        _persist_at_original_key(orig_term, docket, case)
        n_audio = len(case.get("oral_argument_audio") or [])
        n_tx = _fetch_audio_for_rescue(case) if n_audio > 0 else 0

        if n_audio == 0:
            n_no_audio += 1
            w.writerow([orig_term, docket, scdb_name, True, rescue_strategy,
                        rescued_term, 0, 0,
                        f"matched {case.get('name','?')[:60]} but no audio"])
        else:
            n_rescued += 1
            w.writerow([orig_term, docket, scdb_name, True, rescue_strategy,
                        rescued_term, n_audio, n_tx,
                        f"matched {case.get('name','?')[:60]}"])
        fout.flush()

    fout.close()
    elapsed = time.monotonic() - t_start
    logger.info(
        "Rescue done in %.0fs. matched_with_audio=%d matched_no_audio=%d failed=%d",
        elapsed, n_rescued, n_no_audio, n_failed,
    )


if __name__ == "__main__":
    main()
