"""Two-step Oyez fetcher with rate limiting, retries, and on-disk caching.

Step 1: GET https://api.oyez.org/cases/{term}/{docket}      → case metadata
Step 2: For each oral_argument_audio[].href → transcript JSON

The case endpoint does NOT contain transcript turns; those live behind the
case_media href. Both layers are cached. Rate limiter is GLOBAL across both
layers (≤ 1 req/sec end-to-end), per Oyez politeness norms.

Usage:
    python -m src.fetch_oyez --term 2014 --docket 13-604
CRISP-DM phase: Data Understanding.
Data acquisition — 2-step Oyez transcript fetch.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OYEZ_BASE = "https://api.oyez.org"
USER_AGENT = "JusticeCast/0.1 (academic; contact saurav.kanegaonkar@gmail.com)"

RAW_DIR = Path("data/raw/oyez")
CASES_DIR = RAW_DIR / "cases"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Global ≤ N req/sec limiter shared across both fetch layers."""

    def __init__(self, rate_per_sec: float = 1.0):
        self._min_interval = 1.0 / rate_per_sec
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()

    def reset(self) -> None:
        self._last = 0.0


_limiter = _RateLimiter(rate_per_sec=1.0)


class _RetriableHTTPError(requests.HTTPError):
    """5xx / 429 — retry. Distinct from 4xx terminal failures."""


class CaseNotFound(Exception):
    """Oyez returned a search-fallback list instead of a case dict.

    Happens for original-jurisdiction dockets ("128 ORIG"), application
    dockets ("21A244"), and other non-standard formats Oyez doesn't index
    under the requested key. Treated as a per-case failure — do not retry.
    """


@retry(
    retry=retry_if_exception_type(
        (_RetriableHTTPError, requests.Timeout, requests.ConnectionError)
    ),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get_json(url: str) -> dict:
    _limiter.wait()
    t0 = time.monotonic()
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    duration = time.monotonic() - t0
    if r.status_code in (429,) or r.status_code >= 500:
        logger.warning("HTTP %d for %s (%.2fs) — retriable", r.status_code, url, duration)
        raise _RetriableHTTPError(f"{r.status_code} on {url}", response=r)
    r.raise_for_status()
    logger.debug("GET %s -> %d (%.2fs)", url, r.status_code, duration)
    return r.json()


def _safe_docket(docket: str) -> str:
    return str(docket).replace("/", "_")


def fetch_case(term: int, docket: str, force: bool = False) -> dict:
    """Step 1: case metadata. Cached at oyez/cases/{term}_{docket}.json.

    Raises CaseNotFound if Oyez returns a search-fallback list rather than
    a case dict (typical for non-standard dockets like "128 ORIG").
    """
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    cache = CASES_DIR / f"{term}_{_safe_docket(docket)}.json"
    if cache.exists() and not force:
        logger.debug("CACHE HIT case %s/%s", term, docket)
        data = json.loads(cache.read_text())
    else:
        url = f"{OYEZ_BASE}/cases/{term}/{docket}"
        logger.info("CACHE MISS case %s/%s — fetching", term, docket)
        data = _get_json(url)
        if isinstance(data, list):
            raise CaseNotFound(
                f"Oyez returned a {len(data)}-entry search list for {term}/{docket} "
                f"— no exact case match (likely non-standard docket format)"
            )
        cache.write_text(json.dumps(data))

    if isinstance(data, list):
        raise CaseNotFound(
            f"Cached response for {term}/{docket} is a search list — invalid"
        )
    return data


def fetch_transcript(audio_id: int, force: bool = False) -> dict:
    """Step 2: transcript JSON for an oral_argument_audio entry."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = TRANSCRIPTS_DIR / f"{audio_id}.json"
    if cache.exists() and not force:
        logger.debug("CACHE HIT transcript %s", audio_id)
        return json.loads(cache.read_text())
    url = f"{OYEZ_BASE}/case_media/oral_argument_audio/{audio_id}"
    logger.info("CACHE MISS transcript %s — fetching", audio_id)
    data = _get_json(url)
    cache.write_text(json.dumps(data))
    return data


@dataclass
class FetchResult:
    term: int
    docket: str
    case_fetched: bool
    n_audio_sessions: int
    transcripts_fetched: int
    error: str | None = None


def fetch_case_full(term: int, docket: str, force: bool = False) -> FetchResult:
    """Step 1 + Step 2 for a single case. Never raises — errors are returned in `error`."""
    try:
        case = fetch_case(term, docket, force=force)
    except CaseNotFound as e:
        logger.warning("Skipping %s/%s: %s", term, docket, e)
        return FetchResult(term, docket, False, 0, 0, error=f"CaseNotFound: {e}")
    except Exception as e:
        logger.error("Step 1 failed for %s/%s: %s", term, docket, e)
        return FetchResult(term, docket, False, 0, 0, error=str(e))

    try:
        audios = case.get("oral_argument_audio") or []
        fetched = 0
        for entry in audios:
            audio_id = (entry or {}).get("id")
            if audio_id is None:
                continue
            try:
                fetch_transcript(audio_id, force=force)
                fetched += 1
            except Exception as e:
                logger.error(
                    "Step 2 failed for transcript %s (case %s/%s): %s",
                    audio_id, term, docket, e,
                )
        return FetchResult(term, docket, True, len(audios), fetched)
    except Exception as e:
        logger.exception("Unexpected error processing %s/%s", term, docket)
        return FetchResult(term, docket, True, 0, 0, error=f"post-fetch error: {e}")


def bulk_fetch(case_keys: Iterable[tuple[int, str]], force: bool = False) -> list[FetchResult]:
    """Sequential bulk fetch. Rate limiter handles politeness."""
    results = []
    for term, docket in case_keys:
        results.append(fetch_case_full(term, docket, force=force))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--term", type=int, required=True)
    parser.add_argument("--docket", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    res = fetch_case_full(args.term, args.docket, force=args.force)
    print(res)


if __name__ == "__main__":
    main()
