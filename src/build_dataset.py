"""Build the joined `(case, Justice)` modeling table from cached SCDB + Oyez.

Reads:
    data/raw/SCDB_2025_01_justiceCentered_Citation.csv
    data/raw/oyez/cases/{term}_{docket}.json
    data/raw/oyez/transcripts/{audio_id}.json
    data/processed/justice_id_map.csv

Writes:
    data/processed/justice_case_rows.parquet

One row per (case_id, justice). Cases without oral argument are filtered out
(no text → no signal). Justices who appear in SCDB but didn't speak during
oral argument (e.g., Thomas in many cases) are also filtered out.

Multi-audio cases (re-arguments, etc.) concatenate the Justice's utterances
across ALL `oral_argument_audio[]` entries, with `n_audio_sessions` recorded
as metadata.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src import fetch_scdb

OYEZ_CASES_DIR = Path("data/raw/oyez/cases")
OYEZ_TRANSCRIPTS_DIR = Path("data/raw/oyez/transcripts")
JUSTICE_ID_MAP_PATH = Path("data/processed/justice_id_map.csv")
OUT_PATH = Path("data/processed/justice_case_rows.parquet")

logger = logging.getLogger(__name__)


def derive_voted_petitioner(party_winning, majority) -> int | None:
    """Return 1 if the Justice voted with the petitioner, 0 if respondent, None if undefined.

    Codebook (scdb.wustl.edu/documentation.php):
        partyWinning: 0 = no favorable disposition for petitioner (lost),
                      1 = petitioner received favorable disposition (won),
                      2 = unclear  → excluded
        majority:     1 = dissent, 2 = majority, NaN = did not participate → excluded

    Verified on Heien v. North Carolina (2014/13-604):
        Sotomayor (sole dissent, partyWinning=0, majority=1) → 1
        All other 8 majority Justices → 0
    """
    if pd.isna(party_winning) or pd.isna(majority):
        return None
    if party_winning == 2:
        return None
    return int((party_winning == 1) == (majority == 2))


def _is_justice(speaker: dict) -> bool:
    for role in speaker.get("roles") or []:
        if (role or {}).get("type") == "scotus_justice":
            return True
    return False


def parse_transcript_turns(audio_id: int) -> list[dict]:
    """Yield one row per Justice utterance from a cached transcript."""
    path = OYEZ_TRANSCRIPTS_DIR / f"{audio_id}.json"
    data = json.loads(path.read_text())
    transcript = data.get("transcript")
    if not transcript:
        return []
    rows: list[dict] = []
    for section in transcript.get("sections") or []:
        for turn in section.get("turns") or []:
            speaker = turn.get("speaker") or {}
            if not _is_justice(speaker):
                continue
            text = " ".join(
                (tb.get("text") or "").strip()
                for tb in (turn.get("text_blocks") or [])
            ).strip()
            if not text:
                continue
            rows.append(
                {
                    "oyez_identifier": speaker.get("identifier"),
                    "oyez_speaker_name": speaker.get("name"),
                    "audio_id": audio_id,
                    "text": text,
                }
            )
    return rows


def collect_case_utterances(term: int, docket: str) -> tuple[list[dict], int]:
    """All Justice utterances for one case + n_audio_sessions."""
    safe_docket = str(docket).replace("/", "_")
    case_path = OYEZ_CASES_DIR / f"{term}_{safe_docket}.json"
    if not case_path.exists():
        return [], 0
    case = json.loads(case_path.read_text())
    if not isinstance(case, dict):
        # Polluted cache entry (Oyez search-fallback list) — treat as no-audio.
        logger.debug("Skipping non-dict cached response for %s/%s", term, docket)
        return [], 0
    audios = case.get("oral_argument_audio") or []
    rows: list[dict] = []
    for entry in audios:
        audio_id = entry.get("id")
        if audio_id is None:
            continue
        if not (OYEZ_TRANSCRIPTS_DIR / f"{audio_id}.json").exists():
            logger.warning("Missing transcript cache for audio_id=%s (case %s/%s)",
                           audio_id, term, docket)
            continue
        rows.extend(parse_transcript_turns(audio_id))
    return rows, len(audios)


def _aggregate_per_justice(utterances: list[dict], term: int, docket: str, n_audio: int) -> pd.DataFrame:
    if not utterances:
        return pd.DataFrame()
    df = pd.DataFrame(utterances)
    agg = (
        df.groupby("oyez_identifier", as_index=False)
          .agg(
              text=("text", lambda s: " ".join(s)),
              turn_count=("text", "size"),
              oyez_speaker_name=("oyez_speaker_name", "first"),
          )
    )
    agg["term"] = term
    agg["docket"] = str(docket)
    agg["n_audio_sessions"] = n_audio
    agg["word_count"] = agg["text"].str.split().str.len()
    return agg


def _cached_case_keys() -> list[tuple[int, str]]:
    keys = []
    for p in OYEZ_CASES_DIR.glob("*.json"):
        term_str, _, docket = p.stem.partition("_")
        try:
            keys.append((int(term_str), docket))
        except ValueError:
            logger.warning("Skipping malformed cache filename: %s", p.name)
    return sorted(keys)


def build_dataset() -> pd.DataFrame:
    """Build the joined parquet from whatever's currently in the Oyez cache."""
    if not JUSTICE_ID_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Justice ID map missing at {JUSTICE_ID_MAP_PATH} — hand-build it first."
        )
    id_map = pd.read_csv(JUSTICE_ID_MAP_PATH)
    scdb = fetch_scdb.load_scdb()

    case_keys = _cached_case_keys()
    if not case_keys:
        logger.warning("No cached Oyez cases found at %s", OYEZ_CASES_DIR)
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    cases_with_no_audio = 0
    for term, docket in case_keys:
        utts, n_audio = collect_case_utterances(term, docket)
        if n_audio == 0:
            cases_with_no_audio += 1
            continue
        agg = _aggregate_per_justice(utts, term, docket, n_audio)
        if not agg.empty:
            chunks.append(agg)

    if not chunks:
        logger.warning("No utterances aggregated after parsing %d cached cases", len(case_keys))
        return pd.DataFrame()

    utts_df = pd.concat(chunks, ignore_index=True)
    logger.info("Aggregated %d (case, Justice) utterance rows from %d cached cases (%d had no audio)",
                len(utts_df), len(case_keys), cases_with_no_audio)

    scdb_cols = [
        "caseId", "caseName", "term", "docket", "dateDecision",
        "justice", "justiceName",
        "partyWinning", "majority", "majVotes", "minVotes",
    ]
    scdb_slim = scdb[scdb_cols].copy()
    scdb_slim["docket"] = scdb_slim["docket"].astype(str)
    scdb_slim["dateDecision_dt"] = pd.to_datetime(
        scdb_slim["dateDecision"], errors="coerce", format="mixed"
    )

    # Dedupe SCDB on (term, docket, justice). Some dockets recur as a follow-up
    # per-curiam (e.g., Medellin v. Texas 2007/06-984 has caseId 2007-026 for
    # the merits decision and 2007-073 for the later stay denial). The oral
    # argument transcript belongs to the original merits case, so we keep the
    # earliest dateDecision for each (term, docket, justice).
    pre_dedup = len(scdb_slim)
    scdb_slim = (
        scdb_slim.sort_values("dateDecision_dt", kind="stable")
                 .drop_duplicates(["term", "docket", "justice"], keep="first")
                 .drop(columns=["dateDecision_dt"])
    )
    if len(scdb_slim) < pre_dedup:
        logger.info("Deduped SCDB by (term, docket, justice): %d → %d rows",
                    pre_dedup, len(scdb_slim))

    joined = (
        utts_df.merge(id_map[["oyez_identifier", "scdb_justice_id"]],
                      on="oyez_identifier", how="left")
               .merge(scdb_slim,
                      left_on=["term", "docket", "scdb_justice_id"],
                      right_on=["term", "docket", "justice"],
                      how="left")
    )

    unmapped = joined[joined["scdb_justice_id"].isna()]
    if not unmapped.empty:
        logger.warning("Unmapped Oyez identifiers: %s",
                       sorted(unmapped["oyez_identifier"].unique().tolist()))

    no_scdb_match = joined[joined["caseId"].isna() & joined["scdb_justice_id"].notna()]
    if not no_scdb_match.empty:
        logger.warning("(term, docket, justice) tuples with no SCDB row: %d",
                       len(no_scdb_match))

    joined["unanimous"] = (joined["minVotes"] == 0).astype("Int64")
    joined["voted_petitioner"] = joined.apply(
        lambda r: derive_voted_petitioner(r["partyWinning"], r["majority"]),
        axis=1,
    ).astype("Int64")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %s — %d rows × %d cols", OUT_PATH, *joined.shape)
    return joined


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    df = build_dataset()
    if df.empty:
        print("EMPTY")
    else:
        cols = ["caseId", "caseName", "oyez_identifier", "justiceName",
                "word_count", "turn_count", "n_audio_sessions",
                "partyWinning", "majority", "unanimous", "voted_petitioner"]
        print(df[cols].to_string(index=False))
