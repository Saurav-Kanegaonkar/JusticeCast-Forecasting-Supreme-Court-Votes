"""Tests for src.fetch_scdb and src.fetch_oyez.

These rely on the SCDB CSV being cached locally (Phase 0 already pulled it)
and on the Heien transcript cache being present (Stop A smoke test).
The bulk fetch from Phase 1 Stop B is NOT required to run these.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src import fetch_oyez, fetch_scdb

HEIEN_TERM = 2014
HEIEN_DOCKET = "13-604"
HEIEN_AUDIO_ID = 23272


def test_scdb_loads_with_expected_shape():
    df = fetch_scdb.load_scdb()
    assert df.shape[0] > 80_000, f"Expected > 80k SCDB vote rows, got {df.shape[0]}"
    assert df.shape[1] == 61, f"Expected 61 SCDB columns, got {df.shape[1]}"
    expected = {"caseId", "term", "docket", "partyWinning", "majority",
                "majVotes", "minVotes", "justice", "justiceName"}
    assert expected.issubset(df.columns)


def test_scdb_encoding_is_latin1():
    df = fetch_scdb.load_scdb()
    sample = df.loc[df["lawMinor"].astype(str).str.contains("§", na=False), "lawMinor"]
    assert len(sample) > 0, (
        "Section symbol § not found — file was likely read with the wrong encoding"
    )


def test_oyez_step1_heien_shape():
    case = fetch_oyez.fetch_case(HEIEN_TERM, HEIEN_DOCKET)
    assert isinstance(case, dict)
    assert case.get("docket_number") == HEIEN_DOCKET
    assert str(case.get("term")) == str(HEIEN_TERM)
    audios = case.get("oral_argument_audio")
    assert isinstance(audios, list) and len(audios) == 1
    assert audios[0].get("id") == HEIEN_AUDIO_ID


def test_oyez_step2_heien_transcript_shape():
    data = fetch_oyez.fetch_transcript(HEIEN_AUDIO_ID)
    transcript = data.get("transcript")
    assert transcript is not None
    sections = transcript.get("sections")
    assert isinstance(sections, list) and len(sections) > 0

    justices_seen = set()
    for sect in sections:
        for turn in sect.get("turns", []) or []:
            speaker = turn.get("speaker") or {}
            for role in speaker.get("roles") or []:
                if (role or {}).get("type") == "scotus_justice":
                    justices_seen.add(speaker.get("identifier"))
    assert "sonia_sotomayor" in justices_seen, "Sotomayor missing from Heien transcript"
    assert len(justices_seen) >= 8, (
        f"Expected ≥ 8 Justices in Heien (Thomas silent), got {len(justices_seen)}"
    )


def test_rate_limiter_enforces_minimum_interval():
    limiter = fetch_oyez._RateLimiter(rate_per_sec=4.0)
    limiter.reset()
    t0 = time.monotonic()
    for _ in range(4):
        limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.75, (
        f"4 calls at 4 req/sec should take ≥ 0.75s; took {elapsed:.3f}s"
    )


def test_oyez_cache_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_oyez, "CASES_DIR", tmp_path / "cases")
    monkeypatch.setattr(fetch_oyez, "TRANSCRIPTS_DIR", tmp_path / "transcripts")

    cache_path = tmp_path / "cases" / f"{HEIEN_TERM}_{HEIEN_DOCKET}.json"
    cache_path.parent.mkdir(parents=True)
    sentinel = {"docket_number": HEIEN_DOCKET, "term": str(HEIEN_TERM),
                "oral_argument_audio": [], "_sentinel": "from-cache"}
    cache_path.write_text(json.dumps(sentinel))

    case = fetch_oyez.fetch_case(HEIEN_TERM, HEIEN_DOCKET)
    assert case.get("_sentinel") == "from-cache"
