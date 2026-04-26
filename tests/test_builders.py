"""Tests for src.build_dataset — label derivation, parser, multi-audio aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import build_dataset, fetch_scdb

HEIEN_TERM = 2014
HEIEN_DOCKET = "13-604"
HEIEN_AUDIO_ID = 23272


def test_derive_voted_petitioner_truth_table():
    # petitioner won AND in majority → voted with petitioner
    assert build_dataset.derive_voted_petitioner(1, 2) == 1
    # petitioner won AND in dissent → voted against petitioner
    assert build_dataset.derive_voted_petitioner(1, 1) == 0
    # petitioner lost AND in majority → voted against petitioner
    assert build_dataset.derive_voted_petitioner(0, 2) == 0
    # petitioner lost AND in dissent → voted with petitioner (the loser)
    assert build_dataset.derive_voted_petitioner(0, 1) == 1
    # unclear winner → undefined
    assert build_dataset.derive_voted_petitioner(2, 2) is None
    # NaN inputs → undefined
    assert build_dataset.derive_voted_petitioner(float("nan"), 2) is None
    assert build_dataset.derive_voted_petitioner(1, float("nan")) is None


def test_parser_filters_to_justices_only():
    rows = build_dataset.parse_transcript_turns(HEIEN_AUDIO_ID)
    assert len(rows) > 0
    advocates = {"jeffrey_l_fisher", "robert_c_montgomery", "rachel_p_kovner"}
    speakers = {r["oyez_identifier"] for r in rows}
    assert advocates.isdisjoint(speakers), (
        f"Advocate(s) leaked into parsed Justice utterances: {speakers & advocates}"
    )
    assert "sonia_sotomayor" in speakers, "Sotomayor missing from parsed Heien turns"


def test_heien_label_spotcheck_end_to_end():
    """The mandatory gate: Sotomayor=1, all other spoken Justices=0."""
    df = build_dataset.build_dataset()
    assert not df.empty, "build_dataset returned empty DataFrame"

    heien = df[(df["term"] == HEIEN_TERM) & (df["docket"] == HEIEN_DOCKET)].copy()
    assert not heien.empty, f"No Heien rows in build_dataset output"

    sotomayor = heien[heien["oyez_identifier"] == "sonia_sotomayor"]
    assert len(sotomayor) == 1
    assert int(sotomayor["voted_petitioner"].iloc[0]) == 1, (
        "Sotomayor expected voted_petitioner=1 (lone dissent in 8-1 Heien decision)"
    )

    others = heien[heien["oyez_identifier"] != "sonia_sotomayor"]
    assert (others["voted_petitioner"] == 0).all(), (
        f"All non-Sotomayor Justices should be 0 for Heien, got "
        f"{others[['oyez_identifier','voted_petitioner']].to_dict('records')}"
    )

    assert (heien["unanimous"] == 0).all(), "Heien (8-1) is not unanimous"


def test_every_oyez_justice_in_heien_maps_to_scdb():
    """No Justice utterance row should be left without a SCDB join."""
    df = build_dataset.build_dataset()
    heien = df[(df["term"] == HEIEN_TERM) & (df["docket"] == HEIEN_DOCKET)]
    assert heien["caseId"].notna().all(), (
        f"Unjoined Justices in Heien: "
        f"{heien[heien['caseId'].isna()]['oyez_identifier'].tolist()}"
    )


def test_multi_audio_aggregation_concatenates():
    """Synthetic test: two audio sessions for one (case, Justice) → text concatenated."""
    utts = [
        {"oyez_identifier": "elena_kagan", "oyez_speaker_name": "Elena Kagan",
         "audio_id": 100, "text": "first session question"},
        {"oyez_identifier": "elena_kagan", "oyez_speaker_name": "Elena Kagan",
         "audio_id": 100, "text": "first session followup"},
        {"oyez_identifier": "elena_kagan", "oyez_speaker_name": "Elena Kagan",
         "audio_id": 200, "text": "second session question"},
    ]
    agg = build_dataset._aggregate_per_justice(utts, term=2020, docket="20-1234", n_audio=2)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["turn_count"] == 3
    assert row["n_audio_sessions"] == 2
    assert "first session question" in row["text"]
    assert "second session question" in row["text"]
    assert row["word_count"] == len(row["text"].split())
