"""Tests for src.build_modeling_table — Phase 2 cleanup."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import build_modeling_table


def test_is_original_jurisdiction_patterns():
    assert build_modeling_table.is_original_jurisdiction("128 ORIG")
    assert build_modeling_table.is_original_jurisdiction("8 ORIG")
    assert build_modeling_table.is_original_jurisdiction("138, Orig.")
    assert build_modeling_table.is_original_jurisdiction("No. 137, Orig.")
    assert build_modeling_table.is_original_jurisdiction("22O141")
    assert build_modeling_table.is_original_jurisdiction("22O65")
    # standard dockets must NOT match
    assert not build_modeling_table.is_original_jurisdiction("13-604")
    assert not build_modeling_table.is_original_jurisdiction("08-205")
    assert not build_modeling_table.is_original_jurisdiction("21A244")


def test_modeling_table_drops_nan_label_rows():
    if not build_modeling_table.OUT_PATH.exists():
        pytest.skip("modeling_table.parquet not built — run `python -m src.build_modeling_table`")
    df = pd.read_parquet(build_modeling_table.OUT_PATH)
    assert df["voted_petitioner"].notna().all(), (
        "modeling_table contains NaN voted_petitioner — cleanup didn't drop them"
    )


def test_modeling_table_drops_low_word_count_rows():
    if not build_modeling_table.OUT_PATH.exists():
        pytest.skip("modeling_table.parquet not built")
    df = pd.read_parquet(build_modeling_table.OUT_PATH)
    floor = build_modeling_table.WORD_COUNT_FLOOR_DEFAULT
    assert (df["word_count"] >= floor).all(), (
        f"modeling_table has rows with word_count < {floor}"
    )


def test_modeling_table_drops_original_jurisdiction():
    if not build_modeling_table.OUT_PATH.exists():
        pytest.skip("modeling_table.parquet not built")
    df = pd.read_parquet(build_modeling_table.OUT_PATH)
    bad = df["docket"].apply(build_modeling_table.is_original_jurisdiction)
    assert not bad.any(), (
        f"modeling_table has {bad.sum()} original-jurisdiction rows"
    )


def test_modeling_table_label_dtypes_are_int8():
    if not build_modeling_table.OUT_PATH.exists():
        pytest.skip("modeling_table.parquet not built")
    df = pd.read_parquet(build_modeling_table.OUT_PATH)
    assert str(df["voted_petitioner"].dtype) == "int8"
    assert str(df["unanimous"].dtype) == "int8"


def test_modeling_table_audit_csv_exists_and_decreases_monotonically():
    if not build_modeling_table.AUDIT_PATH.exists():
        pytest.skip("audit CSV not built")
    audit = pd.read_csv(build_modeling_table.AUDIT_PATH)
    assert (audit["rows_dropped"] >= 0).all(), "Cleanup should never add rows"
    assert audit["rows_out"].is_monotonic_decreasing, (
        "Each cleanup stage must produce ≤ the previous stage's row count"
    )
