"""Canonical train/test split for ALL modeling tracks.

Non-Negotiable #15: BoW (Phase 3 / 4) and embeddings (Phase 4.5) tracks must
evaluate on identical fold-0 test rows for apples-to-apples comparison. This
module is the single source of truth — both tracks import from here.

Non-Negotiable #1: split is grouped by case_id via StratifiedGroupKFold so
all justice-rows from a given case stay in the same fold (no leakage).

Non-Negotiables #2 / #5: stratified on the binary label, fixed
random_state=42, n_splits=5, shuffle=True.
CRISP-DM phase: Data Preparation.
Canonical fold-0 train/test split shared across both modeling tracks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MODELING_TABLE_PATH = Path("data/processed/modeling_table.parquet")
N_SPLITS = 5
RANDOM_STATE = 42
TEST_FOLD_INDEX = 0  # fold 0 is held out as the test set everywhere


@dataclass
class Split:
    """The canonical train/test split, materialized as both indices and frames."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray


def get_cv_splitter() -> StratifiedGroupKFold:
    """The CV splitter used both for the fold-0 test split AND for nested
    GridSearchCV inside Phase 4. Same n_splits, same random_state, same shuffle."""
    return StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )


def get_train_test_split(df: pd.DataFrame | None = None) -> Split:
    """Materialize the canonical fold-0-test split for the modeling table.

    If `df` is None, loads `data/processed/modeling_table.parquet`. If
    provided, expects columns `voted_petitioner` (binary, 0/1) and `caseId`.
    """
    if df is None:
        df = pd.read_parquet(MODELING_TABLE_PATH)
    df = df.reset_index(drop=True)

    y = df["voted_petitioner"].astype(int).to_numpy()
    groups = df["caseId"].to_numpy()

    splitter = get_cv_splitter()
    folds = list(splitter.split(df.index.to_numpy(), y, groups=groups))
    train_idx, test_idx = folds[TEST_FOLD_INDEX]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    split = Split(
        train_idx=train_idx,
        test_idx=test_idx,
        train_df=train_df,
        test_df=test_df,
        y_train=y[train_idx],
        y_test=y[test_idx],
        groups_train=groups[train_idx],
        groups_test=groups[test_idx],
    )
    assert_no_case_leakage(split)
    return split


def assert_no_case_leakage(split: Split) -> None:
    """Hard check: no caseId appears in both train and test. Raises on leakage."""
    overlap = set(split.groups_train) & set(split.groups_test)
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlap)} caseIds overlap between train and test. "
            f"Group split is broken. Examples: {sorted(overlap)[:5]}"
        )
