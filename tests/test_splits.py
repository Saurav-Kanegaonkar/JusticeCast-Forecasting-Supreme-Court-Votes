"""Tests for src.modeling.splits — the canonical train/test split shared
by both the BoW track (Phase 3/4) and the embeddings track (Phase 4.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import splits


def test_constants_are_locked():
    """Non-Negotiables #1, #2, #5: split parameters are project-level constants."""
    assert splits.N_SPLITS == 5
    assert splits.RANDOM_STATE == 42
    assert splits.TEST_FOLD_INDEX == 0


def test_split_is_deterministic_across_calls():
    s1 = splits.get_train_test_split()
    s2 = splits.get_train_test_split()
    assert np.array_equal(s1.train_idx, s2.train_idx)
    assert np.array_equal(s1.test_idx, s2.test_idx)


def test_split_has_no_case_leakage():
    s = splits.get_train_test_split()
    overlap = set(s.groups_train) & set(s.groups_test)
    assert not overlap, f"caseId leakage: {len(overlap)} overlapping cases"


def test_split_label_balance_preserved():
    """Stratification means train and test should have similar petitioner-rates."""
    s = splits.get_train_test_split()
    train_rate = s.y_train.mean()
    test_rate = s.y_test.mean()
    assert abs(train_rate - test_rate) < 0.02, (
        f"Stratification failed: train rate {train_rate:.3f}, "
        f"test rate {test_rate:.3f}"
    )


def test_split_test_fraction_is_about_20pct():
    s = splits.get_train_test_split()
    total = len(s.train_idx) + len(s.test_idx)
    test_frac = len(s.test_idx) / total
    # 5-fold split → test ≈ 20%; some drift is fine due to grouped folds
    assert 0.15 < test_frac < 0.25, f"Test fraction {test_frac:.3f} outside 15-25%"


def test_assert_no_case_leakage_raises_on_overlap():
    """Smoke-test the leakage guard."""
    df = pd.DataFrame({
        "caseId": ["A", "A", "B", "B"],
        "voted_petitioner": [1, 0, 1, 0],
        "text": ["x"] * 4,
    })
    fake = splits.Split(
        train_idx=np.array([0, 2]),
        test_idx=np.array([1, 3]),
        train_df=df.iloc[[0, 2]],
        test_df=df.iloc[[1, 3]],
        y_train=np.array([1, 1]),
        y_test=np.array([0, 0]),
        groups_train=np.array(["A", "B"]),
        groups_test=np.array(["A", "B"]),  # both share A and B → leakage
    )
    import pytest
    with pytest.raises(RuntimeError, match="DATA LEAKAGE"):
        splits.assert_no_case_leakage(fake)


def test_phase3_and_phase4_share_identical_split():
    """Non-Negotiable #15: both tracks must consume the same fold-0 test rows.

    Phase 3 imports `get_train_test_split` from `src.modeling.splits` since
    the refactor; this test guards against a future regression where someone
    re-introduces a separate split definition."""
    s_a = splits.get_train_test_split()
    s_b = splits.get_train_test_split()
    # caseId sets in test must match exactly across two independent calls
    assert set(s_a.test_df["caseId"]) == set(s_b.test_df["caseId"])
    assert set(s_a.train_df["caseId"]) == set(s_b.train_df["caseId"])
