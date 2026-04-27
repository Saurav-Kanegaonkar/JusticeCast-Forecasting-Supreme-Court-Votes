"""Tests for src.phase45_baseline_sweep + src.phase45_gridsearch.

Most outcome-based assertions skip if the embedding cache or result CSVs
aren't present. The construction tests (no compute) always run."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src import phase45_baseline_sweep as p45b
from src import phase45_gridsearch as p45g


# ---------- Construction (no compute) ----------

def test_baseline_classifier_set():
    clfs = p45b.make_classifiers()
    assert set(clfs) == {"logreg", "svm_rbf", "random_forest"}
    assert isinstance(clfs["logreg"], LogisticRegression)
    assert isinstance(clfs["svm_rbf"], SVC)
    assert clfs["svm_rbf"].kernel == "rbf"
    assert clfs["svm_rbf"].probability is True
    assert isinstance(clfs["random_forest"], RandomForestClassifier)
    # All use class_weight='balanced' (Non-Negotiable #6)
    for c in clfs.values():
        assert c.class_weight == "balanced"


def test_gridsearch_makes_correct_estimators():
    assert isinstance(p45g.make_logreg(), LogisticRegression)
    assert isinstance(p45g.make_svm_rbf(), SVC)
    assert p45g.make_svm_rbf().kernel == "rbf"
    assert p45g.make_svm_rbf().probability is True
    assert isinstance(p45g.make_rf(), RandomForestClassifier)


def test_gridsearch_grids_match_cai_plan():
    """LogReg 5×2=10, SVM-RBF 3×3=9, RF 3×3×3=27 settings."""
    assert len(p45g.LOGREG_GRID["C"]) == 5
    assert len(p45g.LOGREG_GRID["l1_ratio"]) == 2
    assert len(p45g.SVM_RBF_GRID["C"]) == 3
    assert len(p45g.SVM_RBF_GRID["gamma"]) == 3
    assert len(p45g.RF_GRID["n_estimators"]) == 3
    assert len(p45g.RF_GRID["max_depth"]) == 3
    assert len(p45g.RF_GRID["min_samples_split"]) == 3


def test_gridsearch_n_jobs_settings():
    """Lessons from BoW Phase 4: linear/svm n_jobs=4, RF outer n_jobs=1."""
    assert p45g.N_JOBS_LINEAR == 4
    assert p45g.N_JOBS_SVM == 4
    assert p45g.N_JOBS_RF_OUTER == 1


# ---------- Result CSVs (require runs) ----------

@pytest.fixture
def baseline_results():
    if not p45b.BASELINE_RESULTS_PATH.exists():
        pytest.skip("embedding_baseline_results.csv missing — run baseline sweep")
    return pd.read_csv(p45b.BASELINE_RESULTS_PATH)


def test_baseline_has_6_combos(baseline_results):
    """2 embedding models × 3 classifiers = 6."""
    assert len(baseline_results) == 6


def test_baseline_combos_cover_both_embeddings_and_all_classifiers(baseline_results):
    classifiers = set(baseline_results["classifier"])
    assert classifiers == {"logreg", "svm_rbf", "random_forest"}
    embeddings = set(baseline_results["embedding_model"])
    assert "all-MiniLM-L6-v2" in embeddings
    assert "all-mpnet-base-v2" in embeddings


def test_baseline_no_combo_anti_predictive(baseline_results):
    bad = baseline_results[baseline_results["roc_auc"] < 0.5]
    assert bad.empty, (
        f"Combos with ROC AUC < 0.5: "
        f"{bad[['combo_id', 'roc_auc']].to_dict('records')}"
    )


@pytest.fixture
def gridsearch_results():
    if not p45g.GRIDSEARCH_RESULTS_PATH.exists():
        pytest.skip("embedding_gridsearch_results.csv missing — run gridsearch")
    return pd.read_csv(p45g.GRIDSEARCH_RESULTS_PATH)


def test_gridsearch_has_three_models(gridsearch_results):
    assert set(gridsearch_results["model"]) == {"logreg", "svm_rbf", "random_forest"}


def test_gridsearch_logreg_row_count(gridsearch_results):
    """LogReg: 5 C × 2 l1_ratio = 10 settings."""
    n = (gridsearch_results["model"] == "logreg").sum()
    assert n == 10, f"Expected 10 LogReg rows, got {n}"


def test_gridsearch_svm_row_count(gridsearch_results):
    """SVM-RBF: 3 C × 3 gamma = 9 settings."""
    n = (gridsearch_results["model"] == "svm_rbf").sum()
    assert n == 9, f"Expected 9 SVM-RBF rows, got {n}"


def test_gridsearch_rf_row_count(gridsearch_results):
    """RF: 3 × 3 × 3 = 27 settings."""
    n = (gridsearch_results["model"] == "random_forest").sum()
    assert n == 27, f"Expected 27 RF rows, got {n}"


def test_per_justice_auc_table_present():
    if not p45g.PHASE45_PER_JUSTICE_AUC_PATH.exists():
        pytest.skip("phase45_per_justice_auc.csv missing")
    auc = pd.read_csv(p45g.PHASE45_PER_JUSTICE_AUC_PATH)
    assert len(auc) >= 10
    has_ci = auc["ci_lo_95"].notna().sum()
    assert has_ci >= 5
