"""Phase 3: 9-combination baseline sweep.

(3 vectorizers) × (3 classifiers) on the modeling table, evaluated on a held-
out test fold (fold 0 of `StratifiedGroupKFold(n_splits=5, random_state=42)`
with `groups=case_id`, per Non-Negotiable #1).

Outputs:
    reports/results/baseline_results.csv     — per-combo metrics + timing
    reports/results/per_justice_lift.csv     — per-(combo, Justice) lift table
    reports/results/top_features_best_linear.csv
                                              — top 30 ± coefficients of the
                                                best linear model

Usage:
    python -m src.phase3_baseline_sweep
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.text_clean import STOPWORDS_FOR_VECTORIZER, vectorizer_preprocessor

MODELING_TABLE_PATH = Path("data/processed/modeling_table.parquet")
BASELINE_RESULTS_PATH = Path("reports/results/baseline_results.csv")
PER_JUSTICE_LIFT_PATH = Path("reports/results/per_justice_lift.csv")
TOP_FEATURES_PATH = Path("reports/results/top_features_best_linear.csv")

RANDOM_STATE = 42

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vectorizers + classifiers (the 9-combo grid)
# ---------------------------------------------------------------------------

def make_vectorizers() -> dict[str, object]:
    """All three consume the same STOPWORDS_FOR_VECTORIZER (Non-Negotiable #3)
    AND the advocate-name `vectorizer_preprocessor` from text_clean (Phase 3.5
    addition — strips `mr <surname>`-style spans before tokenization)."""
    common = dict(
        stop_words=STOPWORDS_FOR_VECTORIZER,
        preprocessor=vectorizer_preprocessor,
        lowercase=False,  # preprocessor handles lowercase
    )
    return {
        "bow_unigram": CountVectorizer(ngram_range=(1, 1), **common),
        "tfidf_unigram": TfidfVectorizer(ngram_range=(1, 1), **common),
        "tfidf_bigram": TfidfVectorizer(ngram_range=(1, 2), min_df=2, **common),
    }


def make_classifiers() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def is_linear_combo(classifier_name: str) -> bool:
    return classifier_name in {"logreg", "linear_svc"}


# ---------------------------------------------------------------------------
# Train / evaluate one pipeline
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    combo_id: str
    vectorizer: str
    classifier: str
    n_train: int
    n_test: int
    n_features: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fit_time_sec: float
    predict_time_sec: float


def _scores_for_auc(pipe: Pipeline, X_test) -> np.ndarray:
    """Return decision-function scores for AUC.

    LinearSVC has no predict_proba — use decision_function (sklearn's
    roc_auc_score accepts rank scores). LogReg uses predict_proba; RF
    too. All three return shape (n,) for the positive class.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X_test)[:, 1]
    return pipe.decision_function(X_test)


def fit_combo(
    combo_id: str,
    vectorizer_name: str,
    classifier_name: str,
    X_train_text: pd.Series,
    y_train: np.ndarray,
    X_test_text: pd.Series,
    y_test: np.ndarray,
) -> tuple[FitResult, Pipeline, np.ndarray]:
    """Fit one pipeline; return metrics, the fitted pipeline, and test predictions."""
    vec = make_vectorizers()[vectorizer_name]
    clf = make_classifiers()[classifier_name]
    pipe = Pipeline([("vect", vec), ("clf", clf)])

    t0 = time.monotonic()
    pipe.fit(X_train_text, y_train)
    fit_time = time.monotonic() - t0

    t1 = time.monotonic()
    y_pred = pipe.predict(X_test_text)
    y_score = _scores_for_auc(pipe, X_test_text)
    predict_time = time.monotonic() - t1

    fitted_vec = pipe.named_steps["vect"]
    n_features = len(fitted_vec.get_feature_names_out())

    result = FitResult(
        combo_id=combo_id,
        vectorizer=vectorizer_name,
        classifier=classifier_name,
        n_train=len(y_train),
        n_test=len(y_test),
        n_features=n_features,
        accuracy=accuracy_score(y_test, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_score),
        fit_time_sec=fit_time,
        predict_time_sec=predict_time,
    )
    return result, pipe, y_pred


# ---------------------------------------------------------------------------
# Per-Justice lift (Non-Negotiable #12: per-Justice baselines as the reference)
# ---------------------------------------------------------------------------

def per_justice_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute each Justice's full-corpus majority-class baseline."""
    g = df.groupby("oyez_identifier")["voted_petitioner"]
    return pd.DataFrame({
        "oyez_identifier": g.mean().index,
        "petitioner_rate_full": g.mean().values,
        "n_full_rows": g.size().values,
    }).assign(
        per_justice_baseline=lambda d: d["petitioner_rate_full"].apply(
            lambda p: max(p, 1 - p)
        )
    )


def compute_per_justice_lift(
    test_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray,
    baselines: pd.DataFrame, combo_id: str,
) -> pd.DataFrame:
    """Per-Justice accuracy on the test fold, then lift over the full-corpus baseline."""
    out = (
        test_df.assign(_pred=y_pred, _true=y_test)
               .groupby("oyez_identifier")
               .apply(
                   lambda g: pd.Series({
                       "n_test_rows": len(g),
                       "model_accuracy": (g["_pred"] == g["_true"]).mean(),
                   }),
                   include_groups=False,
               )
               .reset_index()
               .merge(baselines, on="oyez_identifier", how="left")
    )
    out["lift_over_baseline"] = out["model_accuracy"] - out["per_justice_baseline"]
    out.insert(0, "combo_id", combo_id)
    return out


# ---------------------------------------------------------------------------
# Top features (best linear model)
# ---------------------------------------------------------------------------

def top_features_for_linear(pipe: Pipeline, n: int = 30) -> pd.DataFrame:
    """Top n positive and top n negative coefficients from a linear pipeline."""
    vec = pipe.named_steps["vect"]
    clf = pipe.named_steps["clf"]
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_.ravel()  # binary classifier => shape (n_features,)
    order = np.argsort(coefs)
    top_neg = [(int(i + 1), "respondent (negative)", feature_names[idx], float(coefs[idx]))
               for i, idx in enumerate(order[:n])]
    top_pos = [(int(i + 1), "petitioner (positive)", feature_names[idx], float(coefs[idx]))
               for i, idx in enumerate(order[-n:][::-1])]
    return pd.DataFrame(
        top_pos + top_neg,
        columns=["rank", "direction", "feature", "coefficient"],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    df = pd.read_parquet(MODELING_TABLE_PATH).reset_index(drop=True)
    logger.info("Loaded %s — %d rows × %d cols",
                MODELING_TABLE_PATH, *df.shape)

    X = df["text"]
    y = df["voted_petitioner"].astype(int).values
    groups = df["caseId"].values

    # Single train/test split: fold 0 = test, folds 1–4 = train.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(X, y, groups=groups))
    test_idx, train_idx = splits[0][1], splits[0][0]
    # Wait — convention is (train, test) per fold; fold 0 means "first fold
    # is the test set". Map accordingly.
    train_idx, test_idx = splits[0][0], splits[0][1]

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Sanity: case_id leakage check
    train_cases = set(train_df["caseId"])
    test_cases = set(test_df["caseId"])
    overlap = train_cases & test_cases
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlap)} caseIds overlap between train and test. "
            f"Group split is broken."
        )
    logger.info("Train: %d rows / %d cases. Test: %d rows / %d cases. Leakage: 0.",
                len(train_df), len(train_cases), len(test_df), len(test_cases))
    logger.info("Train petitioner-rate: %.3f. Test petitioner-rate: %.3f.",
                y_train.mean(), y_test.mean())

    BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    baselines = per_justice_baselines(df)

    all_results: list[FitResult] = []
    all_lift_chunks: list[pd.DataFrame] = []
    fitted_pipes: dict[str, Pipeline] = {}

    vectorizer_names = list(make_vectorizers().keys())
    classifier_names = list(make_classifiers().keys())

    total_t0 = time.monotonic()
    for v in vectorizer_names:
        for c in classifier_names:
            combo_id = f"{v}__{c}"
            logger.info("=== fitting %s ===", combo_id)
            res, pipe, y_pred = fit_combo(combo_id, v, c, X_train, y_train, X_test, y_test)
            logger.info(
                "  acc=%.3f  bal_acc=%.3f  AUC=%.3f  feats=%d  "
                "fit=%.1fs  predict=%.2fs",
                res.accuracy, res.balanced_accuracy, res.roc_auc,
                res.n_features, res.fit_time_sec, res.predict_time_sec,
            )
            all_results.append(res)
            fitted_pipes[combo_id] = pipe
            all_lift_chunks.append(
                compute_per_justice_lift(test_df, y_test, y_pred, baselines, combo_id)
            )

    total_elapsed = time.monotonic() - total_t0
    logger.info("All 9 combos done in %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)

    # Persist baseline_results.csv
    rows = [{
        "combo_id": r.combo_id, "vectorizer": r.vectorizer, "classifier": r.classifier,
        "n_train": r.n_train, "n_test": r.n_test, "n_features": r.n_features,
        "accuracy": r.accuracy, "balanced_accuracy": r.balanced_accuracy,
        "precision": r.precision, "recall": r.recall, "f1": r.f1, "roc_auc": r.roc_auc,
        "fit_time_sec": r.fit_time_sec, "predict_time_sec": r.predict_time_sec,
    } for r in all_results]
    pd.DataFrame(rows).to_csv(BASELINE_RESULTS_PATH, index=False)
    logger.info("Wrote %s", BASELINE_RESULTS_PATH)

    pd.concat(all_lift_chunks, ignore_index=True).to_csv(
        PER_JUSTICE_LIFT_PATH, index=False
    )
    logger.info("Wrote %s", PER_JUSTICE_LIFT_PATH)

    # Top features for the best LINEAR model (highest ROC AUC among logreg/linear_svc)
    linear_results = [r for r in all_results if is_linear_combo(r.classifier)]
    best_linear = max(linear_results, key=lambda r: r.roc_auc)
    logger.info("Best linear combo: %s (AUC=%.3f)",
                best_linear.combo_id, best_linear.roc_auc)
    top_feats = top_features_for_linear(fitted_pipes[best_linear.combo_id], n=30)
    top_feats.insert(0, "combo_id", best_linear.combo_id)
    top_feats.to_csv(TOP_FEATURES_PATH, index=False)
    logger.info("Wrote %s", TOP_FEATURES_PATH)

    # Console summary
    print("\n" + "=" * 70)
    print("BASELINE RESULTS (sorted by ROC AUC, descending)")
    print("=" * 70)
    df_results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    cols = ["combo_id", "accuracy", "balanced_accuracy", "roc_auc",
            "f1", "n_features", "fit_time_sec"]
    print(df_results[cols].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
