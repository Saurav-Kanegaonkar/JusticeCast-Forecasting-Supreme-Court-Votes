"""Phase 4.5 — embeddings-track baseline sweep (peer to BoW Phase 3).

For each of two pre-trained embedding models (MiniLM, MPNet) × three
classifiers (LogReg, SVM-RBF, RandomForest), fit on train and evaluate on
the canonical fold-0 test set from src/modeling/splits.py.

Outputs:
    reports/results/embedding_baseline_results.csv
    reports/results/embedding_baseline_per_justice.csv
CRISP-DM phase: Modeling.
Phase 4.5 — embeddings 6-combo untuned baseline sweep.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from src.compute_embeddings import MODELS, load_embeddings
from src.modeling.splits import RANDOM_STATE, get_train_test_split

BASELINE_RESULTS_PATH = Path("reports/results/embedding_baseline_results.csv")
PER_JUSTICE_PATH = Path("reports/results/embedding_baseline_per_justice.csv")

logger = logging.getLogger(__name__)


def make_classifiers() -> dict[str, object]:
    """Three classifiers per the cai-plan. SVM-RBF is the methodological
    upgrade enabled by dense vectors — would have been inappropriate on the
    200K-dim sparse BoW representation."""
    return {
        "logreg": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            probability=True,  # needed for AUC + Phase 5 calibration
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


@dataclass
class FitResult:
    combo_id: str
    embedding_model: str
    classifier: str
    embedding_dim: int
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fit_time_sec: float
    predict_time_sec: float


def _scores_for_auc(clf, X_test) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]
    return clf.decision_function(X_test)


def fit_combo(
    embedding_key: str, classifier_name: str,
    X_train, y_train, X_test, y_test,
) -> tuple[FitResult, object, np.ndarray]:
    clf = make_classifiers()[classifier_name]
    combo_id = f"{embedding_key}__{classifier_name}"
    t0 = time.monotonic()
    clf.fit(X_train, y_train)
    fit_time = time.monotonic() - t0

    t1 = time.monotonic()
    y_pred = clf.predict(X_test)
    y_score = _scores_for_auc(clf, X_test)
    predict_time = time.monotonic() - t1

    res = FitResult(
        combo_id=combo_id,
        embedding_model=MODELS[embedding_key][0],
        classifier=classifier_name,
        embedding_dim=MODELS[embedding_key][1],
        n_train=len(y_train), n_test=len(y_test),
        accuracy=accuracy_score(y_test, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_score),
        fit_time_sec=fit_time, predict_time_sec=predict_time,
    )
    return res, clf, y_pred


def _per_justice_baselines(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("oyez_identifier")["voted_petitioner"]
    return pd.DataFrame({
        "oyez_identifier": g.mean().index,
        "petitioner_rate_full": g.mean().values,
    }).assign(per_justice_baseline=lambda d:
              d["petitioner_rate_full"].apply(lambda p: max(p, 1 - p)))


def per_justice_metrics(
    test_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray,
    baselines: pd.DataFrame, combo_id: str,
) -> pd.DataFrame:
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


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    split = get_train_test_split()
    y_train, y_test = split.y_train, split.y_test
    test_df = split.test_df

    # Embeddings are aligned to the modeling-table row order (preserved when
    # compute_embeddings.py encodes from the parquet without reset_index).
    # split.train_idx / .test_idx are positional indices into that same
    # ordering, so emb[split.train_idx] == rows for split.train_df.
    modeling_table = pd.read_parquet("data/processed/modeling_table.parquet")
    baselines = _per_justice_baselines(modeling_table)

    BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[FitResult] = []
    all_per_justice_chunks: list[pd.DataFrame] = []

    overall_t0 = time.monotonic()
    for embedding_key in ["minilm", "mpnet"]:
        emb, row_idx = load_embeddings(embedding_key)
        # Sanity: embeddings must be aligned to modeling_table row order.
        assert len(emb) == len(modeling_table), (
            f"{embedding_key} embeddings have {len(emb)} rows, "
            f"modeling_table has {len(modeling_table)}"
        )
        X_train = emb[split.train_idx]
        X_test = emb[split.test_idx]
        for classifier_name in ["logreg", "svm_rbf", "random_forest"]:
            combo_id = f"{embedding_key}__{classifier_name}"
            logger.info("=== fitting %s (X_train=%s, X_test=%s) ===",
                        combo_id, X_train.shape, X_test.shape)
            res, _clf, y_pred = fit_combo(
                embedding_key, classifier_name,
                X_train, y_train, X_test, y_test,
            )
            logger.info(
                "  acc=%.3f bal_acc=%.3f AUC=%.3f  fit=%.1fs predict=%.2fs",
                res.accuracy, res.balanced_accuracy, res.roc_auc,
                res.fit_time_sec, res.predict_time_sec,
            )
            all_results.append(res)
            all_per_justice_chunks.append(
                per_justice_metrics(test_df, y_test, y_pred, baselines, combo_id)
            )

    elapsed = time.monotonic() - overall_t0
    logger.info("All 6 embedding baselines done in %.0fs (%.1f min)",
                elapsed, elapsed / 60)

    rows = [{
        "combo_id": r.combo_id,
        "embedding_model": r.embedding_model,
        "classifier": r.classifier,
        "embedding_dim": r.embedding_dim,
        "n_train": r.n_train, "n_test": r.n_test,
        "accuracy": r.accuracy, "balanced_accuracy": r.balanced_accuracy,
        "precision": r.precision, "recall": r.recall,
        "f1": r.f1, "roc_auc": r.roc_auc,
        "fit_time_sec": r.fit_time_sec, "predict_time_sec": r.predict_time_sec,
    } for r in all_results]
    pd.DataFrame(rows).to_csv(BASELINE_RESULTS_PATH, index=False)
    logger.info("Wrote %s", BASELINE_RESULTS_PATH)

    pd.concat(all_per_justice_chunks, ignore_index=True).to_csv(
        PER_JUSTICE_PATH, index=False)
    logger.info("Wrote %s", PER_JUSTICE_PATH)

    print("\n" + "=" * 70)
    print("EMBEDDING BASELINE RESULTS (sorted by ROC AUC)")
    print("=" * 70)
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    cols = ["combo_id", "embedding_dim", "accuracy", "balanced_accuracy",
            "roc_auc", "f1", "fit_time_sec"]
    print(df[cols].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
