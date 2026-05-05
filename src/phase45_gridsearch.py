"""Phase 4.5 — embeddings GridSearchCV (peer to BoW Phase 4).

Tunes LogReg, SVM-RBF, and RandomForest on the better-performing embedding
model (selected by ROC AUC on the baseline sweep). Same fold-0 test set as
BoW Phase 4. Same evaluation harness. Reports per-Justice ROC AUC with
bootstrap CIs.

Memory discipline: n_jobs=4 (not -1) for outer parallelism, mirroring the
BoW Phase 4 lesson. RBF SVM with `probability=True` is the slow part.

Outputs:
    reports/results/embedding_gridsearch_results.csv
    reports/results/phase45_test_eval.csv
    reports/results/phase45_per_justice_auc.csv
    reports/results/phase45_top_features.csv  (RF importances or LogReg coefs)
CRISP-DM phase: Modeling.
Phase 4.5 — embeddings GridSearchCV on the better-performing encoder.
"""
from __future__ import annotations

import logging
import time
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.compute_embeddings import MODELS, load_embeddings
from src.modeling.splits import (
    RANDOM_STATE,
    get_cv_splitter,
    get_train_test_split,
)

GRIDSEARCH_RESULTS_PATH = Path("reports/results/embedding_gridsearch_results.csv")
PHASE45_TEST_EVAL_PATH = Path("reports/results/phase45_test_eval.csv")
PHASE45_PER_JUSTICE_AUC_PATH = Path("reports/results/phase45_per_justice_auc.csv")
PHASE45_TOP_FEATURES_PATH = Path("reports/results/phase45_top_features.csv")

# Per Phase-4 lesson: n_jobs=4 outer for non-tree classifiers (svm/logreg),
# n_jobs=1 outer for RF (already n_jobs=-1 internally).
N_JOBS_LINEAR = 4
N_JOBS_SVM = 4
N_JOBS_RF_OUTER = 1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipelines (no vectorizer step — input is already dense embeddings)
# ---------------------------------------------------------------------------

def make_logreg() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",  # supports both l1_ratio=0 and 1
        random_state=RANDOM_STATE,
    )


def make_svm_rbf() -> SVC:
    return SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        probability=True,  # for AUC + Phase 5 calibration
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Grids (per cai-plan v10)
# ---------------------------------------------------------------------------

LOGREG_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    # `l1_ratio` is the modern (sklearn 1.8+) replacement for `penalty=`.
    # Per sklearn docs, `liblinear` supports only the corners
    # `l1_ratio ∈ {0, 1}` (= pure L2, pure L1); the continuous-elasticnet
    # range requires `solver='saga'`. So this two-point grid is the full
    # liblinear-compatible sweep, not a sparse approximation of [0..1].
    "l1_ratio": [0.0, 1.0],  # 0.0 = pure L2, 1.0 = pure L1 (liblinear corners)
}

SVM_RBF_GRID = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001],
}

RF_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 20, 50],
    "min_samples_split": [2, 5, 10],
}


# ---------------------------------------------------------------------------
# GridSearchCV runners
# ---------------------------------------------------------------------------

def run_gridsearch(
    estimator, param_grid: dict, X_train, y_train, groups_train,
    n_jobs: int, label: str,
) -> GridSearchCV:
    n_settings = 1
    for v in param_grid.values():
        n_settings *= len(v)
    logger.info("%s: %d settings × 5-fold = %d fits (n_jobs=%d)",
                label, n_settings, n_settings * 5, n_jobs)
    gs = GridSearchCV(
        estimator,
        param_grid,
        scoring="roc_auc",
        cv=get_cv_splitter(),
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    return gs


def evaluate_on_test(label: str, gs: GridSearchCV, X_test, y_test) -> dict:
    est = gs.best_estimator_
    y_pred = est.predict(X_test)
    if hasattr(est, "predict_proba"):
        y_score = est.predict_proba(X_test)[:, 1]
    else:
        y_score = est.decision_function(X_test)
    return {
        "model": label,
        "best_cv_score": gs.best_score_,
        "best_params": str(gs.best_params_),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_score),
        "cv_test_gap": gs.best_score_ - roc_auc_score(y_test, y_score),
    }


def per_justice_auc_with_bootstrap(
    est, X_test, y_test, justice_labels: pd.Series,
    n_bootstrap: int = 1000, min_n_for_ci: int = 30,
) -> pd.DataFrame:
    if hasattr(est, "predict_proba"):
        y_score = est.predict_proba(X_test)[:, 1]
    else:
        y_score = est.decision_function(X_test)

    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict] = []
    for justice, idx in justice_labels.groupby(justice_labels).groups.items():
        idx = np.asarray(idx)
        n = len(idx)
        y_j = y_test[idx]
        s_j = y_score[idx]
        if len(np.unique(y_j)) < 2:
            point = float("nan")
        else:
            point = roc_auc_score(y_j, s_j)
        ci_lo = ci_hi = float("nan")
        if n >= min_n_for_ci and not np.isnan(point):
            boots = []
            for _ in range(n_bootstrap):
                resample = rng.choice(n, size=n, replace=True)
                yb = y_j[resample]
                sb = s_j[resample]
                if len(np.unique(yb)) < 2:
                    continue
                boots.append(roc_auc_score(yb, sb))
            if boots:
                ci_lo = float(np.percentile(boots, 2.5))
                ci_hi = float(np.percentile(boots, 97.5))
        rows.append({
            "oyez_identifier": justice,
            "n_test_rows": int(n),
            "point_auc": point,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
        })
    return pd.DataFrame(rows).sort_values("point_auc", ascending=False)


def persist_cv_results(gs: GridSearchCV, label: str, append: bool) -> None:
    cv = pd.DataFrame(gs.cv_results_)
    keep = ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
            "params", "mean_test_score", "std_test_score", "rank_test_score"]
    cv = cv[keep].copy()
    cv.insert(0, "model", label)
    GRIDSEARCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(GRIDSEARCH_RESULTS_PATH, mode="a" if append else "w",
              index=False, header=not append)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _pick_better_embedding_from_baseline() -> str:
    """Read the baseline-sweep CSV and pick the embedding model with the
    highest ROC AUC across any classifier."""
    base = pd.read_csv("reports/results/embedding_baseline_results.csv")
    # Map embedding_model name back to the short key
    name_to_key = {v[0]: k for k, v in MODELS.items()}
    base["embedding_key"] = base["embedding_model"].map(name_to_key)
    best_per_emb = base.groupby("embedding_key")["roc_auc"].max()
    winner = best_per_emb.idxmax()
    logger.info("Baseline AUC by embedding: %s",
                best_per_emb.round(4).to_dict())
    logger.info("Selected embedding for tuning: %s (best baseline AUC=%.4f)",
                winner, best_per_emb[winner])
    return winner


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    embedding_key = _pick_better_embedding_from_baseline()
    emb, _row_idx = load_embeddings(embedding_key)

    split = get_train_test_split()
    X_train = emb[split.train_idx]
    X_test = emb[split.test_idx]
    y_train = split.y_train
    y_test = split.y_test
    groups_train = split.groups_train

    logger.info("Train: %d × %d. Test: %d × %d.",
                X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1])

    test_evals: list[dict] = []
    fitted: dict[str, object] = {}

    overall_t0 = time.monotonic()

    # LogReg
    logger.info("=" * 60); logger.info("LogReg"); logger.info("=" * 60)
    t0 = time.monotonic()
    gs_lr = run_gridsearch(make_logreg(), {"C": LOGREG_GRID["C"],
                                            "l1_ratio": LOGREG_GRID["l1_ratio"]},
                            X_train, y_train, groups_train,
                            n_jobs=N_JOBS_LINEAR, label="LogReg")
    elapsed_lr = time.monotonic() - t0
    logger.info("LogReg done in %.0fs (%.1f min). best CV AUC=%.4f, %s",
                elapsed_lr, elapsed_lr / 60, gs_lr.best_score_, gs_lr.best_params_)
    persist_cv_results(gs_lr, "logreg", append=False)
    test_evals.append(evaluate_on_test("logreg", gs_lr, X_test, y_test))
    fitted["logreg"] = gs_lr.best_estimator_

    # SVM-RBF
    logger.info("=" * 60); logger.info("SVM-RBF"); logger.info("=" * 60)
    t0 = time.monotonic()
    gs_svm = run_gridsearch(make_svm_rbf(), SVM_RBF_GRID,
                             X_train, y_train, groups_train,
                             n_jobs=N_JOBS_SVM, label="SVM-RBF")
    elapsed_svm = time.monotonic() - t0
    logger.info("SVM-RBF done in %.0fs (%.1f min). best CV AUC=%.4f, %s",
                elapsed_svm, elapsed_svm / 60, gs_svm.best_score_, gs_svm.best_params_)
    persist_cv_results(gs_svm, "svm_rbf", append=True)
    test_evals.append(evaluate_on_test("svm_rbf", gs_svm, X_test, y_test))
    fitted["svm_rbf"] = gs_svm.best_estimator_

    # RandomForest
    logger.info("=" * 60); logger.info("RandomForest"); logger.info("=" * 60)
    t0 = time.monotonic()
    gs_rf = run_gridsearch(make_rf(), RF_GRID,
                            X_train, y_train, groups_train,
                            n_jobs=N_JOBS_RF_OUTER, label="RF")
    elapsed_rf = time.monotonic() - t0
    logger.info("RF done in %.0fs (%.1f min). best CV AUC=%.4f, %s",
                elapsed_rf, elapsed_rf / 60, gs_rf.best_score_, gs_rf.best_params_)
    persist_cv_results(gs_rf, "random_forest", append=True)
    test_evals.append(evaluate_on_test("random_forest", gs_rf, X_test, y_test))
    fitted["random_forest"] = gs_rf.best_estimator_

    overall_elapsed = time.monotonic() - overall_t0
    logger.info("Phase 4.5 GridSearch total: %.0fs (%.1f min)",
                overall_elapsed, overall_elapsed / 60)

    pd.DataFrame(test_evals).to_csv(PHASE45_TEST_EVAL_PATH, index=False)
    logger.info("Wrote %s", PHASE45_TEST_EVAL_PATH)

    winner_row = max(test_evals, key=lambda r: r["test_roc_auc"])
    winner = winner_row["model"]
    winner_est = fitted[winner]
    logger.info("=" * 60)
    logger.info("Phase 4.5 winner: %s on %s — test AUC=%.4f, CV AUC=%.4f",
                winner, embedding_key, winner_row["test_roc_auc"],
                winner_row["best_cv_score"])
    logger.info("=" * 60)

    per_just = per_justice_auc_with_bootstrap(
        winner_est, X_test, y_test, split.test_df["oyez_identifier"],
    )
    per_just.insert(0, "model", winner)
    per_just.insert(1, "embedding", MODELS[embedding_key][0])
    per_just.to_csv(PHASE45_PER_JUSTICE_AUC_PATH, index=False)
    logger.info("Wrote %s", PHASE45_PER_JUSTICE_AUC_PATH)

    # Top features: dense embeddings have no per-token interpretability.
    # For LogReg, save the per-dimension coefficients (top 30 ± by magnitude)
    # — useful for sanity-checking which embedding dims drive the prediction,
    # but no token-level story like BoW. RF: feature importances (same).
    # For SVM-RBF: no analogous coefficients (kernel space). Save a marker row.
    if winner == "logreg":
        coefs = winner_est.coef_.ravel()
        order = np.argsort(coefs)
        rows = []
        for i, idx in enumerate(order[-30:][::-1]):
            rows.append({"model": winner, "rank": i + 1, "direction": "petitioner (positive)",
                         "feature": f"emb_dim_{idx}", "coefficient": float(coefs[idx])})
        for i, idx in enumerate(order[:30]):
            rows.append({"model": winner, "rank": i + 1, "direction": "respondent (negative)",
                         "feature": f"emb_dim_{idx}", "coefficient": float(coefs[idx])})
        pd.DataFrame(rows).to_csv(PHASE45_TOP_FEATURES_PATH, index=False)
    elif winner == "random_forest":
        importances = winner_est.feature_importances_
        order = np.argsort(importances)[::-1][:30]
        rows = [{"model": winner, "rank": i + 1, "direction": "importance",
                 "feature": f"emb_dim_{idx}", "coefficient": float(importances[idx])}
                for i, idx in enumerate(order)]
        pd.DataFrame(rows).to_csv(PHASE45_TOP_FEATURES_PATH, index=False)
    else:
        # SVM-RBF — no per-feature coefficients. Document the gap.
        pd.DataFrame([{"model": winner, "rank": 1, "direction": "n/a",
                        "feature": "rbf-kernel — no per-dim coefs",
                        "coefficient": float("nan")}]).to_csv(
            PHASE45_TOP_FEATURES_PATH, index=False)
    logger.info("Wrote %s", PHASE45_TOP_FEATURES_PATH)

    print("\n" + "=" * 70)
    print(f"PHASE 4.5 SUMMARY (embedding: {MODELS[embedding_key][0]})")
    print("=" * 70)
    df_eval = pd.DataFrame(test_evals).sort_values("test_roc_auc", ascending=False)
    print(df_eval[["model", "best_cv_score", "test_roc_auc", "cv_test_gap",
                   "test_accuracy", "test_balanced_accuracy"]].to_string(
        index=False, float_format="%.4f"))
    print(f"\nWinner: {winner} on {embedding_key} (test AUC = {winner_row['test_roc_auc']:.4f})")
    print(f"Total Phase 4.5 GridSearch wall-clock: {overall_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
