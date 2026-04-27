"""Phase 4: BoW GridSearchCV — sequential strategy.

Stage 4A: Two GridSearchCV runs (LogReg + LinearSVC) sharing the same
    vectorizer parameter grid. Tunes (classifier hyperparams) × (vectorizer
    hyperparams) × 5-fold CV with `groups=case_id`.

Stage 4B: RandomForest GridSearchCV with the vectorizer fixed at Stage 4A's
    winning config. Tunes RF hyperparams × 5-fold CV.

Both stages report 5-fold CV mean ± std AUC, then refit the best estimator
on full train and evaluate on the held-out fold-0 test set (canonical
split from src/modeling/splits.py).

Outputs:
    reports/results/gridsearch_results.csv          — every fit's CV row
    reports/results/phase4_test_eval.csv            — best-of-each on test
    reports/results/phase4_per_justice_auc.csv      — bootstrap CIs per Justice
    reports/results/phase4_top_features.csv         — top 30 ± from final model

Usage:
    python -m src.phase4_gridsearch
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.modeling.splits import (
    RANDOM_STATE,
    get_cv_splitter,
    get_train_test_split,
)
from src.text_clean import STOPWORDS_FOR_VECTORIZER, vectorizer_preprocessor

GRIDSEARCH_RESULTS_PATH = Path("reports/results/gridsearch_results.csv")
PHASE4_TEST_EVAL_PATH = Path("reports/results/phase4_test_eval.csv")
PHASE4_PER_JUSTICE_AUC_PATH = Path("reports/results/phase4_per_justice_auc.csv")
PHASE4_TOP_FEATURES_PATH = Path("reports/results/phase4_top_features.csv")

VECTORIZER_GRID = {
    "vect__min_df": [2, 5],
    "vect__max_df": [0.9, 0.95],
    "vect__ngram_range": [(1, 1), (1, 2), (1, 3)],
}

# n_jobs sized to keep memory under control. Phase 4 has TF-IDF trigrams
# producing 500K+ feature matrices; n_jobs=-1 spins up 8 workers each
# holding a copy and OOMs the machine. n_jobs=4 halves that.
N_JOBS_LINEAR = 4
# RF parallelizes internally via clf__n_jobs=-1 so the OUTER GridSearchCV
# n_jobs stays at 1 — avoids over-subscription.
N_JOBS_RF_OUTER = 1

# Pipeline memory caching reuses vectorizer fits across grid points that
# share the same vectorizer config. Big win since the same
# (min_df, max_df, ngram_range) repeats across every clf hyperparam combo.
PIPELINE_CACHE_DIR = Path("/tmp/justicecast_phase4_cache")

logger = logging.getLogger(__name__)


def _make_memory_cache() -> Memory:
    """Per-run cache dir; cleared before run starts to avoid stale entries."""
    if PIPELINE_CACHE_DIR.exists():
        shutil.rmtree(PIPELINE_CACHE_DIR)
    PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Memory(location=str(PIPELINE_CACHE_DIR), verbose=0)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def _common_vectorizer_kwargs() -> dict:
    return dict(
        stop_words=STOPWORDS_FOR_VECTORIZER,
        preprocessor=vectorizer_preprocessor,
        lowercase=False,  # preprocessor lowercases
    )


def make_logreg_pipeline(memory: Memory | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            # TfidfVectorizer is the default for the Phase 4 grid; ngram_range
            # is tuned via vect__ngram_range so this initial setting doesn't
            # matter.
            ("vect", TfidfVectorizer(**_common_vectorizer_kwargs())),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",  # handles both L1 and L2 cleanly on binary
                random_state=RANDOM_STATE,
            )),
        ],
        memory=memory,
    )


def make_svm_pipeline(memory: Memory | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("vect", TfidfVectorizer(**_common_vectorizer_kwargs())),
            ("clf", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,  # slow to converge with class_weight + L2
            )),
        ],
        memory=memory,
    )


def make_rf_pipeline_with_fixed_vectorizer(
    best_vec_params: dict, memory: Memory | None = None
) -> Pipeline:
    """RF pipeline with vectorizer hyperparams locked from Stage 4A winner."""
    vec_kwargs = {
        **_common_vectorizer_kwargs(),
        "min_df": best_vec_params["vect__min_df"],
        "max_df": best_vec_params["vect__max_df"],
        "ngram_range": best_vec_params["vect__ngram_range"],
    }
    return Pipeline(
        steps=[
            ("vect", TfidfVectorizer(**vec_kwargs)),
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ],
        memory=memory,
    )


# ---------------------------------------------------------------------------
# Stage 4A
# ---------------------------------------------------------------------------

def run_stage_4a_logreg(X_train, y_train, groups_train, memory: Memory) -> GridSearchCV:
    pipe = make_logreg_pipeline(memory=memory)
    # sklearn 1.8 deprecated `penalty` in favor of `l1_ratio`. Mapping is:
    #   l1_ratio=0.0 ↔ penalty='l2'; l1_ratio=1.0 ↔ penalty='l1'. We tune
    #   the two corners (matching the cai-plan's L1 vs L2 grid intent).
    param_grid = {
        **VECTORIZER_GRID,
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__l1_ratio": [0.0, 1.0],
    }
    n_settings = (
        len(VECTORIZER_GRID["vect__min_df"])
        * len(VECTORIZER_GRID["vect__max_df"])
        * len(VECTORIZER_GRID["vect__ngram_range"])
        * len(param_grid["clf__C"])
        * len(param_grid["clf__l1_ratio"])
    )
    logger.info("Stage 4A LogReg: %d settings × 5-fold CV = %d fits",
                n_settings, n_settings * 5)
    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="roc_auc",
        cv=get_cv_splitter(),
        n_jobs=N_JOBS_LINEAR,
        verbose=1,
        return_train_score=False,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    return gs


def run_stage_4a_svm(X_train, y_train, groups_train, memory: Memory) -> GridSearchCV:
    pipe = make_svm_pipeline(memory=memory)
    param_grid = {
        **VECTORIZER_GRID,
        "clf__C": [0.01, 0.1, 1, 10],
    }
    n_settings = (
        len(VECTORIZER_GRID["vect__min_df"])
        * len(VECTORIZER_GRID["vect__max_df"])
        * len(VECTORIZER_GRID["vect__ngram_range"])
        * len(param_grid["clf__C"])
    )
    logger.info("Stage 4A SVM: %d settings × 5-fold CV = %d fits",
                n_settings, n_settings * 5)
    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="roc_auc",
        cv=get_cv_splitter(),
        n_jobs=N_JOBS_LINEAR,
        verbose=1,
        return_train_score=False,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    return gs


# ---------------------------------------------------------------------------
# Stage 4B
# ---------------------------------------------------------------------------

def run_stage_4b_rf(
    X_train, y_train, groups_train, best_vec_params: dict, memory: Memory
) -> GridSearchCV:
    pipe = make_rf_pipeline_with_fixed_vectorizer(best_vec_params, memory=memory)
    param_grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 20, 50],
        "clf__min_samples_split": [2, 5, 10],
    }
    n_settings = 3 * 3 * 3
    logger.info("Stage 4B RF: %d settings × 5-fold CV = %d fits",
                n_settings, n_settings * 5)
    # RF is already n_jobs=-1 internally; outer n_jobs=1 avoids over-subscription.
    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="roc_auc",
        cv=get_cv_splitter(),
        n_jobs=N_JOBS_RF_OUTER,
        verbose=1,
        return_train_score=False,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    return gs


# ---------------------------------------------------------------------------
# Test-set evaluation + per-Justice AUC bootstrap
# ---------------------------------------------------------------------------

def evaluate_on_test(model_label: str, gs: GridSearchCV, X_test, y_test) -> dict:
    """Refit-best evaluation on the held-out fold-0 test set."""
    pipe = gs.best_estimator_
    y_pred = pipe.predict(X_test)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = pipe.decision_function(X_test)
    return {
        "model": model_label,
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
    pipe: Pipeline, X_test, y_test, justice_labels: pd.Series,
    n_bootstrap: int = 1000, min_n_for_ci: int = 30,
) -> pd.DataFrame:
    """Per-Justice AUC + bootstrap CI on the fold-0 test set.

    For Justices with fewer than `min_n_for_ci` test rows, return point AUC
    only (CI columns NaN) — bootstrap CIs aren't meaningful at small n.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = pipe.decision_function(X_test)

    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict] = []
    for justice, idx in justice_labels.groupby(justice_labels).groups.items():
        idx = np.asarray(idx)
        n = len(idx)
        y_j = y_test[idx]
        s_j = y_score[idx]
        # AUC undefined if only one class present
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


# ---------------------------------------------------------------------------
# Persist GridSearchCV `cv_results_` rows
# ---------------------------------------------------------------------------

def persist_cv_results(gs: GridSearchCV, model_label: str, append: bool) -> None:
    cv = pd.DataFrame(gs.cv_results_)
    keep_cols = [
        "mean_fit_time", "std_fit_time",
        "mean_score_time", "std_score_time",
        "params",
        "mean_test_score", "std_test_score", "rank_test_score",
    ]
    cv = cv[keep_cols].copy()
    cv.insert(0, "model", model_label)
    GRIDSEARCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(GRIDSEARCH_RESULTS_PATH, mode="a" if append else "w",
              index=False, header=not append)


# ---------------------------------------------------------------------------
# Top-features helper (for the final winning linear model, if any)
# ---------------------------------------------------------------------------

def top_features_linear(pipe: Pipeline, n: int = 30) -> pd.DataFrame:
    vec = pipe.named_steps["vect"]
    clf = pipe.named_steps["clf"]
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_.ravel()
    order = np.argsort(coefs)
    rows = []
    for i, idx in enumerate(order[-n:][::-1]):
        rows.append({
            "rank": i + 1,
            "direction": "petitioner (positive)",
            "feature": feature_names[idx],
            "coefficient": float(coefs[idx]),
        })
    for i, idx in enumerate(order[:n]):
        rows.append({
            "rank": i + 1,
            "direction": "respondent (negative)",
            "feature": feature_names[idx],
            "coefficient": float(coefs[idx]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    split = get_train_test_split()
    X_train, y_train = split.train_df["text"], split.y_train
    X_test, y_test = split.test_df["text"], split.y_test
    groups_train = split.groups_train

    logger.info("Train: %d rows / %d cases. Test: %d rows / %d cases.",
                len(X_train), len(set(groups_train)),
                len(X_test), len(set(split.groups_test)))
    logger.info("Parallelism: linear stages n_jobs=%d, RF outer n_jobs=%d",
                N_JOBS_LINEAR, N_JOBS_RF_OUTER)
    logger.info("Pipeline memory cache: %s", PIPELINE_CACHE_DIR)

    memory = _make_memory_cache()

    test_evals: list[dict] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    overall_t0 = time.monotonic()

    # ----- Stage 4A: LogReg -----
    logger.info("=" * 60)
    logger.info("Stage 4A — LogReg")
    logger.info("=" * 60)
    t0 = time.monotonic()
    gs_lr = run_stage_4a_logreg(X_train, y_train, groups_train, memory)
    elapsed_lr = time.monotonic() - t0
    logger.info("LogReg done in %.0fs (%.1f min). best CV AUC=%.4f, params=%s",
                elapsed_lr, elapsed_lr / 60, gs_lr.best_score_, gs_lr.best_params_)
    persist_cv_results(gs_lr, "logreg", append=False)
    test_evals.append(evaluate_on_test("logreg", gs_lr, X_test, y_test))
    fitted_pipelines["logreg"] = gs_lr.best_estimator_

    # ----- Stage 4A: LinearSVC -----
    logger.info("=" * 60)
    logger.info("Stage 4A — LinearSVC")
    logger.info("=" * 60)
    t0 = time.monotonic()
    gs_svm = run_stage_4a_svm(X_train, y_train, groups_train, memory)
    elapsed_svm = time.monotonic() - t0
    logger.info("SVM done in %.0fs (%.1f min). best CV AUC=%.4f, params=%s",
                elapsed_svm, elapsed_svm / 60, gs_svm.best_score_, gs_svm.best_params_)
    persist_cv_results(gs_svm, "linear_svc", append=True)
    test_evals.append(evaluate_on_test("linear_svc", gs_svm, X_test, y_test))
    fitted_pipelines["linear_svc"] = gs_svm.best_estimator_

    # ----- Pick best vectorizer config from Stage 4A -----
    if gs_lr.best_score_ >= gs_svm.best_score_:
        best_linear_label = "logreg"
        best_linear_gs = gs_lr
    else:
        best_linear_label = "linear_svc"
        best_linear_gs = gs_svm
    best_vec_params = {k: v for k, v in best_linear_gs.best_params_.items()
                       if k.startswith("vect__")}
    logger.info("Stage 4A winner (linear): %s with vec params %s",
                best_linear_label, best_vec_params)

    # ----- Stage 4B: RF with fixed vectorizer -----
    logger.info("=" * 60)
    logger.info("Stage 4B — RandomForest (vectorizer fixed at Stage 4A winner)")
    logger.info("=" * 60)
    t0 = time.monotonic()
    gs_rf = run_stage_4b_rf(X_train, y_train, groups_train, best_vec_params, memory)
    elapsed_rf = time.monotonic() - t0
    logger.info("RF done in %.0fs (%.1f min). best CV AUC=%.4f, params=%s",
                elapsed_rf, elapsed_rf / 60, gs_rf.best_score_, gs_rf.best_params_)
    persist_cv_results(gs_rf, "random_forest", append=True)
    test_evals.append(evaluate_on_test("random_forest", gs_rf, X_test, y_test))
    fitted_pipelines["random_forest"] = gs_rf.best_estimator_

    overall_elapsed = time.monotonic() - overall_t0
    logger.info("Phase 4 total wall-clock: %.0fs (%.1f min)",
                overall_elapsed, overall_elapsed / 60)

    # ----- Persist test-set evaluation -----
    pd.DataFrame(test_evals).to_csv(PHASE4_TEST_EVAL_PATH, index=False)
    logger.info("Wrote %s", PHASE4_TEST_EVAL_PATH)

    # ----- Pick overall winner by test AUC -----
    winner_row = max(test_evals, key=lambda r: r["test_roc_auc"])
    winner = winner_row["model"]
    winner_pipe = fitted_pipelines[winner]
    logger.info("=" * 60)
    logger.info("OVERALL Phase 4 winner: %s — test AUC=%.4f, CV AUC=%.4f",
                winner, winner_row["test_roc_auc"], winner_row["best_cv_score"])
    logger.info("=" * 60)

    # ----- Per-Justice AUC with bootstrap CIs -----
    per_just = per_justice_auc_with_bootstrap(
        winner_pipe, X_test, y_test, split.test_df["oyez_identifier"],
    )
    per_just.insert(0, "model", winner)
    per_just.to_csv(PHASE4_PER_JUSTICE_AUC_PATH, index=False)
    logger.info("Wrote %s", PHASE4_PER_JUSTICE_AUC_PATH)

    # ----- Top features (only for linear winners; RF feature importances
    #       go to a separate file if RF wins) -----
    if winner in ("logreg", "linear_svc"):
        top_feats = top_features_linear(winner_pipe, n=30)
        top_feats.insert(0, "model", winner)
        top_feats.to_csv(PHASE4_TOP_FEATURES_PATH, index=False)
        logger.info("Wrote %s", PHASE4_TOP_FEATURES_PATH)
    else:
        # RF: dump top 30 feature importances
        vec = winner_pipe.named_steps["vect"]
        clf = winner_pipe.named_steps["clf"]
        names = vec.get_feature_names_out()
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1][:30]
        rows = [{"model": winner, "rank": i + 1, "feature": names[idx],
                 "importance": float(importances[idx])}
                for i, idx in enumerate(order)]
        pd.DataFrame(rows).to_csv(PHASE4_TOP_FEATURES_PATH, index=False)
        logger.info("Wrote %s (RF importances)", PHASE4_TOP_FEATURES_PATH)

    # ----- Console summary -----
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY (cv_score, test_roc_auc, gap)")
    print("=" * 70)
    df_eval = pd.DataFrame(test_evals).sort_values("test_roc_auc", ascending=False)
    print(df_eval[["model", "best_cv_score", "test_roc_auc", "cv_test_gap",
                   "test_accuracy", "test_balanced_accuracy"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"\nOverall winner: {winner} (test AUC = {winner_row['test_roc_auc']:.4f})")
    print(f"Stage 4A wall-clock: {(elapsed_lr + elapsed_svm)/60:.1f} min")
    print(f"Stage 4B wall-clock: {elapsed_rf/60:.1f} min")
    print(f"Phase 4 total:       {overall_elapsed/60:.1f} min")

    # Clean up the pipeline cache directory
    if PIPELINE_CACHE_DIR.exists():
        shutil.rmtree(PIPELINE_CACHE_DIR)
        logger.info("Cleaned up pipeline cache at %s", PIPELINE_CACHE_DIR)


if __name__ == "__main__":
    main()
