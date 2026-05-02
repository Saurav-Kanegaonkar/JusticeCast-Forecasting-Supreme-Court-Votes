"""Phase 5 — comparative evaluation harness.

Refits both Phase 4 (BoW) and Phase 4.5 (embeddings) winners on the full
canonical training fold, then computes side-by-side metrics on the canonical
fold-0 test set:

  Honesty triad per Non-Negotiable #13:
    (a) Per-Justice global ROC AUC + bootstrap CIs   (already done in Phase 4 / 4.5)
    (b) Per-Justice ROC AUC split by unanimity        ← Phase 5 new
    (c) Per-Justice ROC AUC on contested cases only   ← Phase 5 new (the strict test)

  Standard suite:
    Confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve,
    calibration curve

  Embeddings interpretation:
    Top 20 highest-predicted (most confident petitioner) + 20 lowest
    (most confident respondent) test utterances, with case context.

Outputs:
    reports/results/phase5_honesty_triad.csv
    reports/results/phase5_per_justice_lift.csv
    reports/results/phase5_calibration_data.csv
    reports/results/phase5_roc_curve_data.csv
    reports/results/phase5_pr_curve_data.csv
    reports/results/phase5_confusion_matrices.csv
    reports/results/phase5_test_predictions.csv      (case_id, justice, true, score, pred for both tracks)
    reports/results/phase5_extreme_utterances.csv    (top 20 / bottom 20 by embedding score)

Usage:
    python -m src.phase5_evaluation
CRISP-DM phase: Model Evaluation.
Phase 5 — refit winners + honesty triad (per-Justice contested-cases AUC).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.compute_embeddings import load_embeddings
from src.modeling.splits import RANDOM_STATE, get_train_test_split
from src.text_clean import STOPWORDS_FOR_VECTORIZER, vectorizer_preprocessor

OUT_DIR = Path("reports/results")

# Phase 4 winner (BoW): LinearSVC with C=0.01, unigrams, min_df=5, max_df=0.9
BOW_BEST_PARAMS = dict(
    vec=dict(ngram_range=(1, 1), min_df=5, max_df=0.9),
    clf=dict(C=0.01),
)

# Phase 4.5 winner (embeddings): LogReg on MiniLM, C=100, l1_ratio=1.0
EMB_BEST_PARAMS = dict(
    embedding_key="minilm",
    clf=dict(C=100, l1_ratio=1.0),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Refit the two winners
# ---------------------------------------------------------------------------

def refit_bow(X_train_text, y_train) -> Pipeline:
    pipe = Pipeline([
        ("vect", TfidfVectorizer(
            **BOW_BEST_PARAMS["vec"],
            stop_words=STOPWORDS_FOR_VECTORIZER,
            preprocessor=vectorizer_preprocessor,
            lowercase=False,
        )),
        ("clf", LinearSVC(
            **BOW_BEST_PARAMS["clf"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )),
    ])
    pipe.fit(X_train_text, y_train)
    return pipe


def refit_embeddings(X_train_emb, y_train) -> LogisticRegression:
    clf = LogisticRegression(
        **EMB_BEST_PARAMS["clf"],
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_emb, y_train)
    return clf


# ---------------------------------------------------------------------------
# The honesty triad
# ---------------------------------------------------------------------------

def per_justice_auc(
    y_true: np.ndarray, y_score: np.ndarray, justices: pd.Series,
    n_bootstrap: int = 1000, min_n_for_ci: int = 30,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict] = []
    for j, idx in justices.groupby(justices).groups.items():
        idx = np.asarray(idx)
        n = len(idx)
        y_j = y_true[idx]
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
                yb, sb = y_j[resample], s_j[resample]
                if len(np.unique(yb)) < 2:
                    continue
                boots.append(roc_auc_score(yb, sb))
            if boots:
                ci_lo = float(np.percentile(boots, 2.5))
                ci_hi = float(np.percentile(boots, 97.5))
        rows.append({
            "oyez_identifier": j, "n": int(n), "point_auc": point,
            "ci_lo_95": ci_lo, "ci_hi_95": ci_hi,
        })
    return pd.DataFrame(rows)


def honesty_triad(
    track: str, test_df: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray,
) -> pd.DataFrame:
    """Per-Justice AUC under three slicings: global, unanimous-only, contested-only."""
    parts = []
    # (a) global
    pj_global = per_justice_auc(y_true, y_score, test_df["oyez_identifier"])
    pj_global["slice"] = "global"
    parts.append(pj_global)

    # (b) unanimous-only and contested-only via masks
    for slice_name, mask in [
        ("unanimous", test_df["unanimous"].to_numpy() == 1),
        ("contested", test_df["unanimous"].to_numpy() == 0),
    ]:
        if mask.sum() == 0:
            continue
        sub_test = test_df[mask].reset_index(drop=True)
        pj = per_justice_auc(y_true[mask], y_score[mask], sub_test["oyez_identifier"])
        pj["slice"] = slice_name
        parts.append(pj)

    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "track", track)
    return out


# ---------------------------------------------------------------------------
# Per-Justice lift over each Justice's individual baseline (Non-Negotiable #12)
# ---------------------------------------------------------------------------

def per_justice_lift(
    track: str, test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
    full_baselines: pd.DataFrame,
) -> pd.DataFrame:
    out = (
        test_df.assign(_pred=y_pred, _true=y_true)
               .groupby("oyez_identifier")
               .apply(lambda g: pd.Series({
                   "n_test_rows": len(g),
                   "model_accuracy": (g["_pred"] == g["_true"]).mean(),
               }), include_groups=False)
               .reset_index()
               .merge(full_baselines, on="oyez_identifier", how="left")
    )
    out["lift_over_baseline"] = out["model_accuracy"] - out["per_justice_baseline"]
    out.insert(0, "track", track)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = get_train_test_split()
    test_df = split.test_df
    y_train, y_test = split.y_train, split.y_test

    modeling_table = pd.read_parquet("data/processed/modeling_table.parquet")
    pet_rate = modeling_table.groupby("oyez_identifier")["voted_petitioner"].mean()
    full_baselines = pd.DataFrame({
        "oyez_identifier": pet_rate.index,
        "petitioner_rate_full": pet_rate.values,
    }).assign(per_justice_baseline=lambda d:
              d["petitioner_rate_full"].apply(lambda p: max(p, 1 - p)))

    # ----- BoW track refit + scores -----
    logger.info("Refitting BoW winner (LinearSVC, %s)...", BOW_BEST_PARAMS)
    t0 = time.monotonic()
    bow_pipe = refit_bow(split.train_df["text"], y_train)
    bow_fit_time = time.monotonic() - t0
    t1 = time.monotonic()
    bow_pred = bow_pipe.predict(split.test_df["text"])
    bow_score = bow_pipe.decision_function(split.test_df["text"])
    bow_predict_time = time.monotonic() - t1
    logger.info("BoW: fit=%.2fs predict=%.3fs", bow_fit_time, bow_predict_time)

    # For calibration curve only, we need probabilities. Wrap a fresh
    # LinearSVC inside CalibratedClassifierCV. Note: this is for the
    # calibration curve only — AUC and confusion matrix use the raw
    # decision_function from the unwrapped winner.
    logger.info("Calibrating BoW for calibration curve...")
    bow_calibrated_pipe = Pipeline([
        ("vect", TfidfVectorizer(
            **BOW_BEST_PARAMS["vec"],
            stop_words=STOPWORDS_FOR_VECTORIZER,
            preprocessor=vectorizer_preprocessor,
            lowercase=False,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                **BOW_BEST_PARAMS["clf"],
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            method="sigmoid",
            cv=5,
        )),
    ])
    bow_calibrated_pipe.fit(split.train_df["text"], y_train)
    bow_proba = bow_calibrated_pipe.predict_proba(split.test_df["text"])[:, 1]

    # ----- Embeddings track refit + scores -----
    logger.info("Refitting Embeddings winner (LogReg on MiniLM, %s)...",
                EMB_BEST_PARAMS["clf"])
    emb, _ = load_embeddings(EMB_BEST_PARAMS["embedding_key"])
    X_train_emb, X_test_emb = emb[split.train_idx], emb[split.test_idx]
    t0 = time.monotonic()
    emb_clf = refit_embeddings(X_train_emb, y_train)
    emb_fit_time = time.monotonic() - t0
    t1 = time.monotonic()
    emb_pred = emb_clf.predict(X_test_emb)
    emb_proba = emb_clf.predict_proba(X_test_emb)[:, 1]
    emb_predict_time = time.monotonic() - t1
    logger.info("Embeddings: fit=%.2fs predict=%.3fs", emb_fit_time, emb_predict_time)

    # ----- Standard suite (per-track scalars + curve data) -----
    summary_rows = []
    for track, pred, score in [("bow", bow_pred, bow_score),
                                ("embeddings", emb_pred, emb_proba)]:
        summary_rows.append({
            "track": track,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, score),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "phase5_summary_metrics.csv", index=False)
    logger.info("Wrote phase5_summary_metrics.csv")

    # ROC curve data (track, fpr, tpr)
    roc_chunks = []
    for track, score in [("bow", bow_score), ("embeddings", emb_proba)]:
        fpr, tpr, _ = roc_curve(y_test, score)
        roc_chunks.append(pd.DataFrame({"track": track, "fpr": fpr, "tpr": tpr}))
    pd.concat(roc_chunks, ignore_index=True).to_csv(
        OUT_DIR / "phase5_roc_curve_data.csv", index=False)

    # PR curve data
    pr_chunks = []
    for track, score in [("bow", bow_score), ("embeddings", emb_proba)]:
        prec, rec, _ = precision_recall_curve(y_test, score)
        pr_chunks.append(pd.DataFrame({"track": track, "precision": prec, "recall": rec}))
    pd.concat(pr_chunks, ignore_index=True).to_csv(
        OUT_DIR / "phase5_pr_curve_data.csv", index=False)

    # Calibration curve data (use probabilities for both tracks)
    cal_chunks = []
    for track, proba in [("bow", bow_proba), ("embeddings", emb_proba)]:
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        cal_chunks.append(pd.DataFrame({
            "track": track, "prob_predicted": prob_pred, "prob_true": prob_true,
        }))
    pd.concat(cal_chunks, ignore_index=True).to_csv(
        OUT_DIR / "phase5_calibration_data.csv", index=False)

    # Confusion matrices
    cm_rows = []
    for track, pred in [("bow", bow_pred), ("embeddings", emb_pred)]:
        cm = confusion_matrix(y_test, pred)
        for true_label in [0, 1]:
            for pred_label in [0, 1]:
                cm_rows.append({"track": track, "true_label": true_label,
                                "predicted_label": pred_label,
                                "count": int(cm[true_label, pred_label])})
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "phase5_confusion_matrices.csv", index=False)

    # ----- Honesty triad (the centerpiece) -----
    logger.info("Computing honesty triad (global/unanimity/contested)...")
    triad_chunks = [
        honesty_triad("bow", test_df, y_test, bow_score),
        honesty_triad("embeddings", test_df, y_test, emb_proba),
    ]
    pd.concat(triad_chunks, ignore_index=True).to_csv(
        OUT_DIR / "phase5_honesty_triad.csv", index=False)
    logger.info("Wrote phase5_honesty_triad.csv")

    # ----- Per-Justice lift, both tracks -----
    lift_chunks = [
        per_justice_lift("bow", test_df, y_test, bow_pred, full_baselines),
        per_justice_lift("embeddings", test_df, y_test, emb_pred, full_baselines),
    ]
    pd.concat(lift_chunks, ignore_index=True).to_csv(
        OUT_DIR / "phase5_per_justice_lift.csv", index=False)
    logger.info("Wrote phase5_per_justice_lift.csv")

    # ----- Test-set predictions (one row per test utterance, both tracks) -----
    preds_df = test_df[["caseId", "caseName", "oyez_identifier", "justiceName",
                         "term", "docket", "unanimous", "voted_petitioner",
                         "word_count", "text"]].copy()
    preds_df["bow_score"] = bow_score
    preds_df["bow_proba"] = bow_proba
    preds_df["bow_pred"] = bow_pred
    preds_df["emb_proba"] = emb_proba
    preds_df["emb_pred"] = emb_pred
    preds_df.to_csv(OUT_DIR / "phase5_test_predictions.csv", index=False)
    logger.info("Wrote phase5_test_predictions.csv")

    # ----- Extreme-score utterances for embeddings interpretation -----
    top20 = preds_df.nlargest(20, "emb_proba").copy()
    bot20 = preds_df.nsmallest(20, "emb_proba").copy()
    top20["extreme_class"] = "top_petitioner"
    bot20["extreme_class"] = "bottom_respondent"
    extremes = pd.concat([top20, bot20], ignore_index=True)
    extremes_cols = ["extreme_class", "caseId", "caseName", "justiceName",
                      "unanimous", "voted_petitioner", "emb_proba", "bow_proba",
                      "word_count", "text"]
    extremes[extremes_cols].to_csv(OUT_DIR / "phase5_extreme_utterances.csv", index=False)
    logger.info("Wrote phase5_extreme_utterances.csv")

    # ----- Console summary -----
    print("\n" + "=" * 70)
    print("PHASE 5 — COMPARATIVE SUMMARY")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format="%.4f"))

    triad = pd.concat(triad_chunks, ignore_index=True)
    print("\n=== Honesty triad: per-Justice ROC AUC, contested-only slice ===")
    contested = (triad[(triad.slice == "contested") & triad.point_auc.notna()]
                 .pivot(index="oyez_identifier", columns="track", values="point_auc")
                 .reset_index())
    contested["lift_emb_minus_bow"] = contested["embeddings"] - contested["bow"]
    contested = contested.sort_values("embeddings", ascending=False)
    print(contested.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
