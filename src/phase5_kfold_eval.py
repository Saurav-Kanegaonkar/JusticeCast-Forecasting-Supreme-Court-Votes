"""5-fold cross-validated test AUC for both tracks (peer-review fix #1).

The original headline reported AUC on a single canonical fold-0 test set.
A reviewer pointed out that fold-0 might be a favorable fold for embeddings
(test AUC 0.569 was 2.9 pp ABOVE the embeddings CV mean of 0.540, while
BoW test AUC was 0.8 pp BELOW its CV mean of 0.540). This module addresses
that: for each of the 5 StratifiedGroupKFold folds, refit both winners on
the 4-fold training portion and score on the held-out test portion. Report
per-fold AUC for both tracks, mean ± std, and the per-fold paired
difference.

This DOES NOT redo hyperparameter tuning per fold (that would be nested
CV — much more expensive). It uses the winners selected on fold-0's CV
and asks: "do those same winners produce comparable lifts on the other 4
folds?"

Inputs:
    data/processed/modeling_table.parquet
    data/processed/embeddings/{minilm}.npy

Output:
    reports/results/phase5_kfold_evaluation.csv
        Columns: fold, n_train, n_test, auc_bow, auc_emb, diff_emb_bow

Usage:
    python -m src.phase5_kfold_eval
CRISP-DM phase: Model Evaluation.
Phase 5 — 5-fold test-AUC sweep for both tracks (peer-review robustness check).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.compute_embeddings import load_embeddings
from src.modeling.splits import (
    MODELING_TABLE_PATH, RANDOM_STATE, get_cv_splitter,
)
from src.phase5_evaluation import BOW_BEST_PARAMS, EMB_BEST_PARAMS
from src.text_clean import STOPWORDS_FOR_VECTORIZER, vectorizer_preprocessor

OUT_PATH = Path("reports/results/phase5_kfold_evaluation.csv")

logger = logging.getLogger(__name__)


def _make_bow_pipe() -> Pipeline:
    return Pipeline([
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


def _make_emb_clf() -> LogisticRegression:
    return LogisticRegression(
        **EMB_BEST_PARAMS["clf"],
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    df = pd.read_parquet(MODELING_TABLE_PATH).reset_index(drop=True)
    y = df["voted_petitioner"].astype(int).to_numpy()
    groups = df["caseId"].to_numpy()
    texts = df["text"].to_numpy()

    emb_matrix, _ = load_embeddings(EMB_BEST_PARAMS["embedding_key"])
    assert len(emb_matrix) == len(df), \
        f"embedding rows ({len(emb_matrix)}) != modeling table rows ({len(df)})"

    splitter = get_cv_splitter()
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(df.index.to_numpy(), y, groups=groups)
    ):
        # Hard leakage check
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE in fold {fold_idx}: {len(overlap)} caseIds in both"
            )

        y_tr, y_te = y[train_idx], y[test_idx]

        # BoW
        bow_pipe = _make_bow_pipe()
        bow_pipe.fit(texts[train_idx], y_tr)
        bow_score_te = bow_pipe.decision_function(texts[test_idx])
        auc_bow = roc_auc_score(y_te, bow_score_te)

        # Embeddings
        emb_clf = _make_emb_clf()
        emb_clf.fit(emb_matrix[train_idx], y_tr)
        emb_score_te = emb_clf.predict_proba(emb_matrix[test_idx])[:, 1]
        auc_emb = roc_auc_score(y_te, emb_score_te)

        diff = auc_emb - auc_bow
        rows.append(dict(
            fold=fold_idx,
            n_train=int(len(train_idx)),
            n_test=int(len(test_idx)),
            auc_bow=float(auc_bow),
            auc_emb=float(auc_emb),
            diff_emb_bow=float(diff),
        ))
        logger.info(
            "fold %d: n_test=%d  AUC bow=%.4f  AUC emb=%.4f  diff=%+0.4f",
            fold_idx, len(test_idx), auc_bow, auc_emb, diff,
        )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    logger.info("Wrote %s", OUT_PATH)

    # Aggregate
    print()
    print("=" * 76)
    print("5-FOLD CROSS-VALIDATED TEST AUC")
    print("=" * 76)
    print(out.to_string(index=False, float_format="%.4f"))
    print()
    print("Mean ± std across folds:")
    print(f"  AUC BoW:        {out['auc_bow'].mean():.4f} ± {out['auc_bow'].std(ddof=1):.4f}")
    print(f"  AUC Embeddings: {out['auc_emb'].mean():.4f} ± {out['auc_emb'].std(ddof=1):.4f}")
    print(f"  Diff (emb-bow): {out['diff_emb_bow'].mean():+.4f} ± "
          f"{out['diff_emb_bow'].std(ddof=1):.4f}")
    n_emb_wins = (out["diff_emb_bow"] > 0).sum()
    print(f"  Folds where embeddings > BoW: {n_emb_wins} / {len(out)}")

    # Paired t-test on per-fold diffs (H0: mean diff == 0)
    t_stat, p_val = stats.ttest_1samp(out["diff_emb_bow"], 0.0)
    df_resid = len(out) - 1
    t_crit = stats.t.ppf(0.975, df_resid)
    se = out["diff_emb_bow"].std(ddof=1) / np.sqrt(len(out))
    ci_lo = out["diff_emb_bow"].mean() - t_crit * se
    ci_hi = out["diff_emb_bow"].mean() + t_crit * se
    print()
    print("Paired t-test on per-fold diffs (H0: mean diff = 0):")
    print(f"  t-stat:  {t_stat:+.3f}  (df={df_resid})")
    print(f"  p-value: {p_val:.4f}")
    print(f"  95% CI for mean diff: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    if p_val < 0.05:
        print("  → Reject H0: 5-fold mean lift is statistically significant.")
    else:
        print("  → Fail to reject H0: 5-fold mean lift is within fold-to-fold noise.")


if __name__ == "__main__":
    main()
