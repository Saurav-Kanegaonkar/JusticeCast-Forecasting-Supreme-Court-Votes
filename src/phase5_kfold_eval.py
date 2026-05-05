"""5-fold cross-validated test AUC for both tracks (peer-review fix #1).

The original headline reported AUC on a single canonical fold-0 test set.
A reviewer pointed out that fold-0 might be a favorable fold for embeddings
(test AUC 0.569 was 2.9 pp ABOVE the embeddings CV mean of 0.540, while
BoW test AUC was 0.8 pp BELOW its CV mean of 0.540). This module addresses
that: for each StratifiedGroupKFold fold, refit both winners on the 4-fold
training portion and score on the held-out test portion. Report per-fold
AUC for both tracks, mean ± std, and the per-fold paired difference.

This DOES NOT redo hyperparameter tuning per fold (that would be nested
CV — much more expensive). It uses the winners selected on fold-0's CV
and asks: "do those same winners produce comparable lifts on the other 4
folds?"

Repeated CV (--n-reps N): a follow-up reviewer note flagged that with
n=5 folds and observed std ≈ 0.019, a paired t-test would need ≈ 15–20
folds to detect the observed mean diff (≈ 0.014) at α=0.05 — meaning
"p=0.18" doesn't mean "no effect," it means 5 folds isn't enough
resolution. Repeated CV with N reps × 5 folds (different `random_state`
per rep) tightens the CI on the mean lift estimate. Note: this is NOT
N×5 independent samples — within a rep the fold splits use one
random_state, and across reps you're sampling from the
"fold-realization distribution," not the data-generating process. So
treat the tighter CI as a better point estimate of generalization
performance, not as a power increase.

Inputs:
    data/processed/modeling_table.parquet
    data/processed/embeddings/{minilm}.npy

Outputs:
    reports/results/phase5_kfold_evaluation.csv  (when --n-reps=1; default)
        Columns: fold, n_train, n_test, auc_bow, auc_emb, diff_emb_bow
    reports/results/phase5_repeated_cv.csv       (when --n-reps>1)
        Columns: rep, fold, random_state, n_train, n_test, auc_bow, auc_emb, diff_emb_bow

Usage:
    python -m src.phase5_kfold_eval                # canonical 5-fold (1 rep)
    python -m src.phase5_kfold_eval --n-reps 10   # 10×5 repeated CV
CRISP-DM phase: Model Evaluation.
Phase 5 — k-fold test-AUC sweep for both tracks (peer-review robustness check).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.compute_embeddings import load_embeddings
from src.modeling.splits import (
    MODELING_TABLE_PATH, N_SPLITS, RANDOM_STATE, get_cv_splitter,
)
from src.phase5_evaluation import BOW_BEST_PARAMS, EMB_BEST_PARAMS
from src.text_clean import STOPWORDS_FOR_VECTORIZER, vectorizer_preprocessor

OUT_PATH_KFOLD = Path("reports/results/phase5_kfold_evaluation.csv")
OUT_PATH_REPEATED = Path("reports/results/phase5_repeated_cv.csv")

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


def _eval_one_split(
    train_idx: np.ndarray, test_idx: np.ndarray,
    texts: np.ndarray, emb_matrix: np.ndarray, y: np.ndarray,
    groups: np.ndarray, fold_label: str,
) -> dict:
    """Refit both winners on `train_idx`, score on `test_idx`, return AUCs.

    Hard leakage check via group overlap.
    """
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE in {fold_label}: {len(overlap)} caseIds in both train and test"
        )

    y_tr, y_te = y[train_idx], y[test_idx]

    bow_pipe = _make_bow_pipe()
    bow_pipe.fit(texts[train_idx], y_tr)
    bow_score_te = bow_pipe.decision_function(texts[test_idx])
    auc_bow = float(roc_auc_score(y_te, bow_score_te))

    emb_clf = _make_emb_clf()
    emb_clf.fit(emb_matrix[train_idx], y_tr)
    emb_score_te = emb_clf.predict_proba(emb_matrix[test_idx])[:, 1]
    auc_emb = float(roc_auc_score(y_te, emb_score_te))

    return dict(
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        auc_bow=auc_bow,
        auc_emb=auc_emb,
        diff_emb_bow=auc_emb - auc_bow,
    )


def run_kfold(
    df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, texts: np.ndarray,
    emb_matrix: np.ndarray, n_reps: int = 1,
) -> pd.DataFrame:
    """Run k-fold CV, optionally repeated. Returns a DataFrame.

    n_reps=1 → canonical 5-fold using the project's RANDOM_STATE=42.
    n_reps>1 → reps × 5 folds; rep r uses random_state=RANDOM_STATE+r.
    """
    rows: list[dict] = []
    for rep in range(n_reps):
        rs = RANDOM_STATE + rep
        if n_reps == 1:
            splitter = get_cv_splitter()  # canonical, RANDOM_STATE
        else:
            splitter = StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=rs,
            )

        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(df.index.to_numpy(), y, groups=groups)
        ):
            label = f"rep {rep} fold {fold_idx}" if n_reps > 1 else f"fold {fold_idx}"
            result = _eval_one_split(
                train_idx, test_idx, texts, emb_matrix, y, groups, label,
            )
            row = {"rep": rep, "fold": fold_idx, "random_state": rs, **result}
            rows.append(row)
            logger.info(
                "%s: n_test=%d  AUC bow=%.4f  AUC emb=%.4f  diff=%+0.4f",
                label, result["n_test"], result["auc_bow"],
                result["auc_emb"], result["diff_emb_bow"],
            )

    return pd.DataFrame(rows)


def _print_summary(out: pd.DataFrame, n_reps: int) -> None:
    label = f"{n_reps}×{N_SPLITS} REPEATED CV" if n_reps > 1 else f"{N_SPLITS}-FOLD CROSS-VALIDATED TEST AUC"
    print()
    print("=" * 76)
    print(label)
    print("=" * 76)

    if n_reps == 1:
        cols = ["fold", "n_train", "n_test", "auc_bow", "auc_emb", "diff_emb_bow"]
        print(out[cols].to_string(index=False, float_format="%.4f"))
    else:
        # Print per-rep mean diff and overall mean diff
        per_rep = (
            out.groupby("rep")
               .agg(mean_auc_bow=("auc_bow", "mean"),
                    mean_auc_emb=("auc_emb", "mean"),
                    mean_diff=("diff_emb_bow", "mean"))
               .reset_index()
        )
        print("Per-rep means:")
        print(per_rep.to_string(index=False, float_format="%.4f"))

    print()
    print(f"Aggregate over {len(out)} fold-realizations:")
    print(f"  AUC BoW:        {out['auc_bow'].mean():.4f} ± {out['auc_bow'].std(ddof=1):.4f}")
    print(f"  AUC Embeddings: {out['auc_emb'].mean():.4f} ± {out['auc_emb'].std(ddof=1):.4f}")
    print(f"  Diff (emb-bow): {out['diff_emb_bow'].mean():+.4f} ± "
          f"{out['diff_emb_bow'].std(ddof=1):.4f}")
    n_emb_wins = (out["diff_emb_bow"] > 0).sum()
    print(f"  Realizations where embeddings > BoW: {n_emb_wins} / {len(out)}")

    # Paired t-test on per-fold diffs (H0: mean diff == 0)
    t_stat, p_val = stats.ttest_1samp(out["diff_emb_bow"], 0.0)
    df_resid = len(out) - 1
    t_crit = stats.t.ppf(0.975, df_resid)
    se = out["diff_emb_bow"].std(ddof=1) / np.sqrt(len(out))
    ci_lo = out["diff_emb_bow"].mean() - t_crit * se
    ci_hi = out["diff_emb_bow"].mean() + t_crit * se
    print()
    print("Paired t-test on per-realization diffs (H0: mean diff = 0):")
    print(f"  t-stat:  {t_stat:+.3f}  (df={df_resid})")
    print(f"  p-value: {p_val:.4f}")
    print(f"  95% CI for mean diff: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    if p_val < 0.05:
        print("  → Reject H0: mean lift is statistically significant.")
    else:
        print("  → Fail to reject H0: mean lift is within realization-to-realization noise.")
    if n_reps > 1:
        print()
        print("Caveat: repeated CV gives a tighter CI on the mean-lift point estimate, "
              "but the realizations are NOT independent samples (same data, different "
              "split random_states). Treat as better generalization-performance estimate, "
              "not as a power boost.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=1,
                        help="Number of repeated-CV repetitions (1=canonical 5-fold).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    df = pd.read_parquet(MODELING_TABLE_PATH).reset_index(drop=True)
    y = df["voted_petitioner"].astype(int).to_numpy()
    groups = df["caseId"].to_numpy()
    texts = df["text"].to_numpy()

    emb_matrix, _ = load_embeddings(EMB_BEST_PARAMS["embedding_key"])
    assert len(emb_matrix) == len(df), \
        f"embedding rows ({len(emb_matrix)}) != modeling table rows ({len(df)})"

    out = run_kfold(df, y, groups, texts, emb_matrix, n_reps=args.n_reps)

    out_path = OUT_PATH_KFOLD if args.n_reps == 1 else OUT_PATH_REPEATED
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    _print_summary(out, n_reps=args.n_reps)


if __name__ == "__main__":
    main()
