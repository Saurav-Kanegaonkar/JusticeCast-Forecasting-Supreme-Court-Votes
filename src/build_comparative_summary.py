"""Build the BoW vs embeddings side-by-side comparative summary CSV.

Reads Phase 4 + Phase 4.5 outputs and emits two artifacts ready for Phase 5
narrative work:

    reports/results/comparative_summary.csv         — top-line: one row per track winner
    reports/results/comparative_per_justice.csv     — long form: Justice × track
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PHASE4_TEST_EVAL = Path("reports/results/phase4_test_eval.csv")
PHASE4_PER_JUSTICE = Path("reports/results/phase4_per_justice_auc.csv")
PHASE45_TEST_EVAL = Path("reports/results/phase45_test_eval.csv")
PHASE45_PER_JUSTICE = Path("reports/results/phase45_per_justice_auc.csv")

OUT_SUMMARY = Path("reports/results/comparative_summary.csv")
OUT_PER_JUSTICE = Path("reports/results/comparative_per_justice.csv")

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    p4 = pd.read_csv(PHASE4_TEST_EVAL)
    p45 = pd.read_csv(PHASE45_TEST_EVAL)

    bow_winner = p4.loc[p4["test_roc_auc"].idxmax()].copy()
    emb_winner = p45.loc[p45["test_roc_auc"].idxmax()].copy()

    summary = pd.DataFrame([
        {
            "track": "BoW",
            "representation": "TF-IDF unigram + custom stopwords + advocate-name preprocessor",
            "classifier": bow_winner["model"],
            "best_params": bow_winner["best_params"],
            "cv_roc_auc": bow_winner["best_cv_score"],
            "test_roc_auc": bow_winner["test_roc_auc"],
            "test_balanced_accuracy": bow_winner["test_balanced_accuracy"],
            "test_accuracy": bow_winner["test_accuracy"],
            "test_f1": bow_winner["test_f1"],
            "cv_test_gap": bow_winner["cv_test_gap"],
        },
        {
            "track": "Embeddings",
            "representation": "all-MiniLM-L6-v2 (384-dim, pre-trained, no fine-tune)",
            "classifier": emb_winner["model"],
            "best_params": emb_winner["best_params"],
            "cv_roc_auc": emb_winner["best_cv_score"],
            "test_roc_auc": emb_winner["test_roc_auc"],
            "test_balanced_accuracy": emb_winner["test_balanced_accuracy"],
            "test_accuracy": emb_winner["test_accuracy"],
            "test_f1": emb_winner["test_f1"],
            "cv_test_gap": emb_winner["cv_test_gap"],
        },
    ])
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    logger.info("Wrote %s", OUT_SUMMARY)

    bow_pj = pd.read_csv(PHASE4_PER_JUSTICE).rename(columns={
        "point_auc": "bow_auc",
        "ci_lo_95": "bow_ci_lo",
        "ci_hi_95": "bow_ci_hi",
    })[["oyez_identifier", "n_test_rows", "bow_auc", "bow_ci_lo", "bow_ci_hi"]]
    emb_pj = pd.read_csv(PHASE45_PER_JUSTICE).rename(columns={
        "point_auc": "emb_auc",
        "ci_lo_95": "emb_ci_lo",
        "ci_hi_95": "emb_ci_hi",
    })[["oyez_identifier", "emb_auc", "emb_ci_lo", "emb_ci_hi"]]
    merged = bow_pj.merge(emb_pj, on="oyez_identifier", how="outer")
    merged["lift_emb_minus_bow"] = merged["emb_auc"] - merged["bow_auc"]
    merged = merged.sort_values("emb_auc", ascending=False)
    merged.to_csv(OUT_PER_JUSTICE, index=False)
    logger.info("Wrote %s", OUT_PER_JUSTICE)

    print("\n" + "=" * 70)
    print("COMPARATIVE TOP-LINE")
    print("=" * 70)
    print(summary[["track", "classifier", "cv_roc_auc", "test_roc_auc",
                   "test_balanced_accuracy"]].to_string(index=False, float_format="%.4f"))
    print(f"\nLift (emb - bow):  +{summary.iloc[1]['test_roc_auc'] - summary.iloc[0]['test_roc_auc']:.4f}")

    print("\n" + "=" * 70)
    print("PER-JUSTICE LIFT (sorted by embedding AUC)")
    print("=" * 70)
    cols = ["oyez_identifier", "n_test_rows", "bow_auc", "emb_auc", "lift_emb_minus_bow"]
    print(merged[cols].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
