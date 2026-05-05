"""DeLong's paired test + paired bootstrap on the BoW-vs-Embeddings AUC gap.

Both models score the same fold-0 test rows, so the AUC difference is a
paired statistic. Fixed in response to peer review: the previous reporting
asserted "+3.7 pp lift" without a hypothesis test or CI on the difference.
This module computes both:

  1. DeLong's two-sided test for paired ROC AUC differences
     (DeLong, DeLong, Clarke-Pearson, 1988; fast vectorized form
     per Sun & Xu, 2014). Returns a Z statistic and p-value.

  2. Paired bootstrap (1000 resamples of test rows) → 95% CI for the
     AUC difference. Robust to violations of DeLong's asymptotic
     normality assumption when AUC is near 0.5.

Inputs:
    reports/results/phase5_test_predictions.csv  (paired predictions per row)

Output:
    reports/results/phase5_delong_test.csv
        Columns: auc_bow, auc_emb, diff_auc, delong_z, delong_p,
                 bootstrap_ci_lo, bootstrap_ci_hi, n_test_rows

Usage:
    python -m src.phase5_delong
CRISP-DM phase: Model Evaluation.
Phase 5 — paired AUC hypothesis test on the BoW-vs-Embeddings gap.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.modeling.splits import RANDOM_STATE

PRED_PATH = Path("reports/results/phase5_test_predictions.csv")
OUT_PATH = Path("reports/results/phase5_delong_test.csv")

N_BOOTSTRAP = 1000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeLong's algorithm (vectorized, Sun & Xu 2014 form)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (average ranks for ties). DeLong's Sun-Xu form."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _delong_structural_components(
    scores: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return AUC and the V_10, V_01 structural components for one model.

    scores: shape (n,)
    y:      shape (n,) binary 0/1
    Returns: (auc, V_pos, V_neg) where V_pos has shape (n_pos,) and
    V_neg has shape (n_neg,).
    """
    pos = scores[y == 1]
    neg = scores[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)

    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))

    auc = (tz[:n_pos].sum() / (n_pos * n_neg)) - (n_pos + 1) / (2.0 * n_neg)

    V_pos = (tz[:n_pos] - tx) / n_neg
    V_neg = 1.0 - (tz[n_pos:] - ty) / n_pos
    return auc, V_pos, V_neg


def delong_paired_test(
    scores_a: np.ndarray, scores_b: np.ndarray, y: np.ndarray
) -> dict:
    """DeLong's paired two-sided test for AUC difference.

    H0: AUC(A) = AUC(B). Returns dict with auc_a, auc_b, diff, z, p.
    """
    auc_a, Va_pos, Va_neg = _delong_structural_components(scores_a, y)
    auc_b, Vb_pos, Vb_neg = _delong_structural_components(scores_b, y)

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()

    # Covariance matrix of (auc_a, auc_b)
    S_pos = np.cov(np.stack([Va_pos, Vb_pos]), ddof=1)
    S_neg = np.cov(np.stack([Va_neg, Vb_neg]), ddof=1)
    S = S_pos / n_pos + S_neg / n_neg

    diff = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se_diff = float(np.sqrt(max(var_diff, 0.0)))
    z = float(diff / se_diff) if se_diff > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return dict(auc_a=float(auc_a), auc_b=float(auc_b),
                diff=float(diff), se_diff=se_diff, z=z, p=p)


# ---------------------------------------------------------------------------
# Paired bootstrap CI for AUC difference
# ---------------------------------------------------------------------------

def paired_bootstrap_diff(
    scores_a: np.ndarray, scores_b: np.ndarray, y: np.ndarray,
    n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Resample row indices with replacement; compute AUC diff each time.
    Returns (ci_lo, ci_hi) at 95%.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_a = roc_auc_score(yb, scores_a[idx])
        auc_b = roc_auc_score(yb, scores_b[idx])
        diffs.append(auc_a - auc_b)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    df = pd.read_csv(PRED_PATH)
    y = df["voted_petitioner"].astype(int).to_numpy()
    bow = df["bow_score"].to_numpy()
    emb = df["emb_proba"].to_numpy()

    # We test: is embeddings AUC > BoW AUC? Compute as (emb - bow); positive
    # diff = embeddings wins.
    delong = delong_paired_test(emb, bow, y)
    boot_lo, boot_hi = paired_bootstrap_diff(emb, bow, y)

    out = pd.DataFrame([{
        "auc_bow": delong["auc_b"],
        "auc_emb": delong["auc_a"],
        "diff_auc": delong["diff"],
        "se_diff": delong["se_diff"],
        "delong_z": delong["z"],
        "delong_p": delong["p"],
        "bootstrap_ci_lo": boot_lo,
        "bootstrap_ci_hi": boot_hi,
        "n_test_rows": int(len(y)),
        "n_bootstrap": N_BOOTSTRAP,
    }])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    logger.info("Wrote %s", OUT_PATH)

    print()
    print("=" * 70)
    print("PAIRED AUC TEST: Embeddings vs BoW on fold-0 test set")
    print("=" * 70)
    print(f"N test rows:       {len(y):,}")
    print(f"AUC (BoW):         {delong['auc_b']:.4f}")
    print(f"AUC (Embeddings):  {delong['auc_a']:.4f}")
    print(f"Difference:        {delong['diff']:+.4f}")
    print(f"SE of difference:  {delong['se_diff']:.4f}")
    print()
    print("DeLong's two-sided paired test:")
    print(f"  Z-statistic:     {delong['z']:+.3f}")
    print(f"  p-value:         {delong['p']:.4g}")
    print()
    print(f"Paired bootstrap (n={N_BOOTSTRAP}) 95% CI for diff:")
    print(f"  [{boot_lo:+.4f}, {boot_hi:+.4f}]")
    print()
    if delong["p"] < 0.05:
        print(f"INTERPRETATION: AUC difference is statistically significant "
              f"at α=0.05 (p={delong['p']:.4g}).")
    else:
        print(f"INTERPRETATION: AUC difference is NOT statistically "
              f"significant at α=0.05 (p={delong['p']:.4g}).")


if __name__ == "__main__":
    main()
