"""Phase 4.5 — sentence-transformer encoding driver.

Encodes the modeling-table `text` column with two pre-trained models:
  - all-MiniLM-L6-v2  (384-dim, fast)
  - all-mpnet-base-v2 (768-dim, stronger)

NO fine-tuning. Pre-trained off-the-shelf encoders are the comparison —
out-of-the-box semantic representation vs hand-engineered BoW.
NO BoW preprocessing applied: sentence-transformers consume natural
language. The advocate-name preprocessor and stopword list are BoW-only.

Outputs:
    data/processed/embeddings/{model_name}.npy        # (n_rows, dim) float32
    data/processed/embeddings/row_index.parquet       # caseId, oyez_identifier ordering

Idempotent: skips encoding if cache exists with matching shape.

Usage:
    python -m src.compute_embeddings                  # encode both models
    python -m src.compute_embeddings --model mpnet    # encode only mpnet
    python -m src.compute_embeddings --model minilm
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDINGS_DIR = Path("data/processed/embeddings")
ROW_INDEX_PATH = EMBEDDINGS_DIR / "row_index.parquet"
MODELING_TABLE_PATH = Path("data/processed/modeling_table.parquet")

MODELS = {
    "minilm": ("all-MiniLM-L6-v2", 384),
    "mpnet":  ("all-mpnet-base-v2", 768),
}

DEVICE = "cpu"  # cai-plan: CPU build only, no GPU dependency

logger = logging.getLogger(__name__)


def _cache_path(model_key: str) -> Path:
    return EMBEDDINGS_DIR / f"{model_key}.npy"


def _matches_cache(path: Path, expected_shape: tuple[int, int]) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == expected_shape
    except Exception:
        return False


def encode_model(model_key: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Load model, encode all texts, return (n_rows, dim) float32 array."""
    from sentence_transformers import SentenceTransformer

    model_name, expected_dim = MODELS[model_key]
    logger.info("Loading %s on %s", model_name, DEVICE)
    t_load = time.monotonic()
    model = SentenceTransformer(model_name, device=DEVICE)
    logger.info("Loaded in %.1fs", time.monotonic() - t_load)

    t_enc = time.monotonic()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.monotonic() - t_enc
    logger.info("Encoded %d texts in %.0fs (%.1f min, %.2f sec/text)",
                len(texts), elapsed, elapsed / 60, elapsed / max(len(texts), 1))
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    assert emb.shape == (len(texts), expected_dim), (
        f"Got shape {emb.shape}, expected ({len(texts)}, {expected_dim})"
    )
    return emb


def compute_embeddings(model_key: str, force: bool = False) -> Path:
    """Encode and cache. Returns the .npy path."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MODELING_TABLE_PATH)
    n = len(df)
    _, dim = MODELS[model_key]
    cache = _cache_path(model_key)

    if not force and _matches_cache(cache, (n, dim)):
        logger.info("CACHE HIT for %s at %s — skipping encoding",
                    model_key, cache)
        return cache

    logger.info("CACHE MISS for %s — encoding %d rows", model_key, n)
    emb = encode_model(model_key, df["text"].tolist())
    np.save(cache, emb)
    logger.info("Saved %s (%.0f MB)", cache, cache.stat().st_size / 1024**2)

    # Persist canonical row index ONCE — same ordering for all models.
    if not ROW_INDEX_PATH.exists():
        idx = df[["caseId", "oyez_identifier", "term", "docket"]].copy()
        idx.to_parquet(ROW_INDEX_PATH, index=False)
        logger.info("Wrote row index %s", ROW_INDEX_PATH)
    return cache


def load_embeddings(model_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached embeddings + the row index. Asserts shape match."""
    arr = np.load(_cache_path(model_key))
    idx = pd.read_parquet(ROW_INDEX_PATH)
    assert len(arr) == len(idx), (
        f"Embedding rows ({len(arr)}) != row index ({len(idx)})"
    )
    expected_dim = MODELS[model_key][1]
    assert arr.shape[1] == expected_dim, (
        f"Got dim {arr.shape[1]}, expected {expected_dim} for {model_key}"
    )
    return arr, idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["minilm", "mpnet", "both"], default="both",
        help="Which model(s) to encode",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if cache exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    targets = ["minilm", "mpnet"] if args.model == "both" else [args.model]
    overall_t0 = time.monotonic()
    for model_key in targets:
        compute_embeddings(model_key, force=args.force)
    logger.info("All encoding done in %.0fs (%.1f min)",
                time.monotonic() - overall_t0,
                (time.monotonic() - overall_t0) / 60)


if __name__ == "__main__":
    main()
