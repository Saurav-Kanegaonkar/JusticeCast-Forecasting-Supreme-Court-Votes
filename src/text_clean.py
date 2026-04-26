"""Text-cleaning helpers for the modeling pipeline.

Currently minimal — Phase 2/3 will populate stopwords, stemming, and any
domain-specific normalization once EDA reveals what's actually in the text.
The vectorizers themselves handle most basic preprocessing (lowercase,
token pattern); this module is for things that don't belong inside an
sklearn `Pipeline`.
"""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces; strip ends."""
    return _WS_RE.sub(" ", text or "").strip()
