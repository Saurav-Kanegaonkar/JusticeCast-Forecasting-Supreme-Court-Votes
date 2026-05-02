"""Text-cleaning helpers and stopword config for the modeling pipeline.

Two outputs that downstream code consumes:

1. `preprocess_text(s)` — applied to the modeling-table `text` column
   *before* it's saved to parquet. Strips Oyez transcription bracket
   annotations like `[Laughter]`, `[crosstalk]`, `[applause]`, etc.,
   then collapses whitespace.

2. `STOPWORDS_FOR_VECTORIZER` — frozenset to pass to sklearn vectorizers'
   `stop_words=` parameter in Phase 3. Union of sklearn's English
   stopwords plus a focused custom list of US state names, common
   federal agency abbreviations, and famous case shortnames.

Why custom stopwords: Phase 2B EDA (B1) showed clear content-term
dominance in per-class log-odds. Without stopwording domain-identifying
proper nouns and abbreviations, the model would partly memorize
topic-vs-outcome rather than learn stance-from-questioning. Stopwording
JUST these identifying terms (not thematic legal vocabulary) is the
narrow intervention.
CRISP-DM phase: Data Preparation.
Preprocessing + custom stopword list for the BoW track.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Preprocessing (applied at modeling-table build time)
# ---------------------------------------------------------------------------

# Oyez transcription artifacts: [Laughter], [Crosstalk], [Applause], etc.
# Match any single bracketed annotation; case-insensitive.
_BRACKET_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces; strip ends."""
    return _WS_RE.sub(" ", text or "").strip()


def preprocess_text(text: str) -> str:
    """Strip transcription artifacts and normalize whitespace.

    Applied to the modeling-table `text` column before parquet write.
    Idempotent — running it twice on the same text yields the same output.
    """
    if not text:
        return ""
    text = _BRACKET_ANNOTATION_RE.sub(" ", text)
    return collapse_whitespace(text)


# ---------------------------------------------------------------------------
# Advocate-name pattern stripping (Phase 3.5)
# ---------------------------------------------------------------------------

# Phase 3 surfaced advocate-name leakage in top features ('mr frederick',
# 'mr fisher'). Strip these patterns vectorizer-side via `preprocessor=` so
# we don't have to enumerate surnames or rebuild the parquet.
#
# Conservative scope: only the title forms that name advocates. Do NOT touch
# 'justice <surname>' (Justices addressing each other is a different signal).
_ADVOCATE_TITLE_RE = re.compile(
    # Title (with optional trailing period) + whitespace + capitalized surname.
    # Oyez transcripts use "Mr. Frederick" with period; the optional `\.?`
    # also catches "Mr Frederick" and "Mister Frederick".
    r"\b(?:mr|mrs|ms|mister|madam|madame|general)\.?\s+[a-z][a-z\-']+\b",
    flags=re.IGNORECASE,
)


def vectorizer_preprocessor(text: str) -> str:
    """Lowercase + strip advocate-name patterns. Pass to `preprocessor=`.

    sklearn's vectorizers call `preprocessor` on each document BEFORE
    tokenization. Stripping `mr frederick`-style spans here removes both
    the title and the surname token, which `stop_words=` alone (which acts
    on single tokens after tokenization) cannot do.
    """
    if not text:
        return ""
    text = text.lower()
    text = _ADVOCATE_TITLE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Stopwords for Phase 3 vectorizer config
# ---------------------------------------------------------------------------

# US state names that recur as case identifiers (e.g., "Illinois v. Wardlow").
# Lowercase, single-token only — sklearn's default tokenizer won't catch
# multi-word state names like "north carolina" anyway.
_US_STATE_NAMES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "ohio", "oklahoma", "oregon", "pennsylvania", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "wisconsin",
    "wyoming",
})

# Federal agency / department abbreviations that appear as case content
# tokens but carry no stance signal.
_AGENCY_ABBREVIATIONS = frozenset({
    "epa", "fcc", "fec", "ftc", "fda", "fbi", "cia", "nsa", "dhs", "doj",
    "irs", "ssa", "dol", "hud", "nlrb", "nrc", "ferc", "cms", "faa",
    "dot", "dod", "ada", "bia", "pto", "uspto", "hipaa", "ferpa",
    "erisa", "rico", "cwa", "esa", "nepa", "fmla", "fcra", "tcpa",
    "ihl", "ucmj",
})

# Famous case shortnames that show up as lowercase tokens when Justices
# cite precedent. These act as topic markers (e.g., "tinker" → student
# speech cases) that would let the model memorize topic-→-outcome.
_CASE_SHORTNAMES = frozenset({
    "miranda", "tinker", "gideon", "roe", "casey", "dobbs", "lemon",
    "batson", "terry", "daubert", "chevron", "obergefell", "heller",
    "loving", "lawrence", "windsor", "lopez", "kelo", "morrison",
    "garcetti", "bivens",
})

# Court-procedural terms that are vocabulary about the institution itself
# rather than about case stance. Optional — keeping the set tight.
_COURT_INSTITUTIONAL = frozenset({
    "scotus", "amicus", "amici", "certiorari", "cert", "remand",
    "remanded", "vacated",
})

# Final stopword list passed to sklearn vectorizers. We start from sklearn's
# built-in English list (covers "the", "and", "is", etc. plus ~300 others)
# and augment with our domain-identifying terms. sklearn requires a list
# (not a set) to avoid hashing-order issues across runs.
_CUSTOM_STOPWORDS = (
    _US_STATE_NAMES
    | _AGENCY_ABBREVIATIONS
    | _CASE_SHORTNAMES
    | _COURT_INSTITUTIONAL
)

STOPWORDS_FOR_VECTORIZER: list[str] = sorted(
    set(ENGLISH_STOP_WORDS) | _CUSTOM_STOPWORDS
)


def custom_stopword_overlay() -> frozenset[str]:
    """Just the custom additions, for documentation / EDA inspection."""
    return frozenset(_CUSTOM_STOPWORDS)
