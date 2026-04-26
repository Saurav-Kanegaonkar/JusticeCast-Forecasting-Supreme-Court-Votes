"""Tests for src.text_clean — preprocessor and stopword config."""
from __future__ import annotations

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src import text_clean


def test_collapse_whitespace_basic():
    assert text_clean.collapse_whitespace("a   b\n\tc") == "a b c"
    assert text_clean.collapse_whitespace("") == ""
    assert text_clean.collapse_whitespace("   ") == ""


def test_preprocess_strips_bracket_annotations():
    cases = [
        ("Then -- [Laughter] -- I asked.",  "Then -- -- I asked."),
        ("[Laughter]",                       ""),
        ("[Crosstalk] What is the rule?",    "What is the rule?"),
        ("No issue here.",                   "No issue here."),
        ("Multiple [Laughter] in [Crosstalk] one.", "Multiple in one."),
    ]
    for raw, expected in cases:
        assert text_clean.preprocess_text(raw) == expected, (
            f"preprocess_text({raw!r}) -> {text_clean.preprocess_text(raw)!r}, "
            f"expected {expected!r}"
        )


def test_preprocess_is_idempotent():
    text = "First [Laughter] then [Crosstalk] question."
    once = text_clean.preprocess_text(text)
    twice = text_clean.preprocess_text(once)
    assert once == twice, "preprocess_text must be idempotent"


def test_preprocess_handles_none_and_empty():
    assert text_clean.preprocess_text("") == ""
    assert text_clean.preprocess_text(None) == ""


def test_stopwords_includes_sklearn_english():
    sw = set(text_clean.STOPWORDS_FOR_VECTORIZER)
    assert "the" in sw
    assert "and" in sw
    assert "is" in sw
    # sklearn ENGLISH_STOP_WORDS is fully contained
    assert set(ENGLISH_STOP_WORDS).issubset(sw)


def test_stopwords_includes_custom_overlay():
    sw = set(text_clean.STOPWORDS_FOR_VECTORIZER)
    # US states
    for s in ("illinois", "idaho", "california", "texas"):
        assert s in sw
    # Agency abbreviations
    for a in ("epa", "fcc", "ada", "bia", "pto"):
        assert a in sw
    # Famous case shortnames
    for c in ("miranda", "tinker", "chevron"):
        assert c in sw


def test_stopwords_does_not_overstrip_thematic_legal_vocab():
    """We deliberately keep thematic legal vocabulary out of the stopword list
    because those terms can carry stance through context. Stopwording them
    would cripple the model."""
    sw = set(text_clean.STOPWORDS_FOR_VECTORIZER)
    for keep in ("officer", "warrant", "jury", "school", "religious",
                 "sentence", "attorney", "circuit", "petitioner",
                 "respondent"):
        assert keep not in sw, (
            f"{keep!r} should NOT be in stopwords — it's thematic legal "
            f"vocab that may carry stance signal"
        )


def test_stopwords_is_a_list_not_a_set():
    """sklearn requires a list (not a set) for deterministic ordering."""
    assert isinstance(text_clean.STOPWORDS_FOR_VECTORIZER, list)
    # And sorted for diffability
    assert text_clean.STOPWORDS_FOR_VECTORIZER == sorted(
        text_clean.STOPWORDS_FOR_VECTORIZER
    )
