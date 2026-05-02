"""Tests for src.build_deck_charts — the Phase 7 deck-asset bundle.

Light-touch checks: every PNG in the spec exists after rendering, the
markdown bundle files are present, the chart-builder module wires up
without error. Image content is not byte-compared (matplotlib output
varies across systems); we only assert presence + minimum file size.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import build_deck_charts as bdc

DECK_ASSETS = Path("reports/deck_assets")


def test_all_pngs_listed_match_spec():
    """The ALL_PNGS list in the module is the source of truth for which
    PNGs Phase 7 produces. cai-plan v14 §7.1 specifies 8 charts."""
    assert len(bdc.ALL_PNGS) == 8
    assert "chart_bow_vs_embeddings_3slice.png" in bdc.ALL_PNGS
    assert "chart_per_justice_lift.png" in bdc.ALL_PNGS
    assert "chart_kbjackson_flip.png" in bdc.ALL_PNGS
    assert "chart_bow_baselines.png" in bdc.ALL_PNGS
    assert "chart_embeddings_baselines.png" in bdc.ALL_PNGS
    assert "chart_data_pipeline_funnel.png" in bdc.ALL_PNGS
    assert "data_flow_diagram.png" in bdc.ALL_PNGS
    assert "ml_canvas_summary.png" in bdc.ALL_PNGS


def test_theme_constants_locked():
    """Non-Negotiable #18: the palette is locked. Guard against accidental
    edits that would desynchronize the rendered charts from theme_spec.md."""
    assert bdc.C_BOW == "#2E5C8A"
    assert bdc.C_EMB == "#C9A961"
    assert bdc.C_NAVY == "#1A2E47"
    assert bdc.C_CREAM == "#FAF7F2"


# --- Output file checks (skip if bundle hasn't been built yet) ---

@pytest.fixture(scope="module")
def bundle_dir():
    if not DECK_ASSETS.exists():
        pytest.skip("reports/deck_assets/ missing — run `python -m src.build_deck_charts`")
    return DECK_ASSETS


@pytest.mark.parametrize("png_name", [
    "chart_bow_vs_embeddings_3slice.png",
    "chart_per_justice_lift.png",
    "chart_kbjackson_flip.png",
    "chart_bow_baselines.png",
    "chart_embeddings_baselines.png",
    "chart_data_pipeline_funnel.png",
    "data_flow_diagram.png",
    "ml_canvas_summary.png",
])
def test_each_png_present_and_nonempty(png_name, bundle_dir):
    p = bundle_dir / png_name
    assert p.exists(), f"{png_name} missing"
    # Each PNG should be at least 30 KB; charts at this resolution are larger
    assert p.stat().st_size > 30_000, f"{png_name} is suspiciously small"


@pytest.mark.parametrize("md_name", [
    "theme_spec.md",
    "slide_content_spec.md",
    "headline_numbers.md",
    "prompt_for_powerpoint_extension.md",
])
def test_each_markdown_spec_present(md_name, bundle_dir):
    p = bundle_dir / md_name
    assert p.exists(), f"{md_name} missing"
    # Each spec should have non-trivial content (>1 KB)
    assert p.stat().st_size > 1_000, f"{md_name} is suspiciously small"
