"""Phase 7 — Render the 8 deck-quality chart PNGs into reports/deck_assets/.

All charts share the same locked theme (deep navy + warm gold + cream
background, serif titles + sans-serif body) so the PowerPoint Claude
extension sees a consistent visual identity. Charts are deterministic: same
result-CSVs → same PNGs → same SHA256.

Usage:
    python -m src.build_deck_charts
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path("reports/results")
OUT_DIR = Path("reports/deck_assets")
EXISTING_FIGURES = Path("reports/figures")

# ---------------------------------------------------------------------------
# Locked theme (Non-Negotiable #18)
# ---------------------------------------------------------------------------

C_BOW = "#2E5C8A"           # deep blue — BoW track
C_EMB = "#C9A961"           # warm gold — Embeddings track
C_NAVY = "#1A2E47"          # navy — titles, header bars, accent
C_CREAM = "#FAF7F2"         # cream background
C_TEXT = "#1A1A1A"          # body text
C_TEXT_GREY = "#5A5A5A"     # subtitles
C_TEXT_LIGHT = "#9A9A9A"    # footers
C_RANDOM = "#888888"        # chance line

FONT_SERIF = "DejaVu Serif"     # widely available; substitutes for Lora/Playfair
FONT_SANS = "DejaVu Sans"       # widely available; substitutes for Inter/Source Sans

# 16:9 widescreen at 1920×1080 native; saved at dpi=160 (= exact 1920×1080).
# For retina sharpness in the deck, the extension can upscale or we save at
# 2x via savefig dpi=320 (= 3840×2160).
FIGSIZE_FULL = (12.0, 6.75)
FIGSIZE_HALF = (8.0, 6.0)        # for the focused KBJackson chart
SAVE_DPI = 200                   # 1920×1080-ish; high quality without bloat

logger = logging.getLogger(__name__)


def _set_global_style():
    plt.rcParams.update({
        "figure.facecolor":   C_CREAM,
        "axes.facecolor":     C_CREAM,
        "savefig.facecolor":  C_CREAM,
        "savefig.edgecolor":  C_CREAM,
        "axes.edgecolor":     C_NAVY,
        "axes.labelcolor":    C_TEXT,
        "axes.titlecolor":    C_NAVY,
        "xtick.color":        C_TEXT,
        "ytick.color":        C_TEXT,
        "text.color":         C_TEXT,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         "#D8D2C5",
        "grid.alpha":         0.6,
        "grid.linestyle":     "-",
        "grid.linewidth":     0.6,
        "font.family":        FONT_SANS,
        "font.size":          11,
        "axes.titlesize":     16,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "legend.frameon":     False,
        "legend.fontsize":    10,
    })


def _serif_title(ax, text, pad=14):
    ax.set_title(text, fontfamily=FONT_SERIF, color=C_NAVY,
                 fontweight="bold", pad=pad)


def _save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight",
                facecolor=C_CREAM, edgecolor=C_CREAM)
    plt.close(fig)
    logger.info("Wrote %s (%.0f KB)", out, out.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Chart 1 — BoW vs Embeddings on three slices (HEADLINE)
# ---------------------------------------------------------------------------

def chart_bow_vs_embeddings_3slice():
    triad = pd.read_csv(R / "phase5_honesty_triad.csv")
    p5 = pd.read_csv(R / "phase5_summary_metrics.csv")

    # Per-Justice mean by slice (matches notebook §5.4 aggregate)
    per_just = (triad[triad.point_auc.notna()]
                .groupby(["slice", "track"])["point_auc"].mean().unstack())

    # Global numbers come from the top-line test AUC, not the per-Justice mean
    global_bow = p5.loc[p5.track == "bow", "test_roc_auc"].iloc[0]
    global_emb = p5.loc[p5.track == "embeddings", "test_roc_auc"].iloc[0]

    slices = ["Global (test AUC)", "Unanimous", "Contested"]
    bow_vals = [global_bow, per_just.loc["unanimous", "bow"],
                per_just.loc["contested", "bow"]]
    emb_vals = [global_emb, per_just.loc["unanimous", "embeddings"],
                per_just.loc["contested", "embeddings"]]

    x = np.arange(len(slices))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    b1 = ax.bar(x - width / 2, bow_vals, width, color=C_BOW,
                edgecolor=C_NAVY, linewidth=0.8, label="BoW (TF-IDF + LinearSVC)")
    b2 = ax.bar(x + width / 2, emb_vals, width, color=C_EMB,
                edgecolor=C_NAVY, linewidth=0.8, label="Embeddings (MiniLM + LogReg)")
    ax.axhline(0.5, color=C_RANDOM, linestyle="--", linewidth=1.2,
               label="random baseline (AUC 0.50)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=10, color=C_TEXT)

    for xi, b, e in zip(x, bow_vals, emb_vals):
        lift_pp = (e - b) * 100
        ax.annotate(f"+{lift_pp:.1f} pp",
                    xy=(xi, max(b, e) + 0.024),
                    ha="center", fontsize=11, fontweight="bold",
                    color=C_NAVY,
                    fontfamily=FONT_SERIF)

    ax.set_xticks(x)
    ax.set_xticklabels(slices, fontsize=11)
    ax.set_ylim(0.45, 0.65)
    ax.set_ylabel("ROC AUC")
    _serif_title(ax, "Embedding lift survives the strict contested-cases test",
                 pad=18)
    ax.text(0.5, 1.04, "BoW vs Embeddings on identical fold-0 test rows",
            transform=ax.transAxes, ha="center", va="bottom",
            fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY, fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "chart_bow_vs_embeddings_3slice.png")


# ---------------------------------------------------------------------------
# Chart 2 — Per-Justice lift, both tracks
# ---------------------------------------------------------------------------

def chart_per_justice_lift():
    lift = pd.read_csv(R / "phase5_per_justice_lift.csv")
    pivot = (lift.pivot(index="oyez_identifier", columns="track",
                        values="lift_over_baseline")
             .reset_index().sort_values("embeddings", ascending=False))

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    y = np.arange(len(pivot))
    bow_bars = ax.barh(y - 0.20, pivot["bow"], height=0.40,
                       color=C_BOW, edgecolor=C_NAVY, linewidth=0.6,
                       label="BoW")
    emb_bars = ax.barh(y + 0.20, pivot["embeddings"], height=0.40,
                       color=C_EMB, edgecolor=C_NAVY, linewidth=0.6,
                       label="Embeddings")

    # Highlight KBJackson + Thomas
    highlight_idx = pivot.index[pivot["oyez_identifier"].isin(
        ["ketanji_brown_jackson", "clarence_thomas"])].tolist()
    for h_pos, j_pos in enumerate(pivot["oyez_identifier"]):
        if j_pos in ("ketanji_brown_jackson", "clarence_thomas"):
            bow_bars[h_pos].set_edgecolor(C_NAVY)
            bow_bars[h_pos].set_linewidth(1.8)
            emb_bars[h_pos].set_edgecolor(C_NAVY)
            emb_bars[h_pos].set_linewidth(1.8)

    ax.axvline(0, color=C_NAVY, linewidth=0.8, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot["oyez_identifier"], fontsize=10)
    ax.set_xlabel("Lift over per-Justice baseline (model accuracy − baseline)",
                  fontsize=11)
    _serif_title(ax,
                 "Per-Justice lift over individual baselines — both tracks",
                 pad=14)
    ax.text(0.5, 1.04,
            "KBJackson and Thomas highlighted (darker outline)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY, fontsize=10)
    ax.invert_yaxis()
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "chart_per_justice_lift.png")


# ---------------------------------------------------------------------------
# Chart 3 — KBJackson contested-cases flip (focused single-Justice spotlight)
# ---------------------------------------------------------------------------

def chart_kbjackson_flip():
    triad = pd.read_csv(R / "phase5_honesty_triad.csv")
    kbj = triad[(triad.oyez_identifier == "ketanji_brown_jackson")
                & (triad.slice == "contested")
                & triad.point_auc.notna()]
    bow_auc = float(kbj[kbj.track == "bow"]["point_auc"].iloc[0])
    emb_auc = float(kbj[kbj.track == "embeddings"]["point_auc"].iloc[0])
    lift = emb_auc - bow_auc

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    bars = ax.bar(["BoW", "Embeddings"], [bow_auc, emb_auc],
                  color=[C_BOW, C_EMB],
                  edgecolor=C_NAVY, linewidth=1.0, width=0.55)
    ax.axhline(0.5, color=C_RANDOM, linestyle="--", linewidth=1.2,
               label="random (AUC 0.50)")

    for bar, auc in zip(bars, [bow_auc, emb_auc]):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.012,
                f"{auc:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold",
                fontfamily=FONT_SERIF, color=C_NAVY)

    # Big gold lift annotation
    ax.text(0.5, 0.86,
            f"+{lift:.3f}", transform=ax.transAxes,
            ha="center", va="center",
            fontsize=42, fontweight="bold", color=C_EMB,
            fontfamily=FONT_SERIF)
    ax.text(0.5, 0.78,
            "AUC lift on contested cases", transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY)

    ax.set_ylim(0.0, 0.85)
    ax.set_ylabel("Per-Justice ROC AUC (contested cases)")
    _serif_title(ax, "KBJackson — same Justice, opposite verdicts", pad=14)
    ax.text(0.5, 1.04,
            "Contested-cases AUC: 0.405 (worst on bench) → 0.643 (3rd best)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY, fontsize=10)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "chart_kbjackson_flip.png")


# ---------------------------------------------------------------------------
# Chart 4 — BoW 9-combo baseline sweep
# ---------------------------------------------------------------------------

def chart_bow_baselines():
    bow = pd.read_csv(R / "baseline_results.csv").sort_values("roc_auc")
    fam_color = {"logreg": C_BOW, "linear_svc": "#2ca02c", "random_forest": "#c44"}
    fam_label = {"logreg": "LogReg", "linear_svc": "LinearSVC", "random_forest": "RandomForest"}
    bow["color"] = bow["classifier"].map(fam_color)

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    y = np.arange(len(bow))
    bars = ax.barh(y, bow["roc_auc"], color=bow["color"], edgecolor=C_NAVY, linewidth=0.6)
    ax.axvline(0.5, color=C_RANDOM, linestyle="--", linewidth=1.2,
               label="random (AUC 0.50)")
    for bar, auc in zip(bars, bow["roc_auc"]):
        ax.text(auc + 0.0015, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=9, color=C_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(bow["combo_id"], fontsize=10)
    ax.set_xlim(0.49, 0.55)
    ax.set_xlabel("Test ROC AUC")
    _serif_title(ax,
                 "Bag-of-words baselines cluster at 0.51-0.53 — a real ceiling",
                 pad=14)
    ax.text(0.5, 1.04,
            "9 combinations: 3 vectorizers × 3 classifiers, untuned",
            transform=ax.transAxes, ha="center", va="bottom",
            fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY, fontsize=10)
    handles = [mpatches.Patch(facecolor=c, edgecolor=C_NAVY, label=fam_label[k])
               for k, c in fam_color.items()]
    handles.append(plt.Line2D([0], [0], color=C_RANDOM, linestyle="--",
                              linewidth=1.2, label="random (0.50)"))
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "chart_bow_baselines.png")


# ---------------------------------------------------------------------------
# Chart 5 — Embeddings 6-combo baseline sweep
# ---------------------------------------------------------------------------

def chart_embeddings_baselines():
    emb = pd.read_csv(R / "embedding_baseline_results.csv").sort_values("roc_auc")
    bow_best = pd.read_csv(R / "baseline_results.csv")["roc_auc"].max()

    def _key(model_name): return "minilm" if "MiniLM" in model_name else "mpnet"
    emb["emb_key"] = emb["embedding_model"].map(_key)
    color_map = {"minilm": C_EMB, "mpnet": "#A47B2E"}  # gold + darker gold for mpnet
    label_map = {"minilm": "all-MiniLM-L6-v2 (384-dim)",
                 "mpnet":  "all-mpnet-base-v2 (768-dim)"}
    emb["color"] = emb["emb_key"].map(color_map)

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    y = np.arange(len(emb))
    bars = ax.barh(y, emb["roc_auc"], color=emb["color"], edgecolor=C_NAVY, linewidth=0.6)
    ax.axvline(bow_best, color=C_BOW, linestyle=":", linewidth=1.6,
               label=f"BoW baseline best ({bow_best:.3f})")
    ax.axvline(0.5, color=C_RANDOM, linestyle="--", linewidth=1.2,
               label="random (AUC 0.50)")
    for bar, auc in zip(bars, emb["roc_auc"]):
        ax.text(auc + 0.0015, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=9, color=C_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(emb["combo_id"], fontsize=10)
    ax.set_xlim(0.49, 0.59)
    ax.set_xlabel("Test ROC AUC")
    _serif_title(ax,
                 "Every embedding combination beats the BoW baseline best",
                 pad=14)
    ax.text(0.5, 1.04,
            "6 combinations: 2 embedding models × 3 classifiers, untuned",
            transform=ax.transAxes, ha="center", va="bottom",
            fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY, fontsize=10)
    handles = [mpatches.Patch(facecolor=c, edgecolor=C_NAVY, label=label_map[k])
               for k, c in color_map.items()]
    handles.append(plt.Line2D([0], [0], color=C_BOW, linestyle=":", linewidth=1.6,
                              label=f"BoW baseline best ({bow_best:.3f})"))
    handles.append(plt.Line2D([0], [0], color=C_RANDOM, linestyle="--", linewidth=1.2,
                              label="random (0.50)"))
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "chart_embeddings_baselines.png")


# ---------------------------------------------------------------------------
# Chart 6 — Data pipeline funnel
# ---------------------------------------------------------------------------

def chart_data_pipeline_funnel():
    stages = [
        ("Cases attempted (2005-2024 SCDB window)",   1470),
        ("Step 1 fetch succeeded",                    1420),
        ("Cases with ≥1 oral_argument_audio",         1322),
        ("Cases parsed into joined parquet",          1307),
        ("SCDB ↔ Oyez joined rows",                  10308),
        ("After drop NaN-label rows",                10137),
        ("After drop original-jurisdiction cases",   10120),
        ("Final modeling table (word_count ≥ 30)",   10039),
    ]
    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    # First 4 stages = case-level, last 4 = row-level
    colors = [C_BOW] * 4 + [C_EMB] * 4

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    bars = ax.barh(range(len(stages)), counts, color=colors,
                   edgecolor=C_NAVY, linewidth=0.6)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=10, color=C_TEXT)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1000, max(counts) * 1.5)
    ax.set_xlabel("Count (log scale)")
    _serif_title(ax, "Data pipeline funnel — fetch then cleanup", pad=14)
    ax.text(0.5, 1.04,
            "Case-level fetch (top 4) → row-level cleanup (bottom 4)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontfamily=FONT_SERIF, fontstyle="italic",
            color=C_TEXT_GREY, fontsize=10)
    handles = [mpatches.Patch(facecolor=C_BOW, edgecolor=C_NAVY,
                              label="case-level (Phase 1 fetch)"),
               mpatches.Patch(facecolor=C_EMB, edgecolor=C_NAVY,
                              label="row-level (Phase 2 cleanup)")]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "chart_data_pipeline_funnel.png")


# ---------------------------------------------------------------------------
# Chart 7 — data_flow_diagram.png (copy from existing)
# ---------------------------------------------------------------------------

def copy_data_flow_diagram():
    src = EXISTING_FIGURES / "justicecast_data_flow.png"
    dst = OUT_DIR / "data_flow_diagram.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)
        logger.info("Copied %s → %s", src, dst)
    else:
        logger.warning("Source %s missing — rendering a fallback diagram", src)
        _render_fallback_data_flow_diagram(dst)


def _render_fallback_data_flow_diagram(dst: Path):
    """If the existing figure is missing, render a simple diagram in matplotlib."""
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.75); ax.axis("off")

    def box(x, y, w, h, label, sub=None, color=C_BOW):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04",
                                    linewidth=1.2, edgecolor=C_NAVY,
                                    facecolor=color, alpha=0.16))
        ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center",
                fontsize=11, fontweight="bold",
                fontfamily=FONT_SERIF, color=C_NAVY)
        if sub:
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                    fontsize=9, color=C_TEXT_GREY, fontstyle="italic")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.4, color=C_NAVY))

    box(0.5, 4.6, 3.0, 1.4, "SCDB",
        "83,644 vote rows × 61 cols\nLatin-1, release 2025_01", color=C_BOW)
    box(0.5, 1.4, 3.0, 1.4, "Oyez API",
        "2-step fetch (case → audio)\n1,322 transcripts cached", color=C_EMB)
    box(4.5, 3.0, 3.0, 1.4, "Joined parquet",
        "(case × Justice) rows\n10,308 rows", color=C_NAVY)
    box(8.5, 3.0, 3.0, 1.4, "Modeling table",
        "10,039 rows × 16 Justices\n1,293 cases", color=C_NAVY)

    arrow(3.5, 5.3, 4.5, 4.0); arrow(3.5, 2.1, 4.5, 3.4)
    arrow(7.5, 3.7, 8.5, 3.7)

    ax.text(6.0, 0.5,
            "Both modeling tracks (BoW + Embeddings) consume an identical\n"
            "fold-0 test set via src/modeling/splits.py — no leakage, apples-to-apples.",
            ha="center", va="center", fontsize=10, color=C_TEXT_GREY,
            fontstyle="italic", fontfamily=FONT_SERIF)
    fig.savefig(dst, dpi=SAVE_DPI, bbox_inches="tight",
                facecolor=C_CREAM, edgecolor=C_CREAM)
    plt.close(fig)
    logger.info("Rendered fallback %s (%.0f KB)", dst, dst.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Chart 8 — ML Canvas summary as PNG
# ---------------------------------------------------------------------------

def render_ml_canvas_summary():
    """Render a deck-ready ML Canvas summary PNG. Re-uses the layout from
    src/build_ml_canvas.py but applies the deck theme + saves as PNG.
    """
    from src.build_ml_canvas import BOXES, GOAL_HEADER, GOAL_PHASE

    fig = plt.figure(figsize=FIGSIZE_FULL)
    fig.patch.set_facecolor(C_CREAM)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.84])
    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.axis("off")

    fig.text(0.5, 0.95, "Machine Learning Canvas — JusticeCast",
             ha="center", va="center", fontsize=18, fontweight="bold",
             color=C_NAVY, fontfamily=FONT_SERIF)
    fig.text(0.5, 0.91, f"Goal [{GOAL_PHASE}]:  {GOAL_HEADER}",
             ha="center", va="center", fontsize=10, color=C_TEXT,
             fontstyle="italic", fontfamily=FONT_SERIF)

    # Tag boxes Predict (left two cols) vs Learn (right two cols) with subtle bg
    for col_start, color in [(0, "#EAE3D2"), (2, "#D9E0E8")]:
        ax.add_patch(plt.Rectangle((col_start, 0), 2, 3,
                                   facecolor=color, edgecolor="none",
                                   alpha=0.45))

    for col, row, title, phase, body in BOXES:
        x, y = col * 1.0, (2 - row) * 1.0
        ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), 0.90, 0.90,
                                    boxstyle="round,pad=0.02,rounding_size=0.04",
                                    linewidth=0.9, edgecolor=C_NAVY,
                                    facecolor=C_CREAM))
        ax.text(x + 0.5, y + 0.88, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=C_NAVY,
                fontfamily=FONT_SERIF)
        ax.text(x + 0.5, y + 0.78, f"[{phase}]", ha="center", va="top",
                fontsize=7, fontstyle="italic", color=C_TEXT_GREY)
        ax.text(x + 0.10, y + 0.68, body, ha="left", va="top",
                fontsize=6.5, color=C_TEXT)

    fig.text(0.5, 0.02,
             "Each box tagged with its CRISP-DM phase. Predict (left) vs Learn (right) "
             "Canvas split.",
             ha="center", va="bottom", fontsize=8, color=C_TEXT_GREY,
             fontstyle="italic", fontfamily=FONT_SERIF)

    out = OUT_DIR / "ml_canvas_summary.png"
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight",
                facecolor=C_CREAM, edgecolor=C_CREAM)
    plt.close(fig)
    logger.info("Wrote %s (%.0f KB)", out, out.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_PNGS = [
    "chart_bow_vs_embeddings_3slice.png",
    "chart_per_justice_lift.png",
    "chart_kbjackson_flip.png",
    "chart_bow_baselines.png",
    "chart_embeddings_baselines.png",
    "chart_data_pipeline_funnel.png",
    "data_flow_diagram.png",
    "ml_canvas_summary.png",
]


def render_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_global_style()

    chart_bow_vs_embeddings_3slice()
    chart_per_justice_lift()
    chart_kbjackson_flip()
    chart_bow_baselines()
    chart_embeddings_baselines()
    chart_data_pipeline_funnel()
    copy_data_flow_diagram()
    render_ml_canvas_summary()

    missing = [p for p in ALL_PNGS if not (OUT_DIR / p).exists()]
    if missing:
        raise RuntimeError(f"Missing PNGs after build: {missing}")
    logger.info("All %d deck PNGs rendered to %s", len(ALL_PNGS), OUT_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    render_all()


if __name__ == "__main__":
    main()
