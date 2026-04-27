# JusticeCast

**Comparative study of text representations for stance classification on SCOTUS oral arguments — bag-of-words vs sentence-transformer embeddings, structured around CRISP-DM.**

Course project (BAX 453). Part A (pitch deck, 15 pts) + Part B (reproducible Jupyter notebook, 20 pts).

## Headline finding

A lightweight pre-trained `all-MiniLM-L6-v2` (384-dim, ~80 MB, no fine-tuning) plus tuned LogisticRegression beats a tuned 200K-feature TF-IDF + LinearSVC by **+3.7 percentage points of ROC AUC** on the held-out test set (BoW 0.532 → Embeddings 0.569). The lift survives the strict honesty test: on **contested cases** (where the case-prior doesn't pre-determine the vote), embeddings retain a +4 pp per-Justice mean AUC gap (0.532 → 0.576), and 13 of 15 Justices are above chance with embeddings (vs 9 of 15 with BoW).

The project's sharpest single finding: **KBJackson's predictability flips with the representation**. With BoW, her per-Justice contested AUC is 0.405 (worst on bench, *below* random). With pre-trained embeddings, it jumps to 0.643 — same Justice, same data, opposite conclusions about predictability depending solely on the text representation.

## Method overview — CRISP-DM

The submission notebook is organized around CRISP-DM's six phases as primary section headers (per the rubric's "Apply Data Science Process Model as a guide" requirement, codified as Non-Negotiable #16):

| Phase | What it covers | Notebook section |
| --- | --- | --- |
| **1. Business Understanding** | Project objectives, success criteria, stakeholders, FN/FP cost asymmetry | §1 |
| **2. Data Understanding** | SCDB + Oyez sources, coverage profile, B1–B6 EDA findings | §2 |
| **3. Data Preparation** | Codebook-verified label, custom stopword list, train/test split discipline | §3 |
| **4. Modeling** | Two parallel tracks (BoW + embeddings) with baseline sweeps + GridSearchCV | §4 |
| **5. Model Evaluation** | Standard metrics + the honesty triad (per-Justice contested-cases AUC) | §5 |
| **6. Model Deployment** | Deployment plan, monitoring cadence, methodological frontier | §6 |

The ML Canvas (`reports/ml_canvas.pdf`) tags each box with its corresponding CRISP-DM phase. The pitch deck flow loosely tracks CRISP-DM (without naming the phases in slide headers — it's a pitch, not a process document).

## Reproduce from a fresh clone

```sh
# 0. Prerequisites: Python 3.14+, git
git clone https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes
cd JusticeCast-Forecasting-Supreme-Court-Votes

# 1. Set up venv + install pinned deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch SCDB + Oyez transcripts. Bulk fetch is ≤1 req/sec sequential.
python -m src.run_bulk_fetch         # ~54 min for 2005-2024 window
python -m src.rescue_failed_dockets  # ~1 min — recovers Citizens United + Kiobel
python -m src.build_dataset          # join SCDB ↔ Oyez (~1 min)
python -m src.build_modeling_table   # apply cleanup (instant)

# 3. Encode embeddings (CPU-only, ~12 min combined)
python -m src.compute_embeddings

# 4. Run all modeling phases
python -m src.phase3_baseline_sweep      # ~30 sec — BoW 9-combo baseline
python -m src.phase4_gridsearch          # ~7 min  — BoW GridSearchCV (n_jobs=4)
python -m src.phase45_baseline_sweep     # ~2 min  — embeddings 6-combo baseline
python -m src.phase45_gridsearch         # ~17 min — embeddings GridSearchCV
python -m src.phase5_evaluation          # ~30 sec — refit + honesty triad
python -m src.build_comparative_summary  # instant — side-by-side artifacts
python -m src.build_ml_canvas            # instant — renders ml_canvas.pdf

# 5. Open the submission notebook
jupyter lab notebooks/JusticeCast_Final.ipynb
# Then: Kernel → Restart & Run All  (clean execution in seconds because all
# heavy artifacts are pre-computed and cached)

# 6. Verify the test suite
pytest                                # 90 tests, ~15 sec
```

**Total reproduction time from scratch: ~95 min** (dominated by the 54-min bulk Oyez fetch). Cached, the pipeline runs in under 30 minutes end-to-end.

## Data sources

- **SCDB** (Supreme Court Database, Washington University) — Justice-Centered file, release 2025_01. HTTP-only host (HTTPS misconfigured on their end). Latin-1 / Windows-1252 encoded. Free CSV, no auth. Provides every recorded SCOTUS vote with petitioner/respondent winner, majority/dissent flags.
- **Oyez.org** REST API — `https://api.oyez.org`. Two-step fetch (case metadata → transcript audio JSON). Free, public, no auth. Polite limit ≤1 req/sec. Both layers cache to disk under `data/raw/` (gitignored).

## Repo layout

```
JusticeCast/
├── data/
│   ├── raw/                     # gitignored — SCDB CSV + Oyez JSON cache (377 MB)
│   └── processed/
│       ├── justice_id_map.csv   # tracked — hand-built SCDB ↔ Oyez Justice key
│       ├── modeling_table.parquet      # gitignored — derived
│       ├── justice_case_rows.parquet   # gitignored — derived
│       └── embeddings/                 # gitignored — cached MiniLM + MPNet .npy
├── src/
│   ├── modeling/
│   │   └── splits.py            # canonical fold-0 test split (Non-Negotiable #15)
│   ├── fetch_scdb.py            # SCDB downloader, latin-1 reader
│   ├── fetch_oyez.py            # 2-step Oyez fetcher with rate limit + retries
│   ├── rescue_failed_dockets.py # term ±1 rescue pass (Stop C)
│   ├── build_dataset.py         # join SCDB ↔ Oyez → justice_case_rows.parquet
│   ├── build_modeling_table.py  # apply cleanup decisions → modeling_table.parquet
│   ├── compute_embeddings.py    # encode text with sentence-transformers
│   ├── text_clean.py            # preprocess_text + STOPWORDS_FOR_VECTORIZER
│   ├── phase3_baseline_sweep.py # BoW 9-combo baseline
│   ├── phase4_gridsearch.py     # BoW GridSearchCV (sequential strategy)
│   ├── phase45_baseline_sweep.py# embeddings 6-combo baseline
│   ├── phase45_gridsearch.py    # embeddings GridSearchCV
│   ├── phase5_evaluation.py     # refit winners + honesty triad
│   ├── build_comparative_summary.py # side-by-side artifacts for Phase 5
│   ├── build_ml_canvas.py       # render reports/ml_canvas.pdf
│   ├── checkpoint1_analysis.py  # Phase 1 summary report
│   └── run_bulk_fetch.py        # Phase 1 Stop B driver
├── notebooks/
│   ├── 01_eda.ipynb                  # working — Phase 2/2B EDA
│   ├── 02_phase5_comparative.ipynb   # working — Phase 5 dive
│   └── JusticeCast_Final.ipynb       # SUBMISSION — CRISP-DM-structured
├── reports/
│   ├── proposal.md              # 1-page proposal for the professor (5/7)
│   ├── ml_canvas.pdf            # ML Canvas v0.4 with CRISP-DM phase tags
│   ├── JusticeCast_Pitch.pdf    # pitch deck (Phase 7 deliverable)
│   ├── checkpoint1_summary.md   # Phase 1 detailed report
│   └── results/                 # 24 result CSVs across Phases 1-5
├── tests/                       # 90 pytest tests across 8 files
├── cai-plan.md                  # the plan (CAI-owned, project ground truth)
├── CLAUDE.md                    # CC working notes (Claude Code)
├── project-state.md             # human-readable project status
├── requirements.in              # unpinned direct deps
├── requirements.txt             # pip freeze snapshot (Python 3.14)
└── README.md                    # this file
```

## Tests

```sh
pytest                            # 90 tests, ~15 sec
pytest -q                         # quiet mode
pytest tests/test_phase4_gridsearch.py    # specific file
```

The test suite covers split discipline (`test_splits.py`), label derivation (`test_builders.py`), fetcher mechanics (`test_fetchers.py`), modeling-table cleanup (`test_modeling_table.py`), text preprocessing (`test_text_clean.py`), baseline-sweep mechanics (`test_phase3_baseline.py`), GridSearchCV harness (`test_phase4_gridsearch.py`, `test_phase45.py`), embedding cache + alignment (`test_compute_embeddings.py`), and Phase 5 honesty-triad consistency (`test_phase5_evaluation.py`).

## Team

6-person course team. Member assignments live in CAI chat.
