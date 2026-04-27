# JusticeCast

**Forecasting Supreme Court Justice votes from oral-argument questions — a comparative study of text representations, structured around CRISP-DM.**

Course project for BAX 453. Part A (pitch deck, 15 pts) + Part B (reproducible Jupyter notebook, 20 pts).

---

## What this project is

After every Supreme Court oral argument, appellate litigators spend hours "reading the bench" — inferring how each Justice is likely to vote based on the questions they asked. This is currently gut intuition by senior partners. Legal-tech vendors (Lex Machina, Bloomberg Law, Westlaw Edge, SCOTUSblog) sell adjacent products (judge ruling history, motion-success rates) but **no widely-available product converts oral-argument transcripts into per-Justice vote forecasts**.

JusticeCast asks: **how much "bench-reading" signal actually lives in the verbatim text of a Justice's questions?** And — more interestingly for legal-tech product strategy — does the answer depend on how you represent that text?

We built two parallel models on the same data, the same train/test split, and the same evaluation harness:

1. **The standard text-classification toolkit** the rubric required — TF-IDF + n-grams × Logistic Regression / LinearSVC / Random Forest, tuned with GridSearchCV
2. **A pre-trained sentence-embeddings track** — `all-MiniLM-L6-v2` (384-dim, no fine-tuning) feeding the same three classifier families

The contest between the two representations is the contribution. It's a comparative methodology study, not a single-track null result.

---

## Headline findings

**1. The standard bag-of-words toolkit hits a ceiling around ROC AUC 0.53.** All 9 baseline (vectorizer × classifier) combinations land in 0.507–0.528. Tuning lifts that to 0.532. Bigrams and trigrams add nothing. Top features are thematic legal vocabulary (`officer`, `religious`, `jury`, `sentence`) — the model is partly learning case topic, not stance.

**2. Pre-trained sentence embeddings + Logistic Regression reach test ROC AUC 0.5691** — a **+3.7 percentage-point lift** over the tuned BoW winner, with no fine-tuning and no domain adaptation. The lift is **9× the entire BoW tuning gain**. A lightweight 80 MB encoder (MiniLM) plus a linear classifier outperforms a tuned 200K-feature TF-IDF + LinearSVC.

**3. The lift survives the strict honesty test.** On *contested* cases (where the case-prior doesn't pre-determine the vote and author-identity-plus-priors recovery is least useful), embeddings retain a **+4 pp per-Justice mean AUC gap** over BoW (0.532 → 0.576), and **13 of 15 Justices** are above chance with embeddings — versus 9 of 15 with BoW. The embedding gain is real bench-questioning signal, not just identity-recovery.

**4. The KBJackson centerpiece — the project's sharpest single finding.** With BoW, Ketanji Brown Jackson has the worst per-Justice contested AUC on the bench (0.405 — *below random*); the model can't recover any signal from her questioning. With pre-trained embeddings, her contested AUC jumps to **0.643**, a **+0.238 lift** on the strict test. Same Justice, same data, same labels — opposite conclusions about predictability depending solely on the representation. The most-engaged questioner (median 1,205 words/case, 96% speaking rate) produces text whose semantic structure pre-trained encoders capture but TF-IDF unigrams cannot.

**5. Mixed evidence across the bench, reported honestly.** Embeddings win for most Justices but lose for some. Thomas: BoW 0.494 → Embeddings 0.776 (+0.282) on contested cases — the silent Justice's few utterances carry strong stance signal that embeddings extract. Kennedy regresses: BoW 0.653 → Embeddings 0.525 (-0.128) — the long-time swing-justice voting was tightly correlated with thematic case content that BoW exploited but embeddings collapse in semantic space. *Not every Justice improves with embeddings.*

**6. The absolute level remains modest.** AUC 0.569 is better than chance, better than BoW, but still a weak predictor in isolation. The honest framing: **lower bound on bench-reading from text alone**, before sequence-aware models, audio features (tone, pace, hesitation from the .mp3 files), or structured case-feature integration.

### What this means for legal-tech product strategy

Don't sell a TF-IDF question-classifier. The right product uses pre-trained semantic representations *at minimum*. The marginal cost over BoW is one-time CPU encoding (~12 minutes for a corpus this size). The payoff is access to semantic structure that lexical features cannot reach. In a domain where 3–4 percentage points of AUC translate to material business decisions (which Justices to prep for, where to focus amicus efforts), that gap matters.

---

## Method — CRISP-DM

The submission notebook (`notebooks/JusticeCast_Final.ipynb`) is organized around CRISP-DM's six phases as primary section headers:

| Phase | What it covers | Section |
| --- | --- | --- |
| **1. Business Understanding** | Project objectives, success criteria, stakeholders, FN/FP cost asymmetry | §1 |
| **2. Data Understanding** | Sources (SCDB + Oyez), coverage profile, EDA findings | §2 |
| **3. Data Preparation** | Codebook-verified label, custom stopword list, train/test split discipline | §3 |
| **4. Modeling** | Two parallel tracks (BoW + embeddings) with baseline sweeps + GridSearchCV | §4 |
| **5. Model Evaluation** | Standard metrics + the honesty triad (per-Justice contested-cases AUC) | §5 |
| **6. Model Deployment** | Deployment plan, monitoring cadence, methodological frontier | §6 |

The Machine Learning Canvas (`reports/ml_canvas.pdf`) tags each box with its corresponding CRISP-DM phase.

---

## Data sources

- **Supreme Court Database (SCDB)**, Washington University, release 2025_01. Justice-Centered file: 83,644 vote rows × 61 columns. Latin-1 / Windows-1252 encoded. Free CSV, no auth.
- **Oyez.org** REST API. Two-step fetch: `cases/{term}/{docket}` (case metadata) → `case_media/oral_argument_audio/{audio_id}` (transcript with speaker-tagged turns). Free, public, no auth. Polite limit: ≤ 1 request/second.

The 2005–2024 OT window yields 1,470 unique `(term, docket)` pairs → 1,322 with valid oral argument → 10,308 joined `(case, Justice)` rows → **10,039 rows × 16 Justices in the final modeling table**, after cleanup.

Both API layers cache to disk under `data/raw/` (gitignored, 377 MB). Re-runs hit the cache, not the network.

---

## Reproduce from a fresh clone

```sh
# 0. Prerequisites: Python 3.14+, git
git clone https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes
cd JusticeCast-Forecasting-Supreme-Court-Votes

# 1. venv + pinned deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch data — bulk fetch is ≤1 req/sec sequential
python -m src.run_bulk_fetch          # ~54 min for the 2005-2024 window
python -m src.rescue_failed_dockets   # ~1 min  — recovers Citizens United, Kiobel
python -m src.build_dataset           # join SCDB ↔ Oyez (~1 min)
python -m src.build_modeling_table    # apply Phase 2 cleanup (instant)

# 3. Encode embeddings (CPU-only, both models)
python -m src.compute_embeddings      # ~12 min combined

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
pytest                                    # 90 tests, ~15 sec
```

**Total reproduction time from scratch: ~95 min** (dominated by the 54-min bulk Oyez fetch). With caches warm, the pipeline runs end-to-end in under 30 minutes.

---

## Repo layout

```
JusticeCast/
├── data/
│   ├── raw/                               # gitignored — SCDB CSV + Oyez JSON cache (~377 MB)
│   └── processed/
│       ├── justice_id_map.csv             # tracked — hand-built SCDB ↔ Oyez Justice key
│       ├── modeling_table.parquet         # gitignored — derived
│       ├── justice_case_rows.parquet      # gitignored — derived
│       └── embeddings/                    # gitignored — cached MiniLM + MPNet .npy
├── src/
│   ├── modeling/splits.py                 # canonical fold-0 test split (shared by both tracks)
│   ├── fetch_scdb.py                      # SCDB downloader, Latin-1 reader
│   ├── fetch_oyez.py                      # 2-step Oyez fetcher, rate-limited, retried, cached
│   ├── rescue_failed_dockets.py           # term ±1 rescue pass (recovers Citizens United)
│   ├── build_dataset.py                   # join SCDB ↔ Oyez → justice_case_rows.parquet
│   ├── build_modeling_table.py            # apply cleanup → modeling_table.parquet
│   ├── compute_embeddings.py              # encode text with sentence-transformers
│   ├── text_clean.py                      # preprocess_text + STOPWORDS_FOR_VECTORIZER
│   ├── phase3_baseline_sweep.py           # BoW 9-combo baseline
│   ├── phase4_gridsearch.py               # BoW GridSearchCV (sequential strategy)
│   ├── phase45_baseline_sweep.py          # embeddings 6-combo baseline
│   ├── phase45_gridsearch.py              # embeddings GridSearchCV
│   ├── phase5_evaluation.py               # refit winners + honesty triad
│   ├── build_comparative_summary.py       # side-by-side comparison artifacts
│   ├── build_ml_canvas.py                 # render reports/ml_canvas.pdf
│   ├── checkpoint1_analysis.py            # Phase 1 summary report
│   └── run_bulk_fetch.py                  # Phase 1 Stop B driver
├── notebooks/
│   ├── 01_eda.ipynb                       # working — Phase 2/2B EDA
│   ├── 02_phase5_comparative.ipynb        # working — Phase 5 dive
│   └── JusticeCast_Final.ipynb            # SUBMISSION — CRISP-DM-structured (78+ cells)
├── reports/
│   ├── proposal.md                        # 1-page proposal for the professor (5/7)
│   ├── ml_canvas.pdf                      # ML Canvas v0.4 with CRISP-DM phase tags
│   ├── JusticeCast_Pitch.pdf              # pitch deck (Part A deliverable)
│   ├── checkpoint1_summary.md             # Phase 1 detailed report
│   └── results/                           # 24 result CSVs across Phases 1-5
├── tests/                                 # 90 pytest tests across 8 files
├── requirements.in                        # unpinned direct deps
├── requirements.txt                       # pip freeze snapshot (Python 3.14)
└── README.md                              # this file
```

---

## Tests

```sh
pytest                                      # 90 tests, ~15 sec
pytest -q                                   # quiet mode
pytest tests/test_phase4_gridsearch.py      # specific file
```

The test suite covers split discipline (`test_splits.py`), label derivation (`test_builders.py`), fetcher mechanics (`test_fetchers.py`), modeling-table cleanup (`test_modeling_table.py`), text preprocessing (`test_text_clean.py`), baseline-sweep mechanics (`test_phase3_baseline.py`), GridSearchCV harness (`test_phase4_gridsearch.py`, `test_phase45.py`), embedding cache + alignment (`test_compute_embeddings.py`), and Phase 5 honesty-triad consistency (`test_phase5_evaluation.py`).

---

## Course / Team

BAX 453 course project. 6-person team.
