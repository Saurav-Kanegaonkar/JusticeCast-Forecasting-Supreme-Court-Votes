# JusticeCast

**Forecasting Supreme Court Justice votes from oral-argument questions — a comparative study of text representations, structured around CRISP-DM.**

Course project for BAX 453. Part A (pitch deck, 15 pts) + Part B (reproducible Jupyter notebook, 20 pts).

---

## 1. What this project is

After every Supreme Court oral argument, appellate litigators spend hours "reading the bench" — inferring how each Justice is likely to vote based on the questions they asked. This is currently gut intuition by senior partners. Legal-tech vendors (Lex Machina, Bloomberg Law, Westlaw Edge, SCOTUSblog) sell adjacent products (judge ruling history, motion-success rates) but **no widely-available product converts oral-argument transcripts into per-Justice vote forecasts**.

JusticeCast asks: **how much "bench-reading" signal actually lives in the verbatim text of a Justice's questions?** And — more interestingly for legal-tech product strategy — does the answer depend on how you represent that text?

We built two parallel models on the same data, the same train/test split, and the same evaluation harness:

1. **The standard text-classification toolkit** the rubric required — TF-IDF + n-grams × Logistic Regression / LinearSVC / Random Forest, tuned with GridSearchCV
2. **A pre-trained sentence-embeddings track** — `all-MiniLM-L6-v2` (384-dim, no fine-tuning) feeding the same three classifier families

The contest between the two representations is the contribution. It's a comparative methodology study, not a single-track null result.

---

## 2. The findings

**1. The standard bag-of-words toolkit hits a ceiling around ROC AUC 0.53.** All 9 baseline (vectorizer × classifier) combinations land in 0.507–0.528. Tuning lifts that to 0.532. Bigrams and trigrams add nothing. Top features are thematic legal vocabulary (`officer`, `religious`, `jury`, `sentence`) — the model is partly learning case topic, not stance.

**2. Pre-trained sentence embeddings + Logistic Regression reach test ROC AUC 0.5691** — a **+3.7 percentage-point lift** over the tuned BoW winner, with no fine-tuning and no domain adaptation. The lift is **9× the entire BoW tuning gain**. A lightweight 80 MB encoder (MiniLM) plus a linear classifier outperforms a tuned 200K-feature TF-IDF + LinearSVC.

**3. The lift survives the strict honesty test.** On *contested* cases (where the case-prior doesn't pre-determine the vote and author-identity-plus-priors recovery is least useful), embeddings retain a **+4 pp per-Justice mean AUC gap** over BoW (0.532 → 0.576), and **13 of 15 Justices** are above chance with embeddings — versus 9 of 15 with BoW. The embedding gain is real bench-questioning signal, not just identity-recovery.

**4. The KBJackson centerpiece — the project's sharpest single finding.** With BoW, Ketanji Brown Jackson has the worst per-Justice contested AUC on the bench (0.405 — *below random*); the model can't recover any signal from her questioning. With pre-trained embeddings, her contested AUC jumps to **0.643**, a **+0.238 lift** on the strict test. Same Justice, same data, same labels — opposite conclusions about predictability depending solely on the representation. The most-engaged questioner (median 1,205 words/case, 96% speaking rate) produces text whose semantic structure pre-trained encoders capture but TF-IDF unigrams cannot.

**5. Mixed evidence across the bench, reported honestly.** Embeddings win for most Justices but lose for some. Thomas: BoW 0.494 → Embeddings 0.776 (+0.282) on contested cases — the silent Justice's few utterances carry strong stance signal that embeddings extract. Kennedy regresses: BoW 0.653 → Embeddings 0.525 (-0.128) — the long-time swing-justice voting was tightly correlated with thematic case content that BoW exploited but embeddings collapse in semantic space. *Not every Justice improves with embeddings.*

**6. The absolute level remains modest.** AUC 0.569 is better than chance, better than BoW, but still a weak predictor in isolation. The honest framing: **lower bound on bench-reading from text alone**, before sequence-aware models, audio features (tone, pace, hesitation from the .mp3 files), or structured case-feature integration.

### Why this matters for legal-tech product strategy

Don't sell a TF-IDF question-classifier. The right product uses pre-trained semantic representations *at minimum*. The marginal cost over BoW is one-time CPU encoding (~12 minutes for a corpus this size). The payoff is access to semantic structure that lexical features cannot reach. In a domain where 3–4 percentage points of AUC translate to material business decisions (which Justices to prep for, where to focus amicus efforts), that gap matters.

---

## 3. Repository structure

```
JusticeCast/
├── notebooks/
│   ├── JusticeCast_Final.ipynb          # SUBMISSION — CRISP-DM-structured (83 cells)
│   ├── 01_eda.ipynb                     # working — Phase 2 EDA + B1–B6 expansion
│   └── 02_phase5_comparative.ipynb      # working — Phase 5 dive
├── src/                                 # 17 modules (fetchers, builders, sweeps, eval)
│   └── modeling/splits.py               # canonical fold-0 test split (Non-Negotiable #15)
├── tests/                               # 9 pytest files (104 tests passing)
├── data/
│   ├── raw/                             # gitignored — SCDB CSV + Oyez JSON cache (~377 MB)
│   └── processed/
│       ├── justice_id_map.csv           # tracked — hand-built SCDB ↔ Oyez Justice key
│       ├── modeling_table.parquet       # gitignored — derived
│       ├── justice_case_rows.parquet    # gitignored — derived
│       └── embeddings/                  # gitignored — cached MiniLM + MPNet .npy
├── reports/
│   ├── proposal.md                      # 1-page proposal sent to professor (4/26)
│   ├── ml_canvas.pdf                    # ML Canvas v0.4 with CRISP-DM phase tags
│   ├── JusticeCast_Pitch.ppt            # Phase 7 pitch deck (PowerPoint, 11 slides; team-editable)
│   ├── checkpoint1_summary.md           # auto-generated Phase 1 report
│   ├── deck_assets/                     # 8 chart PNGs + 4 markdown specs (Phase 7)
│   └── results/                         # 25+ result CSVs across Phases 1–5
├── requirements.in / requirements.txt   # pinned Python 3.14 deps
├── STRUCTURE.md                         # one-page file-by-file repo reference
└── README.md                            # this file
```

For a full file-by-file map with one-line descriptions, see [`STRUCTURE.md`](STRUCTURE.md).

---

## 4. Method overview — CRISP-DM

The submission notebook (`notebooks/JusticeCast_Final.ipynb`) is organized around CRISP-DM's six phases as primary section headers (per Non-Negotiable #16):

| Phase | What it covers | Where it lives |
| --- | --- | --- |
| **1. Business Understanding** | Project objectives, success criteria, FN/FP cost asymmetry, rubric mapping | Notebook §1 |
| **2. Data Understanding** | Data sources (SCDB + Oyez), coverage profile, B1–B6 EDA highlights | Notebook §2; `notebooks/01_eda.ipynb` |
| **3. Data Preparation** | Codebook-verified label, custom stopword list, train/test split discipline | Notebook §3; `src/build_dataset.py`, `src/build_modeling_table.py`, `src/text_clean.py`, `src/modeling/splits.py` |
| **4. Modeling** | Two parallel tracks (BoW + embeddings) with baseline sweeps + GridSearchCV | Notebook §4; `src/phase3_baseline_sweep.py`, `src/phase4_gridsearch.py`, `src/phase45_baseline_sweep.py`, `src/phase45_gridsearch.py` |
| **5. Model Evaluation** | Standard metrics + the honesty triad (per-Justice contested-cases AUC) | Notebook §5; `src/phase5_evaluation.py`; `notebooks/02_phase5_comparative.ipynb` |
| **6. Model Deployment** | Deployment plan, monitoring cadence, methodological frontier | Notebook §6 |

The Machine Learning Canvas (`reports/ml_canvas.pdf`) tags each box with its corresponding CRISP-DM phase. The pitch deck (`reports/JusticeCast_Pitch.ppt`) flows loosely along the same six phases without naming them in headers.

---

## 5. How to reproduce

There are **two reproduction paths**. The fast path is what a grader, teammate, or recruiter will use to verify the deliverables. The full pipeline regenerates every artifact from the original SCDB + Oyez sources.

### Fast path — verify the submission deliverables in ~30 seconds

Use this if you just want to open the submission notebook and confirm it runs end-to-end. All heavy artifacts (cached embeddings, result CSVs, parquet files) are pre-computed and live in `data/processed/` and `reports/results/`.

```sh
git clone https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes
cd JusticeCast-Forecasting-Supreme-Court-Votes

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Register the venv as a Jupyter kernel so the notebook finds the right Python
python -m ipykernel install --user --name justicecast_venv --display-name "JusticeCast (.venv)"

# Open the submission notebook and Restart-and-Run-All.
# All cells execute from cached CSVs; no modeling re-runs.
jupyter lab notebooks/JusticeCast_Final.ipynb
# In Jupyter: Kernel → Restart & Run All
# Total wall-clock: under 30 seconds on a fresh kernel.

# Optional: verify the test suite
pytest                                    # 104 tests, ~20 sec
```

### Full pipeline — regenerate every artifact from source (~95 min)

Use this if you want to regenerate the complete corpus from SCDB + Oyez (e.g., to extend the term window, refresh after a new SCDB release, or audit the full pipeline). Total wall-clock is dominated by the 54-minute polite Oyez fetch.

```sh
# 0. Prerequisites: Python 3.14+, git
git clone https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes
cd JusticeCast-Forecasting-Supreme-Court-Votes

# 1. venv + pinned deps (~30 sec)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Register the venv as a Jupyter kernel for the submission notebook
python -m ipykernel install --user --name justicecast_venv --display-name "JusticeCast (.venv)"

# 2. Bulk-fetch SCDB + Oyez (≤ 1 req/sec global, sequential)
python -m src.run_bulk_fetch              # ~54 min for the 2005-2024 window
python -m src.rescue_failed_dockets       # ~1 min  — recovers Citizens United, Kiobel
python -m src.build_dataset               # ~1 min  — join SCDB ↔ Oyez
python -m src.build_modeling_table        # < 5 sec — apply Phase 2 cleanup

# 3. Encode embeddings (CPU-only, both models)
python -m src.compute_embeddings          # ~12 min combined (MiniLM 1 min + MPNet 11 min)

# 4. Run all modeling phases
python -m src.phase3_baseline_sweep       # ~30 sec — BoW 9-combo baseline
python -m src.phase4_gridsearch           # ~7 min  — BoW GridSearchCV (n_jobs=4)
python -m src.phase45_baseline_sweep      # ~2 min  — embeddings 6-combo baseline
python -m src.phase45_gridsearch          # ~17 min — embeddings GridSearchCV
python -m src.phase5_evaluation           # ~30 sec — refit + honesty triad
python -m src.build_comparative_summary   # < 5 sec — side-by-side artifacts
python -m src.build_ml_canvas             # < 5 sec — renders ml_canvas.pdf
python -m src.build_deck_charts           # < 5 sec — renders 8 deck PNGs

# 5. Open the submission notebook
jupyter lab notebooks/JusticeCast_Final.ipynb
# Kernel → Restart & Run All

# 6. Verify the test suite
pytest                                    # 104 tests, ~20 sec
```

**Wall-clock budget**, full pipeline:

| Step | Wall-clock |
| --- | --- |
| venv + deps | ~30 sec |
| Bulk Oyez fetch | ~54 min |
| Rescue + join + cleanup | ~2 min |
| Embeddings encoding | ~12 min |
| Modeling sweeps + GridSearchCV | ~27 min |
| Notebook execution + tests | ~1 min |
| **Total** | **~95 min** |

The bulk fetch is the only real-time bottleneck (Oyez API ≤ 1 req/sec across both fetch layers). Everything else is CPU-bound and runs comfortably on a laptop.

---

## 6. Team

BAX 453, Spring 2026 — six-person team:

- Saurav Kanegaonkar
- Amal Farhad Shaji
- Tanmay Kallakuri
- Vedant Tiwari
- Vedika Shetty
- Akansha Totre
