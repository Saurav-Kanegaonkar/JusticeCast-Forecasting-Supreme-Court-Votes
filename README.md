# JusticeCast

**Forecasting Supreme Court Justice votes from oral-argument questions — a comparative study of text representations, structured around CRISP-DM.**

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

**1. The standard bag-of-words toolkit hits a ceiling around ROC AUC 0.53.**
- All 9 baseline (vectorizer × classifier) combinations land in **0.507–0.528**
- Tuning via GridSearchCV lifts that only to **0.532** (+0.4 pp)
- Bigrams and trigrams add nothing — best vectorizer config is unigrams
- Top features are thematic legal vocabulary (`officer`, `religious`, `jury`, `sentence`) — the model is partly learning case topic, not stance

**2. Pre-trained sentence embeddings beat BoW by a small but reliably measurable margin.**

Three hypothesis tests, three lenses on the same effect:

| Test | What it asks | Mean diff | n | p-value | 95% CI for ΔAUC |
| --- | --- | :-: | :-: | :-: | :-: |
| **DeLong's paired** (fold 0) | "Is the gap on the canonical fold real, or noise within those 2,007 paired predictions?" | +0.0368 | 2,007 rows | **0.023** | [+0.005, +0.068] |
| **5-fold paired t** | "Does the gap persist across different held-out folds?" (canonical n=5) | +0.0139 | 5 folds | 0.18 | [-0.010, +0.038] |
| **10×5 repeated CV paired t** | "Same question, with enough resolution to detect a small effect." | **+0.0176** | 50 fold-realizations | **<0.001** | **[+0.014, +0.021]** |

**The headline is the bottom row.** With observed std ≈ 0.019 across folds and a true mean diff ≈ 0.014, a paired t-test on n=5 folds is genuinely underpowered (would need ≈ 15–20 folds to detect this size effect at α=0.05). Repeated CV (10 reps × 5 folds, different `random_state` per rep) gives 50 fold-realizations and tightens the CI without false-claiming more independent data — the effect is small (~1.4–2.1 pp), but consistent (47 of 50 realizations favor embeddings) and well-estimated.

- **Honest summary**: embeddings beat BoW by **~1.4–2.1 percentage points typical AUC lift** (10×5 repeated CV), the gap is **statistically significant** when tested with adequate resolution (p<0.001 at n=50), but it's **modest in absolute size**. The original "+3.7 pp" headline was the lucky end of the fold-realization distribution, not the typical effect.
- Lightweight 80 MB encoder (MiniLM, 384-dim) plus linear classifier ≈ tuned 200K-feature TF-IDF + LinearSVC; no fine-tuning, no GPU.
- Caveat on repeated CV: 50 realizations are NOT 50 independent samples — they're 50 different splits of the same dataset. The tighter CI is a better point estimate of generalization performance, not a power increase. The direction of the effect (+) is robust; the magnitude is well-estimated for *this* dataset.

**3. Contested-case slice is consistent with the headline story.**
- On *contested* cases (where the case-prior doesn't pre-determine the vote, removing the easy "everyone votes the modal way in unanimous cases" signal), embeddings retain a **+4 pp per-Justice mean AUC gap** over BoW on fold 0 (0.532 → 0.576)
- **13 of 15 Justices** above chance with embeddings on contested cases vs 9 of 15 with BoW (fold 0 point estimates; per-Justice CIs are wide and most cross 0.5 — see caveat below)
- Caveat on interpretation: contested-only filters out case-level prior recovery, but does not isolate text signal from per-Justice base rates. A model can still recognize Justice-style writing and route through that Justice's individual base rate. To fully isolate "from text alone," you'd want leave-one-Justice-out — out of scope here.

**4. The KBJackson "biggest swing" — striking but underpowered (n=19).**
- With BoW on fold 0: KBJackson contested AUC = **0.405** (below random)
- With embeddings on fold 0: KBJackson contested AUC = **0.643** (+0.238)
- **Caveat: n = 19 contested test rows.** Our own bootstrap floor (`min_n_for_ci=30`) refuses to compute a CI here. With AUC SE ≈ 0.10–0.13 at this sample size, the +0.238 swing is approximately 2σ — interesting but **not conclusive**. We report it as the **biggest single-Justice swing** across the bench, not a robust per-Justice claim.
- KBJackson is the most-engaged questioner (median 1,205 words/case, 96% speaking rate); the *direction* of the swing is plausible. The *magnitude* would need a larger contested sample (more terms, more cases) to lock in.

**5. Mixed evidence across the bench, with wide per-Justice CIs.**
- Per-Justice point estimates favor embeddings for most Justices, but **most per-Justice 95% CIs cross 0.5 in both tracks** — read as directional, not definitive.
- **Thomas** has the largest per-Justice point lift: contested AUC BoW 0.494 → Embeddings 0.776 (+0.282)
- **Kennedy regresses** on point estimate: BoW 0.653 → Embeddings 0.525 (−0.128); the long-time swing-justice voting was tightly correlated with thematic case content
- Treat individual-Justice numbers as exploratory; the aggregate story is what's robust.

**6. The absolute level remains modest.**
- AUC ~0.55 (5-fold mean) / 0.57 (fold 0) is better than chance and better than BoW, but a weak predictor in isolation
- Honest framing: **a lower bound on bench-reading from text alone**, not a deployable forecasting tool
- Out-of-scope frontiers that should improve it: sequence-aware models, audio features (tone, pace, hesitation from the Oyez .mp3 files), structured case-feature integration

### Why this matters for legal-tech product strategy

- **Don't sell a TF-IDF question-classifier as a leap forward** — pre-trained semantic representations are at least as good and probably slightly better, with low marginal cost (~12 min CPU encoding for this corpus, no fine-tuning, no GPU).
- **The margin is small enough to flip across folds**, so don't overclaim a "big lift." The right framing is: pre-trained embeddings should be the floor, not BoW; the practical payoff in this domain is access to semantic structure that lexical features cannot reach, even if the AUC gap is modest.

---

## 3. Repository structure

```
JusticeCast/
├── notebooks/
│   ├── JusticeCast_Final.ipynb          # SUBMISSION — CRISP-DM-structured (83 cells)
│   └── 01_eda.ipynb                     # working — Phase 2 EDA + B1–B6 expansion
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
│   ├── phase1_data_audit.md             # auto-generated Phase 1 data-audit report
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
| **5. Model Evaluation** | Standard metrics + the honesty triad (per-Justice contested-cases AUC) | Notebook §5; `src/phase5_evaluation.py` |
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
# 0. Prerequisites: Python 3.11+ (project pinned to 3.14 in requirements.txt; 3.11 onward works), git
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
python -m src.phase5_delong               # < 5 sec — DeLong's paired AUC test
python -m src.phase5_kfold_eval           # ~30 sec — canonical 5-fold robustness sweep
python -m src.phase5_kfold_eval --n-reps 10  # ~5 min — 10×5 repeated CV (load-bearing test)
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
