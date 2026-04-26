# Project Plan: JusticeCast — Forecasting Supreme Court Votes

## Objective

Build a binary text-classification system that, given the verbatim oral-argument questions a Supreme Court Justice asks during a single case, predicts whether that Justice will vote with the petitioner or the respondent. Final deliverables are a polished, reproducible Jupyter notebook (Part B, 20 pts) and a pitch deck (Part A, 15 pts) framing the work as a legal-tech product. Total: 35 pts.

## Tech Stack

- Python 3.14.3, Jupyter Notebook (primary deliverable)
- `pandas==3.0.2`, `numpy==2.4.4` — data wrangling
- `scikit-learn==1.8.0` — `CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedGroupKFold`, `CalibratedClassifierCV`, all metrics
- `requests==2.33.1`, `beautifulsoup4==4.14.3`, `tenacity==9.1.4` — Oyez data fetching with retries and a polite rate limiter
- `matplotlib==3.10.9`, `seaborn==0.13.2` — visualization
- `nltk==3.9.4` — stopwords, stemming/lemmatization
- `joblib==1.5.3`, `pyarrow==24.0.0` — pipeline / fetched-data caching
- `pytest==9.0.3` — tests for fetchers and builders
- All deps pinned in `requirements.txt` (frozen) + tracked unpinned in `requirements.in`

## Data Sources

1. **Supreme Court Database (SCDB)** — Washington University, scdb.wustl.edu (HTTP only — site has misconfigured HTTPS; not a security concern for public read-only data). Justice-Centered file. Free CSV. Latest release: `2025_01`. Direct URL:
   ```
   http://scdb.wustl.edu/_brickFiles/2025_01/SCDB_2025_01_justiceCentered_Citation.csv.zip
   ```
   83,644 vote rows × 61 columns. **Encoding: Latin-1 / Windows-1252** (not UTF-8). Read with `pd.read_csv(path, encoding='latin1')`.

2. **Oyez.org API — TWO-STEP FETCH** (verified empirically against *Heien v. North Carolina*, term 2014, docket **13-604**):
   - **Step 1 — Case metadata:** `GET https://api.oyez.org/cases/{term}/{docket}` returns case-level JSON including `oral_argument_audio[]` array of links.
   - **Step 2 — Transcript:** for each entry in `oral_argument_audio[]`, follow the `href` to `https://api.oyez.org/case_media/oral_argument_audio/{audio_id}`.
   - **Multi-audio cases:** iterate over ALL audio entries per case and concatenate that Justice's utterances across argument sessions. Store the count as `n_audio_sessions` metadata.
   - **Cases without oral argument** (`oral_argument_audio == []`) are filtered out.

Joined on `(term, docket_number)`. Unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

**Justice ID mapping (SCDB ↔ Oyez):** SCDB uses numeric IDs (e.g., `80180`) and short codes (e.g., `HHBurton`); Oyez uses slugs (e.g., `john_g_roberts_jr`). Hand-built `data/processed/justice_id_map.csv` covers the 16 Justices appearing in 2005–2024.

## SCDB Field Semantics (verified Stop A)

Locked in from the SCDB codebook (cached HTML at `data/raw/scdb_codebook/`):

| Field | Semantics |
|---|---|
| `partyWinning` | `0`=petitioner LOST, `1`=petitioner WON, `2`=unclear (EXCLUDE from labels) |
| `majority` | `1`=dissent, `2`=majority, `NaN`=did not participate (EXCLUDE) |
| `vote` | 1..8 categorical (concurrence types). Not used directly. |
| `direction` | `1`=conservative, `2`=liberal. Not used in our binary label. |
| `caseDisposition` | 11-value taxonomy that already governs `partyWinning`. We use the derived `partyWinning`. |

**Final label derivation (locked in `src/build_dataset.py::derive_voted_petitioner`):**

```
voted_petitioner = (partyWinning == 1) == (majority == 2)
```

Returns `None` if either field is missing or `partyWinning == 2`.

⚠️ **Critical note:** SCDB's `majority` field is encoded `1=dissent, 2=majority` — counterintuitive. Phase 0's casual guess assumed the opposite. The Heien spot-check (Stop A) caught this; without it, every label in the dataset would have been silently inverted.

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/
│   │   ├── scdb_justice.csv                             # downloaded once, latin-1
│   │   ├── scdb_codebook/                               # cached codebook HTML
│   │   └── oyez/
│   │       ├── cases/{term}_{docket}.json               # case metadata (Step 1)
│   │       └── transcripts/{audio_id}.json              # transcript turns (Step 2)
│   └── processed/
│       ├── justice_id_map.csv                           # SCDB ↔ Oyez Justice key (16 rows)
│       ├── justice_case_rows.parquet                    # raw joined rows
│       └── modeling_table.parquet                       # post-EDA filtered table
├── src/
│   ├── fetch_scdb.py                                    # download + latin-1 read
│   ├── fetch_oyez.py                                    # 2-step fetch, rate-limited, retried, cached
│   ├── build_dataset.py                                 # join + aggregate per (case, Justice)
│   └── text_clean.py                                    # tokenization, stopwords, stemming
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── JusticeCast_Final.ipynb                          # SUBMISSION notebook
├── reports/
│   ├── proposal.md                                      # for the prof, due 5/7
│   ├── ml_canvas.pdf
│   ├── JusticeCast_Pitch.pdf
│   └── results/                                         # CSVs of every experiment run
├── tests/
│   ├── test_fetchers.py
│   └── test_builders.py
├── requirements.in                                      # unpinned direct deps
├── requirements.txt                                     # pinned (pip freeze)
├── README.md
├── CLAUDE.md
└── project-state.md
```

## Non-Negotiables

These rules apply across every phase. Violations are bugs, not preferences.

1. **No data leakage. Split by `case_id` using `StratifiedGroupKFold`.** All Justices for a given case go into the same split. `train_test_split(stratify=y)` does NOT respect groups — **do not use it for the primary split**. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`: take fold 0 as the held-out test set (~20%), use folds 1–4 as train. For nested CV inside `GridSearchCV`, also use `StratifiedGroupKFold` and pass `groups=case_id` to `.fit()`.
2. **Stratified splits.** Stratify on the binary vote label. `random_state=42` everywhere.
3. **Vectorizers fit on train only.** Use `sklearn.pipeline.Pipeline` so this is enforced by construction.
4. **No post-hoc features.** Only information available at the moment the Justice finished speaking. The vote label is the *only* future signal we touch.
5. **Reproducibility.** Fixed seed (42), pinned dependencies, notebook runs top-to-bottom on a fresh kernel via `Restart & Run All` with zero errors.
6. **Class imbalance handled explicitly.** Petitioner-side wins ~65–70% of SCOTUS cases. Use `class_weight='balanced'`. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged.** A `reports/results/` table with one row per (vectorizer, classifier, hyperparams) combination — accuracy, precision, recall, F1, ROC AUC, **per-fit wall-clock time**.
8. **Cache aggressively.** Oyez calls cached at both layers; SCDB downloaded once.
9. **Frame as Option 1 stance classification, not sentiment.** Same machinery, different label. Proposal makes this explicit.
10. **Hand-verify before bulk operations.** Any irreversible or expensive step (bulk API fetch, multi-hour grid search) gets a smoke test on a hand-checked sample first. Heien spot-check at Stop A caught the `majority` field inversion that would have flipped every label in the dataset — this rule paid for itself on first use.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE (5/7 deadline pending submission)

Repo: https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes

**Reminder for Saurav:** `reports/proposal.md` is drafted but still needs to be submitted to the professor by **5/7**.

### Phase 1: Data Acquisition (Two Stops)

#### Stop A — Pre-Bulk-Fetch Verification ✅ COMPLETE

**Gate passed:** Heien spot-check returned Sotomayor=1, all other 7 speaking Justices=0 (Thomas silent), `unanimous=0`, `n_audio_sessions=1`. Matches expected ground truth.

**Shipped:**
- `src/fetch_scdb.py` — download + Latin-1 read, idempotent
- `src/fetch_oyez.py` — 2-step fetch, global ≤1 req/sec rate limiter, tenacity retries (max 5, 5xx/429/timeout), cached at both layers
- `src/build_dataset.py` — parse Justice utterances, multi-audio concatenation, join, derive labels + unanimous flag
- `src/text_clean.py` — minimal whitespace helper (Phase 2/3 will populate)
- `data/processed/justice_id_map.csv` — 16 Justices (8 verified empirically, 8 standard-convention slugs to be verified at Stop B)
- `tests/test_fetchers.py` + `tests/test_builders.py` — 11 passing
- Codebook cached at `data/raw/scdb_codebook/`; semantics documented in `project-state.md`
- Truth-table test on `derive_voted_petitioner` covers all four label cases plus the two None cases

**Bulk-fetch budget for Stop B:**
- 13,149 SCDB rows in 2005–2024 window → 1,471 unique caseIds → 1,470 unique `(term, docket)` pairs (one duplicate, to investigate at Stop B)
- ~1,470 cases × 2 API calls each = ~2,940 calls
- Sequential at ≤1 req/sec ≈ 49 min; with retries + JSON parse: **55–70 min wall-clock**
- Cache size after bulk: ~750 MB (50 MB cases + 700 MB transcripts), gitignored

**Resolved during Stop A:**
- `majority` encoding is `1=dissent, 2=majority` (opposite of Phase 0 guess) — caught by Heien gate
- Phase 0's `13-1314` docket reference was wrong (that's *Arizona State Legislature v. AIRC*); real Heien is `13-604`. Stop A used the correct docket.
- CC's `partyWinning=6` from Phase 0 was an awk-on-quoted-CSV parsing bug; real value is `1`. Self-diagnosed in CC's prior turn.

#### Stop B — Bulk Fetch + Final Build → Checkpoint 1

- [ ] Bulk-fetch all 2005–2024 cases through `fetch_oyez.py` sequentially (both layers, both cached)
- [ ] Build the full joined parquet via `build_dataset.py` per the spec in Data Sources / Architecture
- [ ] Run all tests via `pytest` — must remain green

- **CHECKPOINT 1** — Report:
  - Final row counts at each stage (cases attempted → fetched → with valid oral arg → with all Justices mapped → final joined)
  - **Justice coverage cross-check (safety net for the 8 unverified slugs):** for each of the 16 Justices in `justice_id_map.csv`, report (a) expected case count from SCDB filtered to their tenure and our window, (b) actual joined-row count after parse. Flag any Justice with suspiciously low actual vs expected — likely indicates a wrong slug, fixable via CSV update + re-parse (no re-fetch).
  - **Thomas-specific count:** how many cases did Thomas actually speak in within 2005–2024? This determines whether per-Justice analysis for Thomas is statistically meaningful at Phase 5.
  - Median word counts per Justice (expect Thomas as a tail)
  - % of rows flagged unanimous
  - Final binary class distribution
  - Multi-audio cases encountered and how concatenation worked
  - **Duplicate `(term, docket)` pair from the budget estimate** — which case is it and why does the dedup happen
  - Any fetch or parse failures, with reasons
  - Total Oyez API calls made and elapsed wall-clock time
  - **Stop and wait.**

### Phase 2: EDA & Inclusion/Exclusion Decisions

- [ ] Produce `notebooks/01_eda.ipynb` with: vote-label distribution overall and per-Justice; word-count distributions per Justice; cases per term; petitioner-win rate per term; correlation between Justice talkativeness and predictability; **breakdown of label balance for unanimous vs contested cases**
- [ ] Decide inclusion/exclusion criteria (CC proposes, CAI reviews):
  - Drop rows with very low Justice word count? Empirical threshold from EDA.
  - **Unanimous cases: KEEP.** Flag via the `unanimous` metadata column. Phase 5 reports metrics split by unanimity.
  - **Thomas:** decide whether to include (with low-n caveat in Phase 5) or drop entirely. Driven by his actual speaking-case count from Checkpoint 1.
  - Drop other Justices with too few cases (recess appointments, partial-term Justices)?
- [ ] Build the **final modeling table** → `data/processed/modeling_table.parquet`
- [ ] Document decisions in `project-state.md`
- **CHECKPOINT 2:** Report final dataset shape, inclusion criteria applied, label balance overall and split by unanimity, expected baselines (majority-class accuracy as a floor). **Stop and wait.**

### Phase 3: Modeling Pipeline (Baseline Sweep)

- [ ] Build `Pipeline` objects for each (vectorizer × classifier) combination:
  - **Vectorizers (3):**
    - BoW: `CountVectorizer(ngram_range=(1,1))`
    - TF-IDF unigram: `TfidfVectorizer(ngram_range=(1,1))`
    - n-grams (TF-IDF bigrams): `TfidfVectorizer(ngram_range=(1,2))`
  - **Classifiers (3):**
    - `LogisticRegression(class_weight='balanced', max_iter=2000)`
    - `LinearSVC(class_weight='balanced')` — `decision_function` for AUC; `CalibratedClassifierCV` only for the calibration curve in Phase 5
    - `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
  - 9 combos total
- [ ] `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id` for the 80/20 split
- [ ] Train all 9, evaluate, log to `reports/results/baseline_results.csv` with per-fit wall-clock time
- [ ] Identify top 3 performers by ROC AUC for tuning
- **CHECKPOINT 3:** Report baseline results table with all 9 combinations, per-fit timing, identify top 3. **Stop and wait** — CAI reviews top-3 selection and approves the Phase 4 compute budget.

### Phase 4: GridSearchCV — Sequential Strategy

**Stage 4A: Two `GridSearchCV` runs (one per linear model) sharing the vectorizer parameter grid**

- [ ] LogReg run: `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` (with appropriate solver)
- [ ] SVM run: `C ∈ [0.01, 0.1, 1, 10]`
- [ ] Vectorizer (joint with both): `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`
- [ ] `StratifiedGroupKFold(n_splits=5)` with `groups=case_id`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] Record best vectorizer config from whichever linear model wins on CV ROC AUC

**Stage 4B: RF with fixed vectorizer**

- [ ] Vectorizer fixed at Stage 4A's winning config
- [ ] `n_estimators ∈ [100, 300, 500]`, `max_depth ∈ [None, 20, 50]`, `min_samples_split ∈ [2, 5, 10]`
- [ ] Same CV scheme

**Both stages:**

- [ ] Log to `reports/results/gridsearch_results.csv` with per-fit timing
- [ ] Refit best per model on full train, evaluate on held-out test
- [ ] Identify final winning model overall

- **CHECKPOINT 4:** Report best hyperparameters per model, CV vs test gap (overfitting check), final winning model, total Phase 4 compute time. **Stop and wait.**

### Phase 5: Evaluation & Interpretability

- [ ] For winning model: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
  - If winner is `LinearSVC`: wrap with `CalibratedClassifierCV(method='sigmoid', cv=5)` **only for the calibration curve**
- [ ] **Sensitivity analysis: per-Justice metrics split by unanimity.** For each Justice, report accuracy and ROC AUC separately on (a) unanimous cases, (b) contested cases. Discuss in prose.
- [ ] Per-Justice performance breakdown (overall) — which Justices the model is good/bad at, with storytelling
- [ ] Top features per class:
  - LogReg/SVM: top 30 positive coefficients (predict petitioner) and top 30 negative (predict respondent)
  - RF: top 30 feature importances
- [ ] Business interpretation paragraph — FP cost vs FN cost in legal-tech use case
- [ ] All in `JusticeCast_Final.ipynb` with prose around each cell
- **CHECKPOINT 5:** Full evaluation section complete. CAI reviews for storytelling. **Stop and wait.**

### Phase 6: ML Canvas + Notebook Polish

- [ ] Fill the Machine Learning Canvas v0.4 quadrants per the BAX 453 template
- [ ] Export `reports/ml_canvas.pdf`
- [ ] Polish `JusticeCast_Final.ipynb`: clean markdown, smooth narrative, every section maps to a rubric line item, all charts have titles + axis labels
- [ ] Polish `README.md`: project summary, reproduce-from-fresh-clone instructions, team credits
- [ ] `Restart & Run All` on a fresh kernel — must succeed end-to-end
- [ ] `pytest` green
- **CHECKPOINT 6:** Final notebook + canvas PDF ready for submission. **Stop and wait.**

### Phase 7: Pitch Deck (Part A)

- [ ] 8–12 slide deck (~10 target):
  1. Title — JusticeCast, team names, date
  2. The Problem — litigators "read the bench" via gut intuition; legal-tech firms monetize this
  3. The Insight — Justices telegraph leanings via questioning style; we measure it
  4. Market & Users — appellate litigators, amicus brief writers, legal-tech platforms
  5. Proposed Business Solution + 2–3 recommended actions
  6. ML Canvas summary
  7. Data — SCDB + Oyez, sample sizes, coverage
  8. Approach — pipeline diagram, vectorizers, classifiers, eval
  9. Results — confusion matrix, ROC AUC, per-Justice storytelling, unanimity sensitivity
  10. Recommendations — go-to-market, risks, next steps
  11. Outro / Q&A
- [ ] Storytelling: open with a vivid case, bookend with the same case
- [ ] Export `reports/JusticeCast_Pitch.pdf`
- **CHECKPOINT 7:** Both deliverables ready for Canvas submission.

## Definition of Done

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with zero errors and zero un-justified warnings
- All 9 vectorizer × classifier baseline combinations evaluated and logged with per-fit timing
- GridSearchCV applied via the sequential strategy (Stage 4A two linear-model runs sharing the vectorizer grid; Stage 4B RF with fixed vectorizer)
- Final winning model has: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
- Per-Justice performance breakdown is in the notebook with prose
- Unanimity sensitivity analysis (per-Justice metrics split by unanimous vs contested) is in the notebook
- Top n-grams for each class extracted and visualized for at least one model
- Business interpretation paragraph (FN vs FP cost) is in the notebook prose
- Machine Learning Canvas v0.4 filled in and exported as PDF
- Pitch deck 8–12 slides, exported as PDF, follows the storytelling arc
- README documents how to reproduce from a fresh clone
- pytest suite runs green
- All artifacts committed with clean history
- Proposal submitted to professor by **5/7** (Phase 0 ✅ drafted, awaiting submission)
- Both deliverables submitted to Canvas by **5/28**

## Constraints

- **Hard deadlines:** proposal 5/7 (drafted); both deliverables 5/28.
- **Oyez API:** be polite (≤ 1 req/sec across the 2-step fetch).
- **SCDB:** Justice-Centered file, release 2025_01. Latin-1 / Windows-1252 encoded — always `encoding='latin1'`.
- **Label derivation: locked in via Stop A.** `(partyWinning == 1) == (majority == 2)`, with `partyWinning == 2` and `majority NaN` excluded.
- **Framing:** Option 1 stance classification, not sentiment. Proposal makes this explicit.
- **Team size:** 6. Distribution lives in chat with CAI.

## Current Instruction

**Status:** Phase 1 Stop A complete (Heien gate passed ✅, codebook locked, fetchers + builder + tests in place). CC is approved to execute Stop B.

**Resolutions from Stop A:**

- **`majority` encoding `1=dissent, 2=majority`: confirmed.** Label derivation `(partyWinning == 1) == (majority == 2)` is correct. Phase 0's casual XNOR guess would have inverted every label.
- **Heien spot-check passed.** 8-1 outcome reproduced: Sotomayor=1, 7 majority Justices=0, Thomas dropped (silent).
- **Justice ID map: 8 of 16 verified empirically; 8 unverified.** Risk mitigated via Checkpoint 1 coverage cross-check; if any slug is wrong, fix the CSV and re-parse (no re-fetch).
- **Sequential bulk fetch: approved.** Easier to reason about and recover from than parallel; rate limit is the binding constraint either way.
- **Window extension question: deferred to Checkpoint 1** as proposed.
- **Duplicate `(term, docket)` pair: investigate at Stop B,** report which case at Checkpoint 1.

**What to produce this turn (Stop B):**

1. Bulk-fetch all 2005–2024 cases sequentially via `fetch_oyez.py` (~55–70 min, both cache layers populated)
2. Build the full joined parquet via `build_dataset.py` per the spec
3. Identify and report the duplicate `(term, docket)` pair (1,471 caseIds vs 1,470 unique pairs)
4. Run `pytest` — all tests must remain green; add new tests if Stop B reveals edge cases worth pinning down

**What to stop and report back on (Checkpoint 1):**

- Final row counts at each pipeline stage (cases attempted → fetched → with valid oral arg → with all Justices mapped → final joined)
- **Justice coverage cross-check** — for each of 16 Justices: expected case count vs actual joined-row count. Flag any Justice whose actual is suspiciously low (likely wrong slug). Show a table.
- **Thomas-specific speaking-case count for 2005–2024.** Drives Phase 2 inclusion decision for him.
- Median (and p25/p75) word counts per Justice
- % of rows flagged unanimous
- Final binary class distribution (and what percentage of rows had to be excluded due to `partyWinning == 2` or `majority NaN`)
- Multi-audio case count and concatenation behavior on a few examples
- The duplicate `(term, docket)` resolution
- Any fetch failures, parse failures, or surprises
- Total wall-clock for the bulk fetch
- **Stop and wait.**

**Pushback welcome on:**

- The 2005–2024 window — empirical Oyez coverage now visible at scale; if reliable transcripts exist further back without quality drop, propose extending in the Checkpoint 1 report
- Any Justice with a coverage gap that suggests a slug error — call it out explicitly with the suspected fix
- The `unanimous` flag derivation — if the bulk data reveals an edge case (e.g., 4-4 ties when a Justice was recused), propose a refinement
- Anything unexpected in the bulk parse that the Stop A smoke-test missed
