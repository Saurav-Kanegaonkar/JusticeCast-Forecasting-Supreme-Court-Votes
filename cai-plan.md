# Project Plan: JusticeCast — Forecasting Supreme Court Votes

## Objective

Build a binary text-classification system that, given the verbatim oral-argument questions a Supreme Court Justice asks during a single case, predicts whether that Justice will vote with the petitioner or the respondent. Final deliverables are a polished, reproducible Jupyter notebook (Part B, 20 pts) and a pitch deck (Part A, 15 pts) framing the work as a legal-tech product. Total: 35 pts.

## Tech Stack

- Python 3.14.3, Jupyter Notebook (primary deliverable)
- `pandas==3.0.2`, `numpy==2.4.4` — data wrangling
- `scikit-learn==1.8.0` — `CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedGroupKFold`, `CalibratedClassifierCV`, all metrics
- `requests==2.33.1`, `beautifulsoup4==4.14.3`, `tenacity==9.1.4` — Oyez data fetching
- `matplotlib==3.10.9`, `seaborn==0.13.2` — visualization
- `nltk==3.9.4` — stopwords, stemming/lemmatization
- `joblib==1.5.3`, `pyarrow==24.0.0` — pipeline / fetched-data caching
- `pytest==9.0.3` — tests for fetchers and builders
- All deps pinned in `requirements.txt` + tracked unpinned in `requirements.in`

## Data Sources

1. **Supreme Court Database (SCDB)** — Washington University, scdb.wustl.edu (HTTP only). Justice-Centered file. Free CSV. Latest release: `2025_01`. Direct URL:
   ```
   http://scdb.wustl.edu/_brickFiles/2025_01/SCDB_2025_01_justiceCentered_Citation.csv.zip
   ```
   83,644 vote rows × 61 columns. **Encoding: Latin-1 / Windows-1252** — read with `pd.read_csv(path, encoding='latin1')`.

2. **Oyez.org API — TWO-STEP FETCH** (verified empirically against *Heien v. North Carolina*, term 2014, docket 13-604):
   - **Step 1 — Case metadata:** `GET https://api.oyez.org/cases/{term}/{docket}` returns case-level JSON including `oral_argument_audio[]` array.
   - **Step 2 — Transcript:** for each entry in `oral_argument_audio[]`, follow the `href` to `https://api.oyez.org/case_media/oral_argument_audio/{audio_id}`.
   - **Multi-audio cases:** iterate over ALL audio entries per case and concatenate that Justice's utterances. Store the count as `n_audio_sessions` metadata.
   - **Cases without oral argument** (`oral_argument_audio == []`) are filtered out.
   - **List-response failure mode:** if Oyez can't match a docket exactly, it returns a 30-entry search-fallback list instead of a case dict. The fetcher detects this and raises `CaseNotFound` rather than caching corrupt data — discovered and patched mid-Phase-1.

Joined on `(term, docket_number)`. Unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

**Justice ID mapping (SCDB ↔ Oyez):** Hand-built `data/processed/justice_id_map.csv` covers the 16 Justices appearing in 2005–2024. **All 16 slugs validated empirically at Checkpoint 1.**

## SCDB Field Semantics (verified Stop A)

Locked in from the SCDB codebook (cached at `data/raw/scdb_codebook/`):

| Field | Semantics |
|---|---|
| `partyWinning` | `0`=petitioner LOST, `1`=petitioner WON, `2`=unclear (EXCLUDE from labels) |
| `majority` | `1`=dissent, `2`=majority, `NaN`=did not participate (EXCLUDE) |
| `vote` | 1..8 categorical (concurrence types). Not used directly. |
| `direction` | `1`=conservative, `2`=liberal. Not used in our binary label. |
| `caseDisposition` | 11-value taxonomy that already governs `partyWinning`. |

**Final label derivation (locked in `src/build_dataset.py::derive_voted_petitioner`):**

```
voted_petitioner = (partyWinning == 1) == (majority == 2)
```

Returns `None` if either field is missing or `partyWinning == 2`.

⚠️ **Critical note:** SCDB's `majority` field is encoded `1=dissent, 2=majority` — counterintuitive. The Heien spot-check (Stop A) caught this; without it, every label would have been silently inverted.

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/
│   │   ├── scdb_justice.csv                             # Latin-1
│   │   ├── scdb_codebook/                               # cached codebook HTML
│   │   └── oyez/
│   │       ├── cases/{term}_{docket}.json               # Step 1 cache
│   │       └── transcripts/{audio_id}.json              # Step 2 cache
│   └── processed/
│       ├── justice_id_map.csv                           # 16 rows, all validated
│       ├── justice_case_rows.parquet                    # 10,272 raw joined rows
│       └── modeling_table.parquet                       # post-EDA filtered
├── src/
│   ├── fetch_scdb.py
│   ├── fetch_oyez.py                                    # 2-step, hardened against list-fallback
│   ├── run_bulk_fetch.py                                # per-case error guard, progress logs
│   ├── checkpoint1_analysis.py                          # generates summary report
│   ├── build_dataset.py                                 # join, dedupe, derive labels
│   └── text_clean.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── JusticeCast_Final.ipynb                          # SUBMISSION notebook
├── reports/
│   ├── proposal.md                                      # for the prof, due 5/7
│   ├── checkpoint1_summary.md
│   ├── ml_canvas.pdf
│   ├── JusticeCast_Pitch.pdf
│   └── results/
│       ├── bulk_fetch_log.csv
│       ├── baseline_results.csv
│       └── gridsearch_results.csv
├── tests/                                               # 11 passing
├── requirements.in / requirements.txt
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
6. **Class imbalance handled explicitly.** Final balance is 62.4% petitioner / 37.6% respondent. Use `class_weight='balanced'`. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged.** A `reports/results/` table per sweep — accuracy, precision, recall, F1, ROC AUC, **per-fit wall-clock time**.
8. **Cache aggressively.** Oyez calls cached at both layers; SCDB downloaded once. Reruns are fast.
9. **Frame as Option 1 stance classification, not sentiment.** Same machinery, different label. Proposal makes this explicit.
10. **Hand-verify before bulk operations.** Heien spot-check at Stop A caught the `majority` field inversion that would have flipped every label — rule paid for itself on first use.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE

Repo: https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes

⚠️ **Reminder for Saurav:** `reports/proposal.md` is drafted but still needs to be submitted to the professor by **5/7**.

### Phase 1: Data Acquisition ✅ COMPLETE (Checkpoint 1)

#### Stop A — Codebook + Smoke Test ✅
Heien gate passed. Codebook semantics locked. `majority` field encoding confirmed `1=dissent, 2=majority` (opposite of the casual Phase 0 guess).

#### Stop B — Bulk Fetch + Checkpoint 1 ✅

**Pipeline result:**
- 1,470 unique `(term, docket)` pairs attempted → 1,420 Step 1 fetched → 1,322 with valid oral argument → 1,307 cases with parsed transcripts
- **10,272 joined `(case, Justice)` rows; 10,121 with valid binary labels** (151 excluded: 24 unclear winner + 127 non-participating Justices)
- **Class balance: 62.4% / 37.6%** (petitioner / respondent)
- **Unanimous rows: 4,256 (42.1%)** — strong statistical power for unanimity-split sensitivity analysis
- **Bulk-fetch wall-clock: 54 minutes** (2.2 sec/case incl. multi-step + retries)
- **Cache footprint: 377 MB** (smaller than 750 MB Stop A estimate)

**Justice coverage cross-check:** all 16 slugs validated. Coverage ratios (actual / expected) range from 79–96%, with one outlier:

- **Thomas: 20.5%** (302 of 1,471 SCDB cases). His silence is a real behavioral signal, not a slug bug. **KEEP him with low-n caveat in Phase 5 per-Justice analysis.**
- **KBJackson: 96.1% coverage, median word count 1,204** — 8× O'Connor's, 5× Thomas's. Most-engaged questioner in the modern court. **Storytelling hook for Phases 5 and 7.**

**Mid-run engineering saves:**
- **List-response failure mode:** Oyez returns a 30-entry search-fallback list when a docket can't be matched. `fetch_oyez.py` was hardened to detect this, raise `CaseNotFound`, and never write corrupt cache. 50 polluted cache files were cleaned post-mortem.
- **Per-case error guard in `run_bulk_fetch.py`** — single bad case can't kill the run.

**Multi-audio cases handled:** 13 in window, including NFIB v. Sebelius (4 sessions, the ACA case), Obergefell, Riley, Miller v. Alabama. Concatenation verified.

**Medellin duplicate resolved:** docket 06-984 has both a merits decision and a per-curiam stay. Dedupe on `(term, docket, justice)` keeping earliest `dateDecision` keeps the merits vote (which is what the oral-argument transcript pairs with).

**Fetch failures (50 of 1,470 = 3.4%):**
- 16 original-jurisdiction cases — Oyez doesn't catalog these. State-vs-state disputes, no formal cert grant. Drop is defensible (different case type entirely).
- 3 application/stay dockets — drop.
- 3 unicode-mojibake dockets — drop.
- 22 standard-format failures — **rescue at Stop C** before Phase 2. Includes Citizens United (2009/08-205) which would matter for pitch-deck storytelling.

#### Stop C — Targeted Rescue Pass (~30 min)

**Rationale:** 22 standard-format failures are 1.5% of mass. They won't move the model needle. But they include landmark cases like Citizens United — story value > statistical value. The "Boil the Ocean" standard says do the rescue.

**Approach:**

- [ ] Build `src/rescue_failed_dockets.py` that, for each failed standard-format case in `bulk_fetch_log.csv`:
  - Try `OT{term}/{docket}` format
  - Try `{term-1}/{docket}` (Oyez sometimes files under the previous term)
  - Try fuzzy match on case name via Oyez search if both fail
- [ ] Document the matching strategy attempted per case in `reports/rescue_log.csv`
- [ ] **Verify Citizens United parses end-to-end** (transcript loaded, Justices identified, labels derived correctly) before claiming the rescue worked
- [ ] Re-run `build_dataset.py` to fold rescued cases into `justice_case_rows.parquet`
- [ ] Update `reports/checkpoint1_summary.md` with new totals
- **STOP C REPORT** — Report:
  - How many of the 22 were rescued, by which strategy
  - Citizens United verification (paste the 9-row table for the case)
  - New row counts in the parquet
  - Any cases that genuinely don't exist on Oyez (legitimate drops)

### Phase 2: EDA & Inclusion/Exclusion Decisions

**Cleanup tasks (do these first, before EDA notebook):**

- [ ] In `build_dataset.py`'s modeling-table step, **drop the 151 NaN-label rows** (24 from `partyWinning == 2`, 127 from `majority NaN`). The 45 unmatched `(case, Justice)` rows referenced in CC's Checkpoint 1 are a subset of these — drop them all together at the modeling-table boundary.
- [ ] **Drop original-jurisdiction cases** from the modeling table. They're substantively different (no cert grant, state-vs-state) and Oyez doesn't catalog them anyway.
- [ ] Decide on a low-word-count threshold based on EDA (e.g., drop rows where Justice spoke fewer than 50 words).

**EDA notebook (`notebooks/01_eda.ipynb`):**

- [ ] Vote-label distribution overall and per-Justice
- [ ] Word-count distributions per Justice (Thomas in low tail, KBJackson in high tail)
- [ ] Cases per term; petitioner-win rate per term
- [ ] Correlation between Justice talkativeness and predictability (preview)
- [ ] **Label balance for unanimous vs contested cases** (Phase 5 sensitivity hooks here)
- [ ] Storytelling-grade visuals: word-count distribution by Justice with Thomas and KBJackson as named outliers; per-term petitioner-win-rate over time
- [ ] All charts have titles + axis labels; no default matplotlib styling

**Decisions documented in `project-state.md`:**

- **Thomas:** KEEP with low-n caveat. 302 cases is enough for stable per-Justice estimates.
- **Unanimous cases:** KEEP. Flagged via `unanimous` metadata, split-analyzed in Phase 5.
- **Original-jurisdiction:** DROP. Substantively different case type.
- **Low-word-count threshold:** TBD from EDA distributions.

**Build the final modeling table** → `data/processed/modeling_table.parquet`

- **CHECKPOINT 2:** Report final dataset shape after cleanup, inclusion criteria applied, label balance overall and split by unanimity, expected baselines (majority-class accuracy as a floor). **Stop and wait.**

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
- **CHECKPOINT 3:** Baseline results table with all 9 combinations, per-fit timing, identify top 3. **Stop and wait** — CAI reviews top-3 selection and approves Phase 4 compute budget.

### Phase 4: GridSearchCV — Sequential Strategy

**Stage 4A: Two `GridSearchCV` runs (one per linear model) sharing the vectorizer parameter grid**

- [ ] LogReg run: `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` (with appropriate solver)
- [ ] SVM run: `C ∈ [0.01, 0.1, 1, 10]`
- [ ] Vectorizer (joint with both): `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`
- [ ] `StratifiedGroupKFold(n_splits=5)` with `groups=case_id`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] Record best vectorizer config from whichever linear model wins

**Stage 4B: RF with fixed vectorizer**

- [ ] Vectorizer fixed at Stage 4A's winning config
- [ ] `n_estimators ∈ [100, 300, 500]`, `max_depth ∈ [None, 20, 50]`, `min_samples_split ∈ [2, 5, 10]`

**Both stages:**

- [ ] Log to `reports/results/gridsearch_results.csv` with per-fit timing
- [ ] Refit best per model on full train, evaluate on held-out test
- [ ] Identify final winning model overall

- **CHECKPOINT 4:** Best hyperparameters per model, CV vs test gap, final winning model, total Phase 4 compute time. **Stop and wait.**

### Phase 5: Evaluation & Interpretability

- [ ] For winning model: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
  - If winner is `LinearSVC`: wrap with `CalibratedClassifierCV(method='sigmoid', cv=5)` **only for the calibration curve**
- [ ] **Sensitivity analysis: per-Justice metrics split by unanimity.** For each Justice, report accuracy and ROC AUC separately on (a) unanimous cases, (b) contested cases. Discuss in prose.
- [ ] **Per-Justice performance breakdown with explicit storytelling hooks:**
  - **KBJackson** (median 1,204 words, 96% coverage) — most-engaged questioner; does her chattiness translate to predictability?
  - **Thomas** (302 cases, 20.5% speaking rate) — low-n; is the model meaningful for him? Treat as a sensitivity case, not a primary claim.
  - **Sotomayor / Kagan / Roberts** as the "core" predictable bench
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
- **CHECKPOINT 6:** Final notebook + canvas PDF ready. **Stop and wait.**

### Phase 7: Pitch Deck (Part A)

- [ ] 8–12 slide deck (~10 target):
  1. Title — JusticeCast, team names, date
  2. The Problem — litigators "read the bench" via gut intuition
  3. The Insight — Justices telegraph leanings via questioning style; we measure it
  4. Market & Users — appellate litigators, amicus brief writers, legal-tech platforms
  5. Proposed Business Solution + 2–3 recommended actions
  6. ML Canvas summary
  7. Data — SCDB + Oyez, sample sizes, coverage
  8. Approach — pipeline diagram, vectorizers, classifiers, eval
  9. **Results** — confusion matrix, ROC AUC, per-Justice storytelling featuring KBJackson (the engaged questioner) and Thomas (the silent outlier), unanimity sensitivity
  10. Recommendations — go-to-market, risks, next steps
  11. Outro / Q&A
- [ ] Storytelling: open with a vivid case (Citizens United if rescue succeeded; Heien as fallback), bookend with the same case
- [ ] Export `reports/JusticeCast_Pitch.pdf`
- **CHECKPOINT 7:** Both deliverables ready for Canvas submission.

## Definition of Done

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with zero errors and zero un-justified warnings
- All 9 vectorizer × classifier baseline combinations evaluated and logged with per-fit timing
- GridSearchCV applied via the sequential strategy (Stage 4A two linear-model runs sharing the vectorizer grid; Stage 4B RF with fixed vectorizer)
- Final winning model has: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
- Per-Justice performance breakdown is in the notebook with KBJackson and Thomas explicitly discussed
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

- **Hard deadlines:** proposal 5/7 (drafted, ⚠️ awaiting submission); both deliverables 5/28
- **Oyez API:** ≤ 1 req/sec across the 2-step fetch
- **SCDB:** Justice-Centered file, release 2025_01. Latin-1 / Windows-1252 encoded
- **Label derivation: locked.** `(partyWinning == 1) == (majority == 2)`, with `partyWinning == 2` and `majority NaN` excluded
- **Framing:** Option 1 stance classification, not sentiment
- **Team size:** 6. Distribution lives in chat with CAI

## Current Instruction

**Status:** Phase 1 ✅ complete (Checkpoint 1 cleared). CC is approved to execute **Stop C (rescue pass)** then proceed to **Phase 2** without an intermediate CAI sign-off — Stop C is a tightly scoped 30-min cleanup, and Phase 2's first task is rebuilding the modeling table after rescue lands.

**Resolutions from Checkpoint 1:**

- **(a) Drop 45 NaN-label rows at modeling-table build: approved.** Folded into Phase 2 cleanup tasks.
- **(b) Keep Thomas with low-n caveat: approved.** Phase 5 explicitly treats him as a sensitivity case.
- **(c) No window extension: approved.** 10,121 labeled rows is plenty.
- **(d) Rescue the 22 standard-format failures: approved (Stop C).** Story value > statistical value — Citizens United is named-and-shamed in the failures and we want it for the pitch deck. ~30 min budget.
- **KBJackson chattiness: added as explicit storytelling hook** in Phases 5 and 7.

**What to produce this turn (Stop C + Phase 2):**

**Stop C (rescue):**
1. Implement `src/rescue_failed_dockets.py` with the three matching strategies (OT-prefix, term-1 shift, fuzzy case-name search)
2. Run rescue on the 22 standard-format failures from `bulk_fetch_log.csv`
3. Document per-case rescue outcome in `reports/rescue_log.csv`
4. **Verify Citizens United (2009/08-205) parses end-to-end** before claiming the rescue worked — paste the per-Justice table
5. Re-run `build_dataset.py` to fold rescued cases into `justice_case_rows.parquet`
6. Update `reports/checkpoint1_summary.md` with new totals
7. Quick stop signal here for CAI to confirm the rescue worked before EDA — keep it short (just the rescue counts + Citizens United table)

**Phase 2 (after rescue stop):**
1. Apply Phase 2 cleanup tasks: drop NaN-label rows, drop original-jurisdiction cases, set low-word-count threshold from EDA
2. Produce `notebooks/01_eda.ipynb` per the spec above
3. Build `data/processed/modeling_table.parquet`
4. Document Phase 2 decisions in `project-state.md`
5. Stop at Checkpoint 2

**Pushback welcome on:**

- **Original-jurisdiction drop** — if EDA reveals these cases have transcripts after all (unlikely but possible), reconsider
- **Low-word-count threshold** — propose the empirical cutoff with a histogram, don't pick arbitrarily
- **Any rescued case that fails the Citizens United-style end-to-end verification** — drop it from the rescue and document why, rather than claim a partial parse
- The KBJackson / Thomas storytelling angles — if EDA reveals a more compelling angle (e.g., a Justice whose talkativeness inversely correlates with predictability), propose substituting it
