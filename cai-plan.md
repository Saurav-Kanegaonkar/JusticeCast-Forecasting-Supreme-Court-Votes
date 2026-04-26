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
   - **List-response failure mode:** if Oyez can't match a docket exactly, it returns a 30-entry search-fallback list. The fetcher detects this and raises `CaseNotFound`.

Joined on `(term, docket_number)`. Unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

**Justice ID mapping (SCDB ↔ Oyez):** Hand-built `data/processed/justice_id_map.csv` covers the 16 Justices appearing in 2005–2024. **All 16 slugs validated empirically at Checkpoint 1.**

## SCDB Field Semantics (verified Stop A)

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

⚠️ **Critical note:** SCDB's `majority` field is encoded `1=dissent, 2=majority` — counterintuitive. Heien spot-check at Stop A caught the inversion.

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/
│   │   ├── scdb_justice.csv                             # Latin-1
│   │   ├── scdb_codebook/                               # cached codebook HTML
│   │   └── oyez/
│   │       ├── cases/{term}_{docket}.json
│   │       └── transcripts/{audio_id}.json
│   └── processed/
│       ├── justice_id_map.csv                           # 16 rows, validated
│       ├── justice_case_rows.parquet                    # 10,308 raw joined rows
│       └── modeling_table.parquet                       # 10,039 rows (post-cleanup)
├── src/
│   ├── fetch_scdb.py
│   ├── fetch_oyez.py                                    # 2-step, hardened
│   ├── run_bulk_fetch.py
│   ├── checkpoint1_analysis.py
│   ├── rescue_failed_dockets.py
│   ├── build_dataset.py                                 # join, dedupe, derive labels
│   ├── build_modeling_table.py                          # cleanup → modeling_table.parquet
│   └── text_clean.py
├── notebooks/
│   ├── 01_eda.ipynb                                     # ⚠ EXPANSION IN PROGRESS
│   ├── 02_modeling.ipynb
│   └── JusticeCast_Final.ipynb
├── reports/
│   ├── proposal.md                                      # for the prof, due 5/7
│   ├── checkpoint1_summary.md
│   ├── ml_canvas.pdf
│   ├── JusticeCast_Pitch.pdf
│   └── results/
│       ├── bulk_fetch_log.csv
│       ├── rescue_log.csv
│       ├── modeling_table_audit.csv
│       ├── baseline_results.csv
│       └── gridsearch_results.csv
├── tests/                                               # 17 passing
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
9. **Frame as Option 1 stance classification, not sentiment.** Same machinery, different label.
10. **Hand-verify before bulk operations.** Heien spot-check caught the `majority` field inversion; rule paid for itself.
11. **EDA the input, not just the labels.** For text classification, the EDA must engage with the text — per-class vocabulary differences, sample inspection, vocab statistics, length-vs-label confound check, per-Justice signature. Label-only EDA is insufficient for understanding what the model can or cannot learn.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE

Repo: https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes

⚠️ **Reminder for Saurav:** `reports/proposal.md` needs to be submitted to the professor by **5/7**.

### Phase 1: Data Acquisition ✅ COMPLETE

10,308 joined `(case, Justice)` rows post-rescue. 1,470 cases attempted → ~1,330 with parsed transcripts. Bulk fetch took 54 minutes; cache footprint 377 MB. All 16 Justice slugs validated. List-response failure mode patched mid-run.

### Phase 2: EDA & Inclusion/Exclusion Decisions

#### 2A — Cleanup + Modeling Table ✅ COMPLETE (Checkpoint 2 substantive)

Cleanup applied: drop NaN-label rows (171), drop original-jurisdiction cases (17), drop word_count < 30 (81). Modeling table: **10,039 rows × 20 cols, 1,293 distinct cases, 16 Justices**. Class balance 62.4/37.6; unanimous 41.9%/contested 58.1%.

#### 2B — EDA Expansion ⚠ IN PROGRESS (CHECKPOINT 2 REOPENED)

The initial EDA notebook focused heavily on label distributions and Justice metadata but did not adequately engage with the text content itself — the actual model input. Reopened to add the missing analyses before any modeling work begins. Non-Negotiable #11 codifies the principle.

**The three blockers (must land before Phase 3 begins):**

- [ ] **B1. Per-class vocabulary differences.** For the modeling-table corpus split by label, compute:
  - Top 30 unigrams and top 30 bigrams ranked by log-odds-with-Dirichlet-prior (preferred over raw frequency ratio for stability) per class
  - Effect size: how strong is the discrimination? If the top-30 list is dominated by case-specific content terms (party names, statute numbers, e.g. "miranda", "section 230", "epa"), flag it explicitly — this likely means the model will partly learn topic memorization. **Decision point at Checkpoint 2-reopen:** if content terms dominate, add a custom stopword list (case names, party names, statute identifiers, common legal-topic nouns) to the Phase 3 vectorizer config. If stance markers dominate, proceed with default sklearn-stopwords behavior.
  - Visualize: a side-by-side bar chart of top n-grams per class, with content terms highlighted in a different color from stance markers
  - Document the finding plainly: "the text is/isn't textually distinguishable by class, and here's what dominates"

- [ ] **B2. Per-Justice majority-class baseline as the true yardstick.** Produce a table with one row per Justice:
  - `n_rows`, `petitioner_vote_rate`, `majority_class_baseline` (= max(rate, 1-rate)), `n_unanimous_rows`, `n_contested_rows`
  - Visualize as a horizontal bar chart sorted by baseline accuracy, with the global 62.4% line shown for reference
  - **Add a one-paragraph note in the notebook explicitly stating:** "global majority-class accuracy of 62.4% is not the right reference for this problem because per-Justice baselines range from ~50% to ~80%. Phase 5 evaluation must compare per-Justice model accuracy to that Justice's individual baseline, not the global average. A Justice with an 80% baseline whose model scores 75% is performing *worse than majority-class* on their rows even if global accuracy looks good."

- [ ] **B3. Sample text inspection.** Print 5 randomly-sampled Justice-utterance blobs (fixed `random_state=42`) covering:
  - 1 from a unanimous case
  - 1 from a contested case
  - 1 from a multi-audio case (e.g., NFIB, Obergefell)
  - 1 from Thomas (low word-count tail)
  - 1 from KBJackson (high word-count tail)
  - For each, print the first ~500 chars of the concatenated text
  - **Audit checklist:** any transcription artifacts (`[laughter]`, `[crosstalk]`, `inaudible`, leading/trailing dashes, double spaces from session boundaries)? Encoding issues? Truncated sentences? Spurious capitalization patterns?
  - **Decision point at Checkpoint 2-reopen:** if transcription artifacts are present and frequent, add a preprocessing step (regex strip) to `text_clean.py` before Phase 3.

**The three non-blockers (add if reasonable budget; otherwise document as known gaps):**

- [ ] **B4. Vocabulary statistics.** Total unique tokens in the corpus (after default sklearn tokenization), Zipf-plot of token-frequency rank, stopword density (% of tokens that are NLTK English stopwords). Informs Phase 3 `min_df` / `max_df` choices empirically rather than from the cai-plan's defaults.

- [ ] **B5. Word count vs label.** Compute mean and median word count per class, with a Mann-Whitney U test for distributional difference. If significantly different, flag as a confound: the model could be using utterance length implicitly. Decide: control for it (e.g., add a length feature explicitly so we can check coefficient sign), or accept and report.

- [ ] **B6. Per-Justice vocabulary signature.** Quick check: for each Justice, top 10 distinctive bigrams (compared to all other Justices). Are the signatures distinctive? If yes, document the implication: the model on TF-IDF + bigrams could be partly learning Justice-author-identity from text style, which combined with stable per-Justice voting patterns gives "free" predictive power that isn't actually bench-reading. This affects how we frame Phase 5 results — honest interpretation rather than overclaiming.

**Updated EDA acceptance criteria:**

- [ ] All three blockers (B1–B3) done and documented in `notebooks/01_eda.ipynb` with prose interpretation, not just charts
- [ ] At least 2 of the 3 non-blockers (B4–B6) done; any skipped ones documented as a known gap in `project-state.md`
- [ ] If B1 surfaces content-term dominance: a custom stopword list is added to `src/text_clean.py` (or a vectorizer-config helper module) ready for Phase 3 to consume
- [ ] If B3 surfaces transcription artifacts: a preprocessing regex pipeline is added to `text_clean.py`, applied to the modeling-table text column, and the modeling-table parquet is rebuilt
- [ ] EDA notebook still passes `Restart & Run All` cleanly via nbconvert

- **CHECKPOINT 2 (reopened):** Report findings from B1–B6, decisions made on stopwords / preprocessing, and any rebuilt artifacts. **Stop and wait.**

### Phase 3: Modeling Pipeline (Baseline Sweep)

- [ ] Build `Pipeline` objects for each (vectorizer × classifier) combination:
  - **Vectorizers (3):**
    - BoW: `CountVectorizer(ngram_range=(1,1))`
    - TF-IDF unigram: `TfidfVectorizer(ngram_range=(1,1))`
    - n-grams (TF-IDF bigrams): `TfidfVectorizer(ngram_range=(1,2))`
  - **All three vectorizers consume the same custom stopword config and preprocessor** decided at Checkpoint 2-reopen (if applicable). Pull from `src/text_clean.py`.
  - **Classifiers (3):**
    - `LogisticRegression(class_weight='balanced', max_iter=2000)`
    - `LinearSVC(class_weight='balanced')` — `decision_function` for AUC; `CalibratedClassifierCV` only for the calibration curve in Phase 5
    - `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
  - 9 combos total
- [ ] `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id` for the 80/20 split
- [ ] Train all 9, evaluate, log to `reports/results/baseline_results.csv` with per-fit wall-clock time
- [ ] **Compare to the per-Justice baselines from B2,** not just the global 62.4% baseline
- [ ] Identify top 3 performers by ROC AUC for tuning
- [ ] Quick eyeball pass on top features for the best linear model — if dominated by case-content terms despite the EDA-driven stopword list, surface it for Checkpoint 3 discussion
- **CHECKPOINT 3:** Baseline results table with all 9 combinations, per-fit timing, top-3 selection, per-Justice lift over baseline, top-features eyeball read. **Stop and wait** — CAI reviews top-3 selection and approves Phase 4 compute budget.

### Phase 4: GridSearchCV — Sequential Strategy

**Stage 4A: Two `GridSearchCV` runs (one per linear model) sharing the vectorizer parameter grid**

- [ ] LogReg run: `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` (with appropriate solver)
- [ ] SVM run: `C ∈ [0.01, 0.1, 1, 10]`
- [ ] Vectorizer (joint with both): `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`
  - Adjust grid bounds based on B4 vocabulary statistics if EDA suggests different defaults
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
- [ ] **Sensitivity analysis: per-Justice metrics split by unanimity.** Per-Justice accuracy and ROC AUC on (a) unanimous cases, (b) contested cases. Discuss in prose. **Frame using the per-Justice baselines from B2 — model lift over each Justice's personal baseline, not over the global 62.4%.**
- [ ] **Per-Justice performance breakdown with explicit storytelling hooks:**
  - **KBJackson** (median 1,205 words, 96% coverage) — most-engaged questioner
  - **Thomas** (295 cases post-cleanup, 20.5% speaking rate) — low-n; sensitivity treatment
  - **Sotomayor / Kagan / Roberts** as the "core" predictable bench
- [ ] Top features per class:
  - LogReg/SVM: top 30 positive coefficients (predict petitioner) and top 30 negative (predict respondent). **Compare against the B1 EDA findings** — does the model's top features align with the pre-modeling vocabulary differences? Surprises matter.
  - RF: top 30 feature importances
- [ ] **Honest interpretation pass.** If B6 surfaced strong per-Justice vocabulary signatures, explicitly discuss whether part of the model's lift comes from author-identity-from-text rather than stance-from-questions. Don't overclaim "we measured how Justices read the bench" if the model is partly recovering "this is Sotomayor and Sotomayor votes liberal."
- [ ] Business interpretation paragraph — FP cost vs FN cost in legal-tech use case
- [ ] All in `JusticeCast_Final.ipynb` with prose around each cell
- **CHECKPOINT 5:** Full evaluation section complete. CAI reviews for storytelling and honesty. **Stop and wait.**

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
  9. **Results** — confusion matrix, ROC AUC, per-Justice storytelling featuring KBJackson and Thomas, unanimity sensitivity, lift over per-Justice baselines (not global)
  10. Recommendations — go-to-market, risks, next steps
  11. Outro / Q&A
- [ ] Storytelling: open with a vivid case (Citizens United if rescue succeeded; Heien as fallback), bookend with the same case
- [ ] Export `reports/JusticeCast_Pitch.pdf`
- **CHECKPOINT 7:** Both deliverables ready for Canvas submission.

## Definition of Done

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with zero errors and zero un-justified warnings
- EDA includes per-class vocabulary differences, per-Justice baseline table, sample text inspection, vocab statistics, length-vs-label check, per-Justice signature check
- All 9 vectorizer × classifier baseline combinations evaluated and logged with per-fit timing
- GridSearchCV applied via the sequential strategy (Stage 4A two linear-model runs sharing the vectorizer grid; Stage 4B RF with fixed vectorizer)
- Final winning model has: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
- Per-Justice performance reported as **lift over each Justice's individual baseline**, not just absolute accuracy
- Unanimity sensitivity analysis (per-Justice metrics split by unanimous vs contested) is in the notebook
- Top n-grams for each class extracted; comparison between EDA-pre-modeling top features and post-modeling top features documented
- Honest interpretation pass on whether model lift comes from stance or author-identity
- Business interpretation paragraph (FN vs FP cost) in the notebook prose
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

**Status:** Phase 2A complete (cleanup + modeling table). **Phase 2B reopened** — EDA was insufficient on the text dimension and CAI approved Checkpoint 2 prematurely. CC executes the EDA expansion before Phase 3 begins.

**Why this is being reopened (transparent for the audit trail):** the initial EDA thoroughly characterized the labels and Justice metadata, but a text classifier's EDA must engage with the text itself. Without per-class vocabulary differences, we'd enter Phase 3 not knowing whether the labels are textually predictable at all. Without per-Justice baselines, "70% accuracy" in Phase 5 is uninterpretable. Without sample text inspection, we may miss preprocessing decisions that show up as Phase 3 noise. The 1–2 hours to add these is small relative to the cost of running modeling on top of incomplete EDA.

**What to produce this turn (Phase 2B EDA Expansion):**

**Blockers (B1–B3 must land):**

1. **B1 — Per-class vocabulary differences.** Top 30 unigrams + top 30 bigrams per class (`voted_petitioner=1` vs `voted_petitioner=0`), ranked by log-odds-with-Dirichlet-prior. Side-by-side bar chart, content terms vs stance markers visually distinguished. Prose interpretation in the notebook. **Decision:** if content-term dominance, build a custom stopword list in `src/text_clean.py` for Phase 3 vectorizer consumption.

2. **B2 — Per-Justice baseline table.** One row per Justice with `n_rows`, `petitioner_vote_rate`, `majority_class_baseline`, `n_unanimous_rows`, `n_contested_rows`. Sorted bar chart with global baseline shown as reference line. **Mandatory prose paragraph** in the notebook stating that per-Justice baselines (not the global 62.4%) are the correct comparison reference, with explicit framing for Phase 5.

3. **B3 — Sample text inspection.** Five fixed-seed random samples (one each from: unanimous case, contested case, multi-audio case, Thomas, KBJackson). Print first ~500 chars of each. Audit checklist for transcription artifacts. **Decision:** if artifacts present, add preprocessing regex to `text_clean.py` and rebuild `modeling_table.parquet` (the modeling-table builder should be idempotent).

**Non-blockers (B4–B6 — do at least 2 of 3, document any skipped as known gaps):**

4. **B4 — Vocabulary statistics.** Total unique tokens, Zipf plot, stopword density.

5. **B5 — Word count vs label.** Mean/median word count per class + Mann-Whitney U test. Document confound implications.

6. **B6 — Per-Justice vocabulary signature.** Top 10 distinctive bigrams per Justice (vs all others). Document author-identity-from-text implications for Phase 5 honesty.

**What to stop and report back on:**

- All B1–B3 outputs (charts + interpretation prose)
- At least 2 of B4–B6 outputs
- Any preprocessing changes made to `text_clean.py` and whether the modeling table was rebuilt
- Any custom stopword list added (and what's in it)
- Decisions deferred or skipped, with reasoning
- The `01_eda.ipynb` still passes `Restart & Run All` via nbconvert

**Pushback welcome on:**

- The log-odds-with-Dirichlet-prior choice for B1 — if you have a sklearn/scipy primitive you prefer (e.g., chi-squared, mutual information), use it; the goal is "robust ranking of class-discriminating tokens," not the specific statistic
- Any blocker that turns out to require disproportionate effort vs value — push back with reasoning, don't silently downgrade
- The "if content-term dominance, build custom stopword list" rule — if the dominance is mild (a few terms in the top 30), consider whether stopwording them is appropriate vs just noting it; full stopword list is for clear dominance, not edge cases
- Anything else where the plan's abstraction collides with code reality
