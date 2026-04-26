# Project Plan: JusticeCast — Forecasting Supreme Court Votes

## Objective

Build a binary text-classification system that, given the verbatim oral-argument questions a Supreme Court Justice asks during a single case, predicts whether that Justice will vote with the petitioner or the respondent. Final deliverables are a polished, reproducible Jupyter notebook (Part B, 20 pts) and a pitch deck (Part A, 15 pts) framing the work as a legal-tech product. Total: 35 pts.

## Tech Stack

- Python 3.11+, Jupyter Notebook (primary deliverable)
- `pandas`, `numpy` — data wrangling
- `scikit-learn` ≥ 1.3 — `CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedGroupKFold`, `CalibratedClassifierCV`, all metrics
- `requests`, `beautifulsoup4` — Oyez data fetching (with a polite rate limiter)
- `matplotlib`, `seaborn` — visualization
- `nltk` (or sklearn-native) — stopwords, stemming/lemmatization
- `joblib` — pipeline / fetched-data caching
- `pytest` — tests for fetchers and builders
- All deps pinned in `requirements.txt`

## Data Sources

1. **Supreme Court Database (SCDB)** — Washington University, scdb.wustl.edu. Justice-Centered file (one row per Justice per case). Free CSV. Contains every recorded SCOTUS vote with petitioner/respondent winner, majority/dissent flags, ideological direction, etc.
2. **Oyez.org API** — `https://api.oyez.org/cases/{term}/{docket}`. Returns full oral-argument transcripts with each utterance tagged by speaker (Justice name + identifier). Free, public, no auth required.

Joined on `(term, docket_number)`. Final unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/                      # SCDB CSV + Oyez JSONs (cached, gitignored)
│   └── processed/                # joined parquet tables
├── src/
│   ├── fetch_scdb.py
│   ├── fetch_oyez.py             # rate-limited, retried, cached
│   ├── build_dataset.py          # join + aggregate per (case, Justice)
│   └── text_clean.py             # tokenization, stopwords, stemming
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── JusticeCast_Final.ipynb   # SUBMISSION notebook, top-to-bottom narrative
├── reports/
│   ├── proposal.md               # for the prof, due 5/7
│   ├── ml_canvas.pdf
│   ├── JusticeCast_Pitch.pdf
│   └── results/                  # CSVs of every experiment run
├── tests/
│   ├── test_fetchers.py
│   └── test_builders.py
├── requirements.txt
├── README.md
├── CLAUDE.md
└── project-state.md
```

## Non-Negotiables

These rules apply across every phase. Violations are bugs, not preferences.

1. **No data leakage. Split by `case_id` using `StratifiedGroupKFold`.** All Justices for a given case go into the same split. `train_test_split(stratify=y)` does NOT respect groups and would leak cases across train/test — **do not use it for the primary split**. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`: take fold 0 as the held-out test set (~20%), use folds 1–4 as train. For nested CV inside `GridSearchCV`, also use `StratifiedGroupKFold` and pass `groups=case_id` to `.fit()`.
2. **Stratified splits.** Stratify on the binary vote label so train and test have matching class balance. `random_state=42` everywhere.
3. **Vectorizers fit on train only.** No vocabulary built from test data, ever. Use `sklearn.pipeline.Pipeline` so this is enforced by construction, not by discipline.
4. **No post-hoc features.** Only use information available at the moment the Justice finished speaking — no decision text, no opinion text, no outcome-derived features in `X`. The vote label is the *only* future signal we touch.
5. **Reproducibility.** Fixed seed (42), pinned dependencies, notebook runs top-to-bottom on a fresh kernel via `Restart & Run All` with zero errors.
6. **Class imbalance handled explicitly.** Petitioner-side wins ~65–70% of SCOTUS cases. Use `class_weight='balanced'` or document the choice. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged.** A `reports/results/` table with one row per (vectorizer, classifier, hyperparams) combination — accuracy, precision, recall, F1, ROC AUC, **and per-fit wall-clock time**. The notebook reads from these tables; we don't re-run sweeps to re-knit the notebook.
8. **Cache aggressively.** Oyez calls cached to disk; SCDB downloaded once. Reruns are fast.
9. **Frame as Option 1 stance classification, not sentiment.** The rubric mentions "sentiment analysis" because Option 2 is sentiment. We are Option 1 (custom topic). Our task is **stance classification** — same machinery, different label. The proposal makes this explicit so the professor signs off on the framing.

## Implementation Phases

### Phase 0: Proposal & Repo Init (deadline 5/7)

- [ ] Initialize git repo with proper `.gitignore` (data/raw, .env, .ipynb_checkpoints excluded)
- [ ] Create Python venv, install deps, pin versions, write `requirements.txt`
- [ ] Scaffold the directory structure under Architecture above
- [ ] Write initial `CLAUDE.md` (CC's own working notes) and `project-state.md` (project ground truth, updated on every checkpoint)
- [ ] Verify SCDB download URL is live and pull a sample row to confirm schema
- [ ] Verify Oyez API responds for a sample case (e.g., `cases/2014/13-1314`, *Heien v. North Carolina*)
- [ ] Draft `reports/proposal.md` — a 1-page proposal for the professor covering: project name (JusticeCast), business problem, proposed solution, why this maps to BoW + TF-IDF + n-grams + 3 classifiers + GridSearchCV, and the deadline structure
- **CHECKPOINT 0:** Report back with: repo layout, dependency list, sample SCDB row (column names + dtypes), sample Oyez transcript JSON snippet showing the speaker-tagging structure, and draft proposal text. **Stop and wait for approval before bulk-fetching data.**

### Phase 1: Data Acquisition

- [ ] Download full SCDB Justice-Centered file → `data/raw/scdb_justice.csv`
- [ ] Build Oyez fetcher with: rate limiting (≤1 req/sec, polite default), retries with exponential backoff, on-disk JSON cache, structured logging
- [ ] Bulk-fetch transcripts for cases in the 2005–2024 SCOTUS terms (~1,500–2,000 cases). Reasoning: Oyez coverage is solid from ~2000 onward; restricting to 2005–2024 gives consistent quality. Adjust if EDA shows broader coverage is reliable.
- [ ] Parse each transcript: extract every utterance, tag each turn with speaker name + role (Justice / advocate / unknown). Filter to Justice utterances only.
- [ ] Aggregate per (case, Justice): concatenate all of that Justice's utterances in that case into one text blob. Store original turn count and word count as metadata columns.
- [ ] **Annotate each row with `unanimous` flag** derived from SCDB (1 if all participating Justices voted the same direction, 0 otherwise). This is metadata, not a model feature.
- [ ] Write parquet output: `data/processed/justice_case_rows.parquet`
- [ ] Write 3+ pytest tests: SCDB row-count sanity check, Oyez fetcher returns expected JSON shape, parser correctly attributes speakers
- **CHECKPOINT 1:** Report row counts, Justice coverage (which Justices, how many cases each), median word counts per Justice per case, % of cases successfully fetched, % of rows flagged unanimous, any failures or anomalies. **Stop and wait.**

### Phase 2: EDA & Inclusion/Exclusion Decisions

- [ ] Produce `notebooks/01_eda.ipynb` with: vote-label distribution overall and per-Justice; word-count distributions per Justice (Thomas will be a tail); cases per term; petitioner-win rate per term; correlation between Justice talkativeness and predictability; **breakdown of label balance for unanimous vs contested cases**
- [ ] Decide inclusion/exclusion criteria (CC proposes, CAI reviews):
  - Drop rows with very low Justice word count? Empirical threshold from EDA.
  - **Unanimous cases: KEEP.** Flag via the `unanimous` metadata column added in Phase 1. The model doesn't see unanimity as a feature; it must learn vote behavior from text alone. We will report metrics split by unanimity as a sensitivity analysis in Phase 5.
  - Drop Justices with too few cases (recess appointments, partial-term Justices)?
- [ ] Build the **final modeling table** based on decisions → `data/processed/modeling_table.parquet`
- [ ] Document decisions in `project-state.md`
- **CHECKPOINT 2:** Report final dataset shape, inclusion criteria applied, label balance overall and split by unanimity, and expected baselines (majority-class accuracy as a floor). **Stop and wait.**

### Phase 3: Modeling Pipeline (Baseline Sweep)

- [ ] Build `Pipeline` objects for each (vectorizer × classifier) combination:
  - **Vectorizers (3):**
    - BoW: `CountVectorizer(ngram_range=(1,1))`
    - TF-IDF unigram: `TfidfVectorizer(ngram_range=(1,1))`
    - n-grams (TF-IDF bigrams): `TfidfVectorizer(ngram_range=(1,2))`
  - **Classifiers (3):**
    - `LogisticRegression(class_weight='balanced', max_iter=2000)`
    - `LinearSVC(class_weight='balanced')` — use `decision_function` output for ROC AUC (sklearn `roc_auc_score` accepts decision-function scores; AUC is rank-based). Wrap with `CalibratedClassifierCV` only when probabilities are needed (e.g., calibration curves in Phase 5).
    - `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
  - Total: 9 baseline combos
- [ ] Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id` to define a single train/test split (fold 0 = test, folds 1–4 = train)
- [ ] Train all 9, evaluate on test, log to `reports/results/baseline_results.csv` **with per-fit wall-clock time** (used to budget Phase 4)
- [ ] Identify top 3 performers by ROC AUC for tuning
- **CHECKPOINT 3:** Report baseline results table with all 9 combinations, per-fit timing, and identify top 3. **Stop and wait** — CAI reviews top-3 selection and approves the Phase 4 compute budget based on observed timings.

### Phase 4: GridSearchCV — Sequential Strategy

We do **not** run a joint grid over (vectorizer × model hyperparams) for all 3 classifiers — that's a 1,620+ fit blowup for RF. Instead, we tune in two stages.

**Stage 4A: Tune linear models jointly (vectorizer + model hyperparams)**

- [ ] For LogReg and SVM (both fast on sparse text features), run a single joint `GridSearchCV`:
  - **LogReg grid:** `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` with appropriate solver (e.g., `liblinear`)
  - **SVM grid:** `C ∈ [0.01, 0.1, 1, 10]`
  - **Vectorizer (joint):** `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`
- [ ] Use `StratifiedGroupKFold(n_splits=5)` with `groups=case_id`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] Record the **best vectorizer config** from Stage 4A — this becomes the fixed vectorizer for RF in Stage 4B

**Stage 4B: Tune RF with fixed vectorizer**

- [ ] Fix the vectorizer at the best config from Stage 4A
- [ ] Run `GridSearchCV` over RF hyperparams only:
  - `n_estimators ∈ [100, 300, 500]`
  - `max_depth ∈ [None, 20, 50]`
  - `min_samples_split ∈ [2, 5, 10]`
- [ ] Same CV scheme, same scoring, same group constraint

**Both stages:**

- [ ] Log all results to `reports/results/gridsearch_results.csv` (with per-fit timing)
- [ ] Refit best estimator per model on full train, evaluate on held-out test
- [ ] Report whether the best linear model or the best RF wins overall — that's the production model for Phase 5

- **CHECKPOINT 4:** Report best hyperparameters per model, CV vs test performance gap (overfitting check), final winning model, and total Phase 4 compute time. **Stop and wait.**

### Phase 5: Evaluation & Interpretability

- [ ] For winning model: confusion matrix, precision, recall, F1, ROC AUC, ROC curve plot, PR curve plot, calibration curve
  - If winning model is `LinearSVC`, wrap with `CalibratedClassifierCV(method='sigmoid', cv=5)` **only for the calibration curve** — not for AUC, not for the confusion matrix (those use the raw classifier's `decision_function` and `predict`)
- [ ] **Sensitivity analysis: per-Justice metrics split by unanimity.** For each Justice, report accuracy and ROC AUC separately on (a) unanimous cases, (b) contested cases. Discuss in prose: if performance on contested cases meaningfully exceeds majority-class baseline, the model is doing real work; if it only excels on unanimous cases, that's a different (weaker) story.
- [ ] Per-Justice performance breakdown (overall, not split): which Justices is the model good/bad at? Reportable as a story (e.g., "Sotomayor is highly predictable; Kagan is not")
- [ ] Top features per class:
  - For LogReg/SVM: coefficients sorted, top 30 positive (predict petitioner) and top 30 negative (predict respondent)
  - For RF: feature importances, top 30
- [ ] Business interpretation paragraph — FP cost vs FN cost in the legal-tech use case:
  - **FP** (predicted petitioner-vote, actual respondent-vote): law firm over-prepares petitioner-friendly arguments → wasted effort, possibly mistargeted amicus brief
  - **FN** (predicted respondent-vote, actual petitioner-vote): missed signal → firm under-prepares for a sympathetic Justice → lost opportunity
- [ ] Writeup goes into `JusticeCast_Final.ipynb` with prose around each cell
- **CHECKPOINT 5:** Full evaluation section is complete. Ask CAI to review for storytelling. **Stop and wait.**

### Phase 6: ML Canvas + Notebook Polish

- [ ] Fill the Machine Learning Canvas v0.4 quadrants:
  - **Goal:** What/why/who — predict Justice vote alignment from oral-argument questions
  - **Predict (lower-left):** Decisions (litigation prep, amicus targeting), ML Task (binary stance classification), Offline Evaluation (ROC AUC, FN/FP cost, unanimity-split sensitivity), Making Predictions (per-Justice per-case)
  - **Learn (upper-right):** Value Propositions, Data Sources (SCDB + Oyez), Collecting Data (term-by-term refresh), Features (BoW / TF-IDF / n-grams), Building Models (re-train annually after term ends)
  - **Evaluate (bottom):** Live Evaluation and Monitoring (track per-term performance drift)
- [ ] Export ML Canvas as PDF → `reports/ml_canvas.pdf`
- [ ] Polish `JusticeCast_Final.ipynb`: clean markdown headers, smooth narrative, every section maps to a rubric line item, all charts have titles + axis labels, all printouts are formatted
- [ ] Write `README.md` with project summary, how to reproduce, team credits
- [ ] Run `Restart & Run All` on a fresh kernel — must succeed end-to-end
- [ ] Run `pytest` — all tests green
- **CHECKPOINT 6:** Final notebook + canvas PDF ready for submission. **Stop and wait** for CAI review before pitch deck.

### Phase 7: Pitch Deck (Part A)

- [ ] 8–12 slide deck (~10 target):
  1. **Title** — JusticeCast, team names, date
  2. **The Problem** — litigators "read the bench" via gut intuition; legal-tech firms monetize this; current tools are surface-level
  3. **The Insight** — Justices telegraph their leanings through how they question; we measure it
  4. **Market & Users** — appellate litigators, amicus brief writers, legal-tech platforms (Lex Machina, Bloomberg Law, Westlaw Edge, SCOTUSblog)
  5. **Proposed Business Solution** — JusticeCast as an API/dashboard product. 2–3 recommended business actions: (a) pre-argument prep tool, (b) post-argument forecast for amicus and press, (c) historical bench-reading benchmarks
  6. **ML Canvas summary**
  7. **Data** — SCDB + Oyez, sample sizes, coverage map
  8. **Approach** — pipeline diagram, vectorizers, classifiers, eval
  9. **Results** — confusion matrix, ROC AUC, per-Justice storytelling, unanimity sensitivity
  10. **Recommendations** — go-to-market for legal-tech firms; risks; next steps (retrain per term, add additional feature ideas)
  11. **Outro / Q&A**
- [ ] Apply storytelling: open with a vivid case, bookend with the same case in the closing
- [ ] Export to PDF → `reports/JusticeCast_Pitch.pdf`
- **CHECKPOINT 7:** Both deliverables ready for Canvas submission.

## Definition of Done

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with zero errors and zero un-justified warnings
- All 9 vectorizer × classifier baseline combinations evaluated and logged with per-fit timing
- GridSearchCV applied via the sequential strategy (Stage 4A linear models joint; Stage 4B RF with fixed vectorizer)
- Final winning model has: confusion matrix (rendered figure), precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
- Per-Justice performance breakdown is in the notebook with prose discussion
- Unanimity sensitivity analysis (per-Justice metrics split by unanimous vs contested) is in the notebook
- Top n-grams for each class extracted and visualized for at least one model
- Business interpretation paragraph (FN vs FP cost) is in the notebook prose
- Machine Learning Canvas v0.4 filled in and exported as PDF
- Pitch deck is 8–12 slides, exported as PDF, follows the storytelling arc
- README documents how to reproduce from a fresh clone
- pytest suite runs green
- All artifacts committed to repo with clean history
- Proposal submitted to professor by **5/7** (Phase 0)
- Both deliverables submitted to Canvas by **5/28**

## Constraints

- **Hard deadlines:** proposal 5/7; both deliverables 5/28. Do not slip.
- **Oyez API:** be polite (≤1 req/sec). Coverage is best for 2005+ terms.
- **SCDB:** Justice-Centered file, latest release. The actual binary label is derived from `partyWinning` and the Justice's vote direction; CC verifies field semantics in Phase 1.
- **Framing:** professor's rubric mentions "Sentiment Analysis" because Option 2 is sentiment. Our framing is **stance classification** under Option 1. The proposal makes this explicit and the professor signs off; if professor pushes back, fall back to "stance toward petitioner = positive sentiment toward petitioner's argument."
- **Team size:** 6. Distribution lives in chat with CAI, not in this file.

## Current Instruction

**Status:** Pre-Phase-0. Four pushback items from CC's pre-execution review have been resolved. CC is approved to execute Phase 0.

**Resolutions (incorporated into the plan above):**

1. **`StratifiedGroupKFold` is the splitting primitive** (Non-Negotiable #1). `train_test_split(stratify=y)` is forbidden for the primary split because it does not respect groups. Use fold 0 as test, folds 1–4 as train.
2. **`LinearSVC` + AUC**: use `decision_function` scores directly with `roc_auc_score`. Wrap with `CalibratedClassifierCV` only for the calibration curve in Phase 5 (Phase 3 and Phase 5 updated).
3. **Unanimous cases: KEEP.** Flagged as metadata (`unanimous` column added in Phase 1). Phase 5 adds a per-Justice sensitivity analysis split by unanimity (Phases 1, 2, and 5 updated).
4. **GridSearchCV strategy: sequential** (Phase 4 rewritten). Stage 4A tunes linear models jointly with vectorizer params (cheap). Stage 4B fixes that vectorizer config and tunes RF only. Per-fit timing logged in Phase 3 baseline sweep informs the actual Phase 4 budget.

**What to produce this turn (Phase 0):**

1. Initialize the project repo per Phase 0 above
2. Write `CLAUDE.md` (CC working notes — your own state file) and a first-draft `project-state.md` (project ground truth)
3. Verify data sources are live: pull a sample SCDB row and a sample Oyez transcript (e.g., `cases/2014/13-1314`, *Heien v. North Carolina*)
4. Draft `reports/proposal.md` — a 1-page proposal for the professor explicitly framing this as Option 1 stance classification with all rubric requirements mapped (BoW + TF-IDF + n-grams, 3 classifiers, GridSearchCV, FN/FP business interpretation)
5. Do **NOT** bulk-download Oyez transcripts yet. Smoke-test only — one sample case.

**What to stop and report back on:**

- Final repo tree
- Confirmed working SCDB schema (paste sample row's columns and types)
- Confirmed working Oyez JSON shape (paste a snippet showing speaker-tagging structure, including how Justice speakers are distinguished from advocates)
- Draft proposal text for CAI review
- Any blockers, surprises, or design decisions made on the fly

**Pushback welcome on:**

- The 2005–2024 term window — if Oyez coverage is solid further back, propose extending after Phase 1 EDA
- Anything in the Non-Negotiables that conflicts with what the data actually looks like
- The `unanimous` flag derivation — if SCDB's vote-direction encoding makes this trickier than expected, propose an alternative
- Anything else where the plan's abstraction collides with code reality
