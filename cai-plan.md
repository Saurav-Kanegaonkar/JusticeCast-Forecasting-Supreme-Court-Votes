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
   83,644 vote rows × 61 columns. **Encoding: Latin-1 / Windows-1252** (not UTF-8). Read with `pd.read_csv(path, encoding='latin1')`. Contains every recorded SCOTUS vote with petitioner/respondent winner, majority/dissent flags, ideological direction, etc.

2. **Oyez.org API — TWO-STEP FETCH** (verified empirically against *Heien v. North Carolina*, term 2014, docket 13-1314):
   - **Step 1 — Case metadata:** `GET https://api.oyez.org/cases/{term}/{docket}` returns case-level JSON including `oral_argument_audio[]` array of links. **Does NOT contain the transcript turns.**
   - **Step 2 — Transcript:** for each entry in `oral_argument_audio[]`, follow the `href` to `https://api.oyez.org/case_media/oral_argument_audio/{audio_id}`. That JSON contains:
     ```
     transcript.sections[].turns[].speaker.identifier   # slug, e.g. "john_g_roberts_jr"
     transcript.sections[].turns[].speaker.name         # full name
     transcript.sections[].turns[].speaker.roles        # array; roles[].type == "scotus_justice" filters for Justices
     transcript.sections[].turns[].speaker.ID           # numeric (Oyez-internal)
     transcript.sections[].turns[].text_blocks[].text   # utterance text
     ```
   - **Multi-audio cases:** some cases (re-arguments, etc.) have multiple `oral_argument_audio[]` entries. Iterate over ALL of them per case and concatenate that Justice's utterances across argument sessions. Store the count as metadata (`n_audio_sessions`).
   - **Cases without oral argument** (`oral_argument_audio == []`) are filtered out — decided on briefs, no text to model.
   - Free, public, no auth required.

Joined on `(term, docket_number)`. Unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

**Justice ID mapping (SCDB ↔ Oyez):** SCDB uses numeric IDs (e.g., `80180`) and short codes (e.g., `HHBurton`); Oyez uses slugs (e.g., `john_g_roberts_jr`). Build a manual `data/processed/justice_id_map.csv` with one row per Justice in our window — for 2005–2024 that's ~15 Justices total, hand-mapping is fast and avoids fuzzy-match errors.

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/
│   │   ├── scdb_justice.csv                             # downloaded once, latin-1
│   │   └── oyez/
│   │       ├── cases/{term}_{docket}.json               # case metadata (Step 1)
│   │       └── transcripts/{audio_id}.json              # transcript turns (Step 2)
│   └── processed/
│       ├── justice_id_map.csv                           # SCDB ↔ Oyez Justice key
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

1. **No data leakage. Split by `case_id` using `StratifiedGroupKFold`.** All Justices for a given case go into the same split. `train_test_split(stratify=y)` does NOT respect groups and would leak cases across train/test — **do not use it for the primary split**. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`: take fold 0 as the held-out test set (~20%), use folds 1–4 as train. For nested CV inside `GridSearchCV`, also use `StratifiedGroupKFold` and pass `groups=case_id` to `.fit()`.
2. **Stratified splits.** Stratify on the binary vote label so train and test have matching class balance. `random_state=42` everywhere.
3. **Vectorizers fit on train only.** No vocabulary built from test data, ever. Use `sklearn.pipeline.Pipeline` so this is enforced by construction, not by discipline.
4. **No post-hoc features.** Only use information available at the moment the Justice finished speaking — no decision text, no opinion text, no outcome-derived features in `X`. The vote label is the *only* future signal we touch.
5. **Reproducibility.** Fixed seed (42), pinned dependencies, notebook runs top-to-bottom on a fresh kernel via `Restart & Run All` with zero errors.
6. **Class imbalance handled explicitly.** Petitioner-side wins ~65–70% of SCOTUS cases. Use `class_weight='balanced'` or document the choice. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged.** A `reports/results/` table with one row per (vectorizer, classifier, hyperparams) combination — accuracy, precision, recall, F1, ROC AUC, **and per-fit wall-clock time**. The notebook reads from these tables; we don't re-run sweeps to re-knit the notebook.
8. **Cache aggressively.** Oyez calls cached to disk (both layers); SCDB downloaded once. Reruns are fast.
9. **Frame as Option 1 stance classification, not sentiment.** The rubric mentions "sentiment analysis" because Option 2 is sentiment. We are Option 1 (custom topic). Our task is **stance classification** — same machinery, different label. The proposal makes this explicit so the professor signs off on the framing.
10. **Hand-verify before bulk operations.** Any irreversible or expensive step (bulk API fetch, multi-hour grid search) gets a smoke test on a hand-checked sample first. The cost of a mid-phase pause is small; the cost of redoing an hour-long operation because of a silent logic bug is large.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE (5/7 deadline)

Repo: https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes

Shipped at Checkpoint 0: repo scaffold, venv, pinned deps, smoke-tested SCDB + Oyez (2-step fetch design discovered), draft proposal at `reports/proposal.md`. See `project-state.md` for full details.

### Phase 1: Data Acquisition (Two Stops)

Phase 1 has two explicit stops because the bulk Oyez fetch is a ~50–70 minute irreversible network operation. Stop A verifies all logic on a single hand-checked case before committing to the bulk pull.

---

#### Stop A — Pre-Bulk-Fetch Verification

**Goal:** every piece of logic that will run during the bulk fetch must be verified on Heien (term 2014, docket 13-1314) before any bulk fetching happens.

**1. Codebook verification (do this first):**

- [ ] Download the SCDB codebook PDF from scdb.wustl.edu
- [ ] Document field semantics for `partyWinning`, `majority`, `vote`, `direction`, `caseDisposition` in `project-state.md` with codebook page citations
- [ ] **Specifically resolve the encoding of `majority`.** SCDB historically codes this as `1=dissent, 2=majority`, which is counterintuitive and would silently invert the candidate XNOR derivation. Confirm or correct.
- [ ] Confirm the `partyWinning=6` reported in Checkpoint 0 was a parsing artifact (CC's awk-on-quoted-CSV bug, already self-diagnosed) and that the actual value space for `partyWinning` is what the codebook describes.

**2. Lock in label derivation:**

- [ ] Encode the binary label derivation in `build_dataset.py` with a docstring citing the codebook section
- [ ] **Heien spot-check (mandatory):** *Heien v. North Carolina* was decided 8-1 with NC (respondent) winning. Sotomayor was the lone dissenter. Expected labels: Sotomayor `voted_petitioner=1`, all eight others `voted_petitioner=0`. Run the derivation code on Heien's nine SCDB rows and confirm the output matches by hand. If it doesn't, the logic is wrong — fix before bulk fetch.

**3. Fetchers:**

- [ ] `fetch_scdb.py`: download zip, extract, **`encoding='latin1'`**, cache to `data/raw/scdb_justice.csv`. Idempotent (skip if cached).
- [ ] `fetch_oyez.py`: 2-step fetch with caching at both layers, ≤ 1 req/sec global rate limit, `tenacity` exponential backoff. **Smoke-test on Heien only — no bulk yet.**

**4. Justice ID map:**

- [ ] Hand-build `data/processed/justice_id_map.csv` for the ~15 Justices appearing in 2005–2024. Columns: `oyez_identifier`, `scdb_justice_id`, `scdb_justice_name`, `display_name`.

**5. End-to-end pipeline rehearsal on Heien only:**

- [ ] Implement `build_dataset.py`
- [ ] Run on the Heien-only cache: should produce 9 rows (one per Justice who participated), with the correct labels confirmed in step 2
- [ ] Verify multi-audio handling on a known multi-session case if one comes up; if Heien is single-session, hand-check at least one multi-session case in Stop B

**6. Bulk-fetch budget estimate:**

- [ ] Query SCDB for `count(*) WHERE 2005 ≤ term ≤ 2024` to get the case-count target
- [ ] Estimate wall-clock for bulk fetch: cases × avg_audio_per_case × (1 second + average request latency)
- [ ] Report the estimate

- **STOP A REPORT** — Report back to CAI:
  - Codebook field semantics for the five label-relevant fields, with citations
  - Final label derivation code + Heien spot-check result (pass/fail with the 9-row table shown)
  - `justice_id_map.csv` contents
  - Heien end-to-end pipeline output (the 9 rows)
  - Bulk-fetch volume estimate (case count + wall-clock)
  - Any anomalies or codebook surprises
  - **Stop and wait for CAI sign-off before bulk fetching.**

---

#### Stop B — Bulk Fetch + Final Build → Checkpoint 1

After Stop A is signed off:

- [ ] Bulk-fetch all 2005–2024 cases through `fetch_oyez.py` (both layers, both cached)
- [ ] Build the full joined parquet via `build_dataset.py`:
  - Parse each transcript JSON: extract every turn, filter where `any(role.type == 'scotus_justice' for role in speaker.roles)`
  - Multi-audio aggregation: for cases with multiple `oral_argument_audio[]` entries, concatenate that Justice's utterances across ALL argument sessions. Store `n_audio_sessions` as metadata.
  - Filter cases with no oral argument (`oral_argument_audio == []`)
  - Aggregate per `(case_id, justice_id)`: concatenated text + turn_count + word_count + n_audio_sessions metadata
  - Join SCDB → Oyez via `(term, docket_number)` and Justice ID map
  - Annotate each row with `unanimous` flag (`minVotes == 0`, cross-check: `majVotes + minVotes == total_voting_justices`)
  - Compute the binary label using the verified derivation from Stop A
  - Write `data/processed/justice_case_rows.parquet`
- [ ] Tests:
  - `tests/test_fetchers.py`: SCDB row-count sanity check; Oyez Step 1 returns expected case-metadata shape; Oyez Step 2 returns expected transcript shape; rate limiter works
  - `tests/test_builders.py`: parser correctly attributes Justice vs advocate utterances on a sample transcript; multi-audio concatenation works on a known multi-session case; label derivation matches codebook on hand-checked rows (Heien is one of these); every parsed Justice maps to SCDB via the ID map
- [ ] Run `pytest` — all green

- **CHECKPOINT 1** — Report:
  - Final row counts at each stage of the pipeline (cases attempted → fetched → with valid oral arg → with all Justices mapped → final joined)
  - Justice coverage breakdown for 2005–2024 (which Justices, how many cases each)
  - Median word counts per Justice (expect Thomas as a tail)
  - % of rows flagged unanimous
  - Verified label derivation logic + binary class distribution
  - Multi-audio cases encountered and how the concatenation worked
  - Any fetch or parse failures, with reasons
  - Total Oyez API calls made and elapsed wall-clock time (helps Phase 3+ budget)
  - **Stop and wait.**

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

**Stage 4A: Tune linear models with vectorizer params (two GridSearchCV runs, one per linear model, sharing the same vectorizer parameter grid)**

- [ ] Run `GridSearchCV` for LogReg over its hyperparams + vectorizer params:
  - **LogReg grid:** `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` with appropriate solver (e.g., `liblinear`)
  - **Vectorizer (joint):** `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`
- [ ] Run `GridSearchCV` for SVM over its hyperparams + the same vectorizer param grid:
  - **SVM grid:** `C ∈ [0.01, 0.1, 1, 10]`
- [ ] Both runs use `StratifiedGroupKFold(n_splits=5)` with `groups=case_id`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] Record the **best vectorizer config** from whichever linear model wins on CV ROC AUC — this becomes the fixed vectorizer for RF in Stage 4B

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
- [ ] Polish `README.md` (added in Phase 0): finalize project summary, how to reproduce from a fresh clone, team credits
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
- GridSearchCV applied via the sequential strategy (Stage 4A two linear-model runs sharing the vectorizer grid; Stage 4B RF with fixed vectorizer)
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
- Proposal submitted to professor by **5/7** (Phase 0 ✅)
- Both deliverables submitted to Canvas by **5/28**

## Constraints

- **Hard deadlines:** proposal 5/7 (✅ drafted, awaiting submission); both deliverables 5/28. Do not slip.
- **Oyez API:** be polite (≤ 1 req/sec across the 2-step fetch). Coverage is best for 2005+ terms.
- **SCDB:** Justice-Centered file, release 2025_01. **CSV is Latin-1 / Windows-1252 encoded** — mojibake on UTF-8 read. Always read with `encoding='latin1'`.
- **Label derivation:** must be codebook-verified before computing. Heien spot-check (Sotomayor=1, others=0) is a mandatory pass before bulk fetch.
- **Framing:** professor's rubric mentions "Sentiment Analysis" because Option 2 is sentiment. Our framing is **stance classification** under Option 1. The proposal makes this explicit and the professor signs off; if professor pushes back, fall back to "stance toward petitioner = positive sentiment toward petitioner's argument."
- **Team size:** 6. Distribution lives in chat with CAI, not in this file.

## Current Instruction

**Status:** Phase 0 complete (Checkpoint 0 ✅). Phase 1 split into Stop A (pre-bulk-fetch verification) and Stop B (bulk fetch → Checkpoint 1). CC begins with Stop A.

**Resolutions from Checkpoint 0 + Phase 1 plan dialog:**

- `partyWinning=6` from Checkpoint 0 was a CC-side awk parsing bug (commas inside quoted `caseName`). Real value is `1`. Self-diagnosed and acknowledged. The codebook-first verification step still applies — we lock in label semantics formally before computing labels at scale.
- **Two-stop Phase 1 split: approved.** Asymmetric reversibility (5 min review vs 50–70 min refetch) makes the interim pause clearly worth it. New Non-Negotiable #10 codifies this principle for future expensive operations.
- **Hand-built Justice ID map: approved.** ~15 rows; faster than vetting an external crosswalk.
- **Heien spot-check on label derivation: required, not optional.** Sotomayor=1, all others=0 is the expected output. Catches inverted-XNOR bugs from `majority` field encoding before they propagate to the bulk pipeline.

**What to produce this turn (Phase 1, Stop A only):**

1. Codebook download + field-semantics writeup in `project-state.md` (especially `majority` encoding — confirm or correct the assumption that `majority == 1` means "in majority")
2. Lock label derivation in `build_dataset.py` with codebook citation
3. **Heien spot-check** — run derivation on Heien's 9 SCDB rows, verify Sotomayor=1 / others=0
4. Implement `fetch_scdb.py` (latin-1, idempotent)
5. Implement `fetch_oyez.py` (2-step, cached, rate-limited, tenacity retries) — smoke-test on Heien only, no bulk
6. Hand-build `data/processed/justice_id_map.csv` (~15 rows for the 2005–2024 window)
7. Implement `build_dataset.py` and run end-to-end on the Heien-only cache (should produce 9 rows with correct labels)
8. Estimate the bulk-fetch volume and wall-clock budget

**What to stop and report back on (Stop A report):**

- Codebook field semantics for the five label-relevant fields with citations
- Final label derivation code + Heien spot-check result (pass/fail with the 9-row table shown so CAI can hand-check)
- `justice_id_map.csv` contents
- Heien end-to-end pipeline output (the 9 rows: case_id, justice_id, word_count, voted_petitioner, unanimous, n_audio_sessions)
- Bulk-fetch volume estimate (case count + wall-clock minutes)
- Any anomalies, codebook surprises, or Oyez quirks discovered along the way

**DO NOT** bulk-fetch yet. Stop A explicitly halts before any large network operation.

**Pushback welcome on:**

- The 2005–2024 window — if you find Oyez coverage is reliable further back without quality drop, propose extending in the Stop A report (don't unilaterally extend)
- Any codebook semantics that contradict the assumed label derivation
- Any case in the Heien spot-check that doesn't behave as expected — that's the signal to stop and re-derive, not to paper over
- The `unanimous` flag derivation (`minVotes == 0`) — if the codebook reveals a cleaner field, switch
- Anything in the Non-Negotiables that conflicts with what the data actually looks like
