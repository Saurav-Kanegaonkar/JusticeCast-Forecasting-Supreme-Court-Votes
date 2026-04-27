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

1. **Supreme Court Database (SCDB)** — Washington University. Justice-Centered file. Free CSV. Latest release: `2025_01`. Direct URL:
   ```
   http://scdb.wustl.edu/_brickFiles/2025_01/SCDB_2025_01_justiceCentered_Citation.csv.zip
   ```
   83,644 vote rows × 61 columns. **Encoding: Latin-1 / Windows-1252** — read with `pd.read_csv(path, encoding='latin1')`.

2. **Oyez.org API — TWO-STEP FETCH** (verified empirically against *Heien v. North Carolina*, term 2014, docket 13-604):
   - **Step 1 — Case metadata:** `GET https://api.oyez.org/cases/{term}/{docket}` returns case-level JSON including `oral_argument_audio[]` array.
   - **Step 2 — Transcript:** for each entry in `oral_argument_audio[]`, follow the `href` to `https://api.oyez.org/case_media/oral_argument_audio/{audio_id}`.
   - **Multi-audio cases:** iterate over ALL audio entries per case and concatenate that Justice's utterances.
   - **Cases without oral argument** are filtered out.
   - **List-response failure mode:** when Oyez can't match a docket exactly, it returns a 30-entry search-fallback list. The fetcher detects this and raises `CaseNotFound`.

Joined on `(term, docket_number)`. Unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

**Justice ID mapping (SCDB ↔ Oyez):** Hand-built `data/processed/justice_id_map.csv` covers the 16 Justices in 2005–2024. All 16 slugs validated empirically.

## SCDB Field Semantics

| Field | Semantics |
|---|---|
| `partyWinning` | `0`=petitioner LOST, `1`=petitioner WON, `2`=unclear (EXCLUDE) |
| `majority` | `1`=dissent, `2`=majority, `NaN`=did not participate (EXCLUDE) |
| `vote` | 1..8 categorical (concurrence types). Not used directly. |
| `direction` | `1`=conservative, `2`=liberal. Not used in our binary label. |
| `caseDisposition` | 11-value taxonomy that already governs `partyWinning`. |

**Final label derivation (locked in `src/build_dataset.py::derive_voted_petitioner`):**

```
voted_petitioner = (partyWinning == 1) == (majority == 2)
```

⚠️ SCDB's `majority` is encoded `1=dissent, 2=majority` — counterintuitive. Heien spot-check at Stop A caught this.

## Text Preprocessing & Stopwords (locked in Phase 2B)

**Preprocessing (in `src/text_clean.py::preprocess_text`, applied during modeling-table build):**
- Strip bracketed transcription annotations (`[Laughter]`, `[Crosstalk]`, etc.) — 1,499 occurrences in 1,078 rows pre-cleanup, 0 post-cleanup
- Collapse multiple whitespace to single space
- Mid-utterance dashes left alone (default sklearn tokenizer drops them naturally)
- Idempotent (tested)

**Stopwords (in `src/text_clean.py::STOPWORDS_FOR_VECTORIZER`, 424 terms):**
- sklearn `ENGLISH_STOP_WORDS` (318 terms)
- + US state names (40)
- + Federal agency abbreviations (35: epa, fcc, ada, bia, pto, doj, irs, etc.)
- + Famous case shortnames (21: miranda, tinker, chevron, etc.)
- + Court-procedural terms (10: cert, certiorari, amicus, scotus, etc.)

**Why this list:** Phase 2B B1 found that pre-stopwording top-30 class-discriminative unigrams were dominated by case-topic and case-identity words (state names, agency abbreviations, famous case names) rather than stance markers. Without filtering, a baseline model would partly learn topic-→-outcome rather than the bench-questioning signal the project is designed to measure.

**What was deliberately NOT stopworded:** thematic legal vocabulary (`officer`, `jury`, `warrant`, `attorney`, `school`, `sentence`, `religious`, etc.) — these can carry stance through context. Stopwording them would cripple the model. Tested via `test_stopwords_does_not_overstrip_thematic_legal_vocab`.

**Phase 3 vectorizer integration:** all three vectorizers (BoW, TF-IDF unigram, TF-IDF bigram) consume the same `STOPWORDS_FOR_VECTORIZER` list via `stop_words=STOPWORDS_FOR_VECTORIZER`. This keeps cross-vectorizer comparisons clean — same words removed everywhere.

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
│       └── modeling_table.parquet                       # 10,039 rows, post-preprocess
├── src/
│   ├── fetch_scdb.py
│   ├── fetch_oyez.py
│   ├── run_bulk_fetch.py
│   ├── checkpoint1_analysis.py
│   ├── rescue_failed_dockets.py
│   ├── build_dataset.py
│   ├── build_modeling_table.py                          # consumes preprocess_text
│   └── text_clean.py                                    # preprocess_text + STOPWORDS_FOR_VECTORIZER
├── notebooks/
│   ├── 01_eda.ipynb                                     # 13 sections, B1-B6 complete
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
├── tests/                                               # 25 passing
├── requirements.in / requirements.txt
├── README.md
├── CLAUDE.md
└── project-state.md
```

## Non-Negotiables

These rules apply across every phase. Violations are bugs, not preferences.

1. **No data leakage. Split by `case_id` using `StratifiedGroupKFold`.** All Justices for a given case go into the same split. `train_test_split(stratify=y)` does NOT respect groups — **do not use it for the primary split**. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`: take fold 0 as the held-out test set (~20%), use folds 1–4 as train. For nested CV inside `GridSearchCV`, also use `StratifiedGroupKFold` and pass `groups=case_id` to `.fit()`.
2. **Stratified splits.** Stratify on the binary vote label. `random_state=42` everywhere.
3. **Vectorizers fit on train only.** Use `sklearn.pipeline.Pipeline`. All three vectorizers consume the same `STOPWORDS_FOR_VECTORIZER` from `text_clean.py`.
4. **No post-hoc features.** Only information available at the moment the Justice finished speaking.
5. **Reproducibility.** Fixed seed (42), pinned dependencies, notebook runs top-to-bottom on a fresh kernel via `Restart & Run All` with zero errors.
6. **Class imbalance handled explicitly.** Final balance is 62.4% petitioner / 37.6% respondent. Use `class_weight='balanced'`. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged** with per-fit wall-clock time.
8. **Cache aggressively.** Oyez calls cached at both layers; SCDB downloaded once.
9. **Frame as Option 1 stance classification.** Same machinery as sentiment, different label.
10. **Hand-verify before bulk operations.** Heien spot-check earned its keep at Stop A.
11. **EDA the input, not just the labels.** For text classification, the EDA must engage with the text — per-class vocabulary differences, sample inspection, vocab statistics, length-vs-label confound check, per-Justice signature. Phase 2B B1-B6 is the canonical execution of this rule.
12. **Per-Justice baselines, not the global, are the comparison reference.** Per-Justice baselines range ~50–80%. The model's success is per-Justice lift over each Justice's individual baseline, not over the global 62.4%. A Justice with an 80% baseline whose model scores 75% is performing *worse than majority-class* on their rows. Phase 5 enforces this framing.
13. **Honesty pass on author-identity vs stance.** B6 confirmed per-Justice vocabulary signatures are detectable. Combined with stable per-Justice voting priors, part of any model lift could come from "this is Justice X" → "Justice X votes Y" rather than bench-questioning signal. The cleanest test of REAL lift is per-Justice ROC AUC on **contested cases** (cases where minVotes > 0) — author-identity is least useful when the Justice could plausibly vote either way. Phase 5 reports this as a primary metric, not a footnote.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE

Repo: https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes

⚠️ **Reminder for Saurav:** `reports/proposal.md` needs to be submitted to the professor by **5/7** (~7 days out).

### Phase 1: Data Acquisition ✅ COMPLETE

10,308 joined rows post-rescue. Bulk fetch took 54 minutes; cache 377 MB. All 16 Justice slugs validated. List-response failure mode patched mid-run.

### Phase 2: EDA & Inclusion/Exclusion Decisions ✅ COMPLETE

#### 2A — Cleanup + Modeling Table ✅
10,039 rows × 20 cols, 1,293 cases, 16 Justices. Class balance 62.4/37.6; unanimous 41.9% / contested 58.1%.

#### 2B — EDA Expansion ✅ (Reopened Checkpoint 2 cleared)

**B1 — Per-class vocabulary differences.** Used Monroe et al. 2008 "Fightin' Words" (variance-adjusted log-odds with Dirichlet prior, α=0.01, min_df=10). **Key finding:** top-30 class-discriminative unigrams pre-stopwording are dominated by case-topic words (`officer`, `church`, `arrest`, `crack`, `algorithm`, state names, agency abbreviations) rather than stance markers. Without intervention, a baseline model would partly learn topic-→-outcome rather than bench-questioning signal. → 424-term `STOPWORDS_FOR_VECTORIZER` built (calibrated to remove case-identity words but preserve thematic legal vocabulary).

**B2 — Per-Justice baselines.** Range 50–80%. Mandatory framing prose locked into Non-Negotiable #12 and Phase 5 spec.

**B3 — Sample text inspection.** 1,499 bracketed annotations (`[Laughter]`, `[Crosstalk]`) found in 1,078 rows (10.7%). 82.8% of rows had mid-utterance dashes. → `preprocess_text()` strips brackets and normalizes whitespace; mid-utterance dashes left alone (default tokenizer drops them). Modeling table rebuilt; 0 bracket annotations remain.

**B4 — Vocabulary statistics.** 32,638 unique tokens; 5.37M total instances. Stopwords are 0.9% of vocab but 59% of token instances (textbook Zipf). Validates `min_df`-based filtering for Phase 3.

**B5 — Word count vs label.** Mann-Whitney U p=0.255. **Length is not a confound.** No length feature needed.

**B6 — Per-Justice vocabulary signature.** Detectable. → Locked Non-Negotiable #13 (per-Justice contested-cases ROC AUC as primary honesty metric in Phase 5).

### Phase 3: Modeling Pipeline (Baseline Sweep)

- [ ] Build `Pipeline` objects for each (vectorizer × classifier) combination:
  - **All three vectorizers consume `STOPWORDS_FOR_VECTORIZER` from `src.text_clean`** as `stop_words=` argument. Same 424-term list across BoW, TF-IDF unigram, TF-IDF bigram. Cross-vectorizer comparisons stay clean.
  - **Vectorizers (3):**
    - BoW: `CountVectorizer(ngram_range=(1,1), stop_words=STOPWORDS_FOR_VECTORIZER)`
    - TF-IDF unigram: `TfidfVectorizer(ngram_range=(1,1), stop_words=STOPWORDS_FOR_VECTORIZER)`
    - n-grams (TF-IDF bigrams): `TfidfVectorizer(ngram_range=(1,2), stop_words=STOPWORDS_FOR_VECTORIZER)`
  - **Classifiers (3):**
    - `LogisticRegression(class_weight='balanced', max_iter=2000)`
    - `LinearSVC(class_weight='balanced')` — `decision_function` for AUC; `CalibratedClassifierCV` only for the calibration curve in Phase 5
    - `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
  - 9 combos total
- [ ] `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id` for the 80/20 split
- [ ] Train all 9, evaluate, log to `reports/results/baseline_results.csv` with per-fit wall-clock time
- [ ] **Compare to per-Justice baselines from B2** (model lift per Justice), not just global 62.4%
- [ ] Identify top 3 performers by ROC AUC for tuning
- [ ] **Top-features comparison.** For the best linear model, extract top 30 positive and top 30 negative coefficients. **Compare these against the B1 EDA top-30 list** (pre-stopwording) and against expectations:
  - If model's top features are bigrams like `isn't true`, `are you`, `but surely`, `let me ask` → genuine stance learning. Project hypothesis supported.
  - If model's top features are still thematic legal vocabulary like `officer detained`, `warrant search`, `school district` → topic-modeling has shifted to a more granular vocabulary. Honest-interpretation pass needed in Phase 5.
  - Either outcome is a legitimate finding. Document it plainly.
- **CHECKPOINT 3:** Baseline results table with all 9 combinations, per-fit timing, top-3 selection, per-Justice lift table (model accuracy − per-Justice baseline), top-features comparison vs B1 EDA. **Stop and wait** — CAI reviews top-3 selection and approves Phase 4 compute budget.

### Phase 4: GridSearchCV — Sequential Strategy

**Stage 4A: Two `GridSearchCV` runs (one per linear model) sharing the vectorizer parameter grid**

- [ ] LogReg run: `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` (with appropriate solver)
- [ ] SVM run: `C ∈ [0.01, 0.1, 1, 10]`
- [ ] Vectorizer (joint with both): `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`
  - `stop_words=STOPWORDS_FOR_VECTORIZER` fixed across grid (not a hyperparameter — it's a project-level decision)
- [ ] `StratifiedGroupKFold(n_splits=5)` with `groups=case_id`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] Record best vectorizer config from whichever linear model wins

**Stage 4B: RF with fixed vectorizer**

- [ ] Vectorizer fixed at Stage 4A's winning config (still using `STOPWORDS_FOR_VECTORIZER`)
- [ ] `n_estimators ∈ [100, 300, 500]`, `max_depth ∈ [None, 20, 50]`, `min_samples_split ∈ [2, 5, 10]`

**Both stages:**

- [ ] Log to `reports/results/gridsearch_results.csv` with per-fit timing
- [ ] Refit best per model on full train, evaluate on held-out test
- [ ] Identify final winning model overall

- **CHECKPOINT 4:** Best hyperparameters per model, CV vs test gap, final winning model, total Phase 4 compute time. **Stop and wait.**

### Phase 5: Evaluation & Interpretability

- [ ] For winning model: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
  - If winner is `LinearSVC`: wrap with `CalibratedClassifierCV(method='sigmoid', cv=5)` **only for the calibration curve**
- [ ] **Per-Justice metrics with three views (the Phase 5 honesty triad):**
  - **(a) Per-Justice global ROC AUC** — model's overall predictive performance for each Justice
  - **(b) Per-Justice ROC AUC split by unanimity** — sensitivity analysis on whether the model's lift comes mostly from unanimous cases (where the prior is so skewed it's hard to be wrong)
  - **(c) Per-Justice ROC AUC on contested cases only (minVotes > 0)** — **the strictest test.** Author-identity is least useful when the Justice could plausibly vote either way. If the model retains meaningful AUC here, the bench-questioning signal is real. If it collapses to ~0.5, most of the apparent lift was author-identity-from-text plus per-Justice priors.
- [ ] **Per-Justice lift over individual baseline**, not over global 62.4%. Visualize: bar chart of (model accuracy − per-Justice baseline) per Justice, with the global model accuracy − global baseline shown as reference.
- [ ] **Storytelling subjects with empirical grounding:**
  - **KBJackson** (median 1,205 words, 96% coverage) — most-engaged questioner; does her chattiness translate to higher predictability or just longer text?
  - **Thomas** (295 cases post-cleanup, 20.5% speaking rate) — low-n; sensitivity treatment, not a primary claim
  - **Sotomayor / Kagan / Roberts** as the "core" predictable bench
- [ ] Top features per class:
  - LogReg/SVM: top 30 positive coefficients (predict petitioner) and top 30 negative (predict respondent). **Compare against B1 EDA findings:** does the model's top-feature list look like genuine stance markers (project hypothesis supported), or has topic-modeling shifted to a more granular vocabulary (honest-interpretation discussion)?
  - RF: top 30 feature importances
- [ ] **Honest interpretation pass** — explicit prose section in the notebook addressing:
  - What did the model actually learn? Stance markers, thematic legal vocabulary, or per-Justice signatures?
  - What does the contested-cases AUC tell us about the project's headline claim?
  - What can we honestly say about "reading the bench" given these findings?
  - Don't overclaim. Don't underclaim. Report what's there.
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
  8. Approach — pipeline diagram, vectorizers (with our calibrated stopword decision), classifiers, eval
  9. **Results** — confusion matrix, ROC AUC, per-Justice lift over individual baselines, KBJackson vs Thomas storytelling, contested-cases AUC as the rigorous test
  10. Recommendations — go-to-market, risks, next steps
  11. Outro / Q&A
- [ ] Storytelling: open with a vivid case (Citizens United if rescue succeeded; Heien as fallback), bookend with the same case
- [ ] Export `reports/JusticeCast_Pitch.pdf`
- **CHECKPOINT 7:** Both deliverables ready for Canvas submission.

## Definition of Done

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with zero errors
- EDA includes per-class vocabulary differences, per-Justice baseline table, sample text inspection, vocab statistics, length-vs-label check, per-Justice signature check (Phase 2B B1-B6 ✅)
- Custom stopword list (`STOPWORDS_FOR_VECTORIZER`) consumed by all three vectorizers
- All 9 vectorizer × classifier baseline combinations evaluated and logged with per-fit timing
- GridSearchCV applied via the sequential strategy
- Final winning model has: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
- Per-Justice performance reported as **lift over each Justice's individual baseline**, not global
- **Per-Justice contested-cases ROC AUC reported as a primary metric** (Non-Negotiable #13 honesty triad)
- Top n-grams for each class extracted; comparison between B1 EDA pre-modeling list and post-modeling top features documented
- **Honest interpretation pass** in Phase 5 prose: what the model learned, contested-cases AUC interpretation, what can be honestly claimed about "reading the bench"
- Business interpretation paragraph (FN vs FP cost) in the notebook prose
- Machine Learning Canvas v0.4 filled in and exported as PDF
- Pitch deck 8–12 slides, exported as PDF, follows the storytelling arc
- README documents how to reproduce from a fresh clone
- pytest suite runs green
- All artifacts committed with clean history
- Proposal submitted to professor by **5/7** (Phase 0 ✅ drafted, ⚠️ awaiting submission)
- Both deliverables submitted to Canvas by **5/28**

## Constraints

- **Hard deadlines:** proposal 5/7 (drafted, ⚠️ awaiting submission, ~7 days out); both deliverables 5/28
- **Oyez API:** ≤ 1 req/sec across the 2-step fetch
- **SCDB:** Justice-Centered file, release 2025_01. Latin-1 / Windows-1252 encoded
- **Label derivation: locked.** `(partyWinning == 1) == (majority == 2)`, with `partyWinning == 2` and `majority NaN` excluded
- **Preprocessing + stopwords: locked.** `preprocess_text` + 424-term `STOPWORDS_FOR_VECTORIZER` from `src/text_clean.py`
- **Framing:** Option 1 stance classification, not sentiment
- **Team size:** 6. Distribution lives in chat with CAI

## Current Instruction

**Status:** Phase 2 complete (Reopened Checkpoint 2 cleared with all B1–B6 done). Custom stopword list and preprocessing locked in. CC is approved to execute Phase 3.

**Resolutions from Phase 2B:**

- **B1 vocabulary findings: confirmed and acted on.** The 424-term `STOPWORDS_FOR_VECTORIZER` is the right calibration — removes case-identity words while preserving thematic legal vocabulary that can carry stance through context.
- **B2 per-Justice baseline framing: locked into Non-Negotiable #12.** Phase 5 evaluates per-Justice lift over individual baselines, never global.
- **B3 preprocessing: applied; modeling table rebuilt.** 0 bracketed annotations remain.
- **B5 length-confound check: ruled out.** No length feature needed.
- **B6 per-Justice signature: detectable.** → Non-Negotiable #13 + Phase 5 honesty triad: per-Justice contested-cases ROC AUC is now a primary metric, not a footnote.
- **Fightin' Words (Monroe et al.) for class-vocabulary comparison: approved post-hoc.** Better choice than my naive log-odds suggestion.

**What to produce this turn (Phase 3 baseline sweep):**

1. Build the 9 `Pipeline` objects per the spec above. All three vectorizers pull `STOPWORDS_FOR_VECTORIZER` from `src/text_clean.py`.
2. Define the train/test split using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id` — fold 0 = test, folds 1–4 = train.
3. Train all 9 combinations, evaluate on the held-out test fold, log to `reports/results/baseline_results.csv` with per-fit wall-clock time.
4. Compute per-Justice accuracy and per-Justice lift over each Justice's individual baseline (from B2). Report as a table.
5. For the best linear model (whichever has highest ROC AUC among the linear combos), extract top 30 positive coefficients and top 30 negative coefficients. Compare to B1 EDA top-30 list:
   - Are stance markers rising to the top now that case-identity words are stopworded?
   - Or has thematic legal vocabulary become the new topic-proxy?
   - Document the answer plainly in the Checkpoint 3 report.
6. Identify top 3 performers by ROC AUC for Phase 4 tuning.
7. `pytest` should remain green throughout.

**What to stop and report back on (Checkpoint 3):**

- Baseline results table: all 9 combinations × {accuracy, balanced accuracy, precision, recall, F1, ROC AUC, per-fit time}
- Per-Justice lift table: model's per-Justice accuracy minus per-Justice baseline accuracy
- Top 3 selection by ROC AUC, with margin of confidence (CV fold variance if reasonable to compute alongside)
- Top-30 features for the best linear model + comparison narrative against B1 EDA findings
- Phase 4 compute-budget extrapolation from observed per-fit timings
- Any surprises: a combination that underperforms the global majority baseline (62.4%) means something is mechanically wrong; surface it immediately

**Pushback welcome on:**

- The 9-combo full grid being unnecessary if early results show one classifier family clearly dominates — but err toward completing the full sweep for the rubric's "compare three vectorizers × three classifiers" requirement
- The top-features comparison framing — if you find a third pattern (neither stance markers nor thematic-vocabulary-as-topic-proxy), document it as the actual finding rather than forcing it into one of the two predicted boxes
- Anything in the per-Justice lift framing that turns out to be statistically misleading at the row counts we have (e.g., O'Connor's 30 rows, KBJackson's 174 — wide confidence intervals on per-Justice metrics; flag this if it would distort the honest reading)
