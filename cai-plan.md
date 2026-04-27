# Project Plan: JusticeCast — Forecasting Supreme Court Votes

## Objective

Build a **comparative study of text representations** for predicting Supreme Court Justice votes from oral-argument questions. Two parallel modeling tracks: (1) the rubric-required bag-of-words pipeline (BoW + TF-IDF + n-grams × LogReg + SVM + RF + GridSearchCV), and (2) a methodologically-appropriate sentence-embeddings track (pre-trained sentence-transformers × the same three classifiers + GridSearchCV). The contribution is the rigorous comparison itself — quantifying how much of "bench-reading" signal lives in lexical features vs semantic representations. Final deliverables are a polished, reproducible Jupyter notebook (Part B, 20 pts) and a pitch deck (Part A, 15 pts) framing the work as a legal-tech methodology study. Total: 35 pts.

## Tech Stack

- Python 3.14.3, Jupyter Notebook (primary deliverable)
- `pandas==3.0.2`, `numpy==2.4.4` — data wrangling
- `scikit-learn==1.8.0` — `CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `SVC`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedGroupKFold`, `CalibratedClassifierCV`, all metrics
- `sentence-transformers` (latest stable; pin during Phase 4.5 setup) — sentence embeddings
- `torch` (CPU-only, latest stable; pin during Phase 4.5 setup) — sentence-transformers backend
- `requests==2.33.1`, `beautifulsoup4==4.14.3`, `tenacity==9.1.4` — Oyez data fetching
- `matplotlib==3.10.9`, `seaborn==0.13.2` — visualization
- `nltk==3.9.4` — stopwords, stemming/lemmatization
- `joblib==1.5.3`, `pyarrow==24.0.0` — pipeline / fetched-data caching
- `pytest==9.0.3` — tests
- All deps pinned in `requirements.txt` + tracked unpinned in `requirements.in`

## Data Sources

1. **Supreme Court Database (SCDB)** — release `2025_01`, 83,644 vote rows × 61 columns, **Latin-1 encoding**.
2. **Oyez.org API — TWO-STEP FETCH** (verified empirically against *Heien v. North Carolina*, 2014/13-604):
   - **Step 1:** `GET https://api.oyez.org/cases/{term}/{docket}` → case-level JSON with `oral_argument_audio[]`
   - **Step 2:** for each entry, `GET https://api.oyez.org/case_media/oral_argument_audio/{audio_id}` → transcript turns
   - Multi-audio cases concatenated; cases without oral argument filtered out
   - List-response failure mode handled via `CaseNotFound`

Joined on `(term, docket_number)`. Unit of analysis: one row = `(case_id, justice_id, concatenated_question_text, vote_label)`.

**Justice ID mapping:** Hand-built `data/processed/justice_id_map.csv`, 16 Justices, all validated empirically.

## SCDB Field Semantics

| Field | Semantics |
|---|---|
| `partyWinning` | `0`=petitioner LOST, `1`=petitioner WON, `2`=unclear (EXCLUDE) |
| `majority` | `1`=dissent, `2`=majority, `NaN`=did not participate (EXCLUDE) |

**Final label derivation:** `voted_petitioner = (partyWinning == 1) == (majority == 2)`. Returns None if either field is missing or `partyWinning == 2`.

⚠️ SCDB's `majority` is encoded `1=dissent, 2=majority` — counterintuitive. Heien spot-check at Stop A caught the inversion.

## Text Preprocessing & Stopwords (BoW track)

**Preprocessing (`src/text_clean.py::preprocess_text`):**
- Strip bracketed transcription annotations
- Collapse whitespace, idempotent

**Stopwords (`STOPWORDS_FOR_VECTORIZER`, ~430 terms):**
- sklearn defaults (318) + state names (40) + agency abbreviations (35) + famous case names (21) + court-procedural terms (10) + advocate-name patterns (Phase 3.5)
- Deliberately preserves thematic legal vocabulary (`officer`, `jury`, `religious`, etc.) — these can carry stance through context

**Embeddings track (Phase 4.5) does NOT use this stopword list.** Sentence-transformers operate on full natural language; stopword removal would degrade their semantics.

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/...
│   └── processed/
│       ├── justice_id_map.csv
│       ├── justice_case_rows.parquet                    # 10,308 raw joined rows
│       ├── modeling_table.parquet                       # 10,039 rows, post-preprocess
│       └── embeddings/                                  # cached sentence-embedding arrays
│           ├── minilm_l6_v2.npy                         # (10039, 384)
│           ├── mpnet_base_v2.npy                        # (10039, 768)
│           └── row_index.parquet                        # case_id, justice_id ordering
├── src/
│   ├── fetch_scdb.py
│   ├── fetch_oyez.py
│   ├── run_bulk_fetch.py
│   ├── checkpoint1_analysis.py
│   ├── rescue_failed_dockets.py
│   ├── build_dataset.py
│   ├── build_modeling_table.py
│   ├── text_clean.py
│   ├── compute_embeddings.py                            # NEW Phase 4.5: encode + cache
│   └── modeling/
│       ├── splits.py                                    # canonical StratifiedGroupKFold
│       ├── bow_pipeline.py                              # BoW track
│       └── embedding_pipeline.py                        # NEW Phase 4.5
├── notebooks/
│   ├── 01_eda.ipynb                                     # 13 sections, B1-B6 complete
│   ├── 02_modeling_bow.ipynb                            # BoW track sweep + tuning
│   ├── 03_modeling_embeddings.ipynb                     # NEW Phase 4.5
│   └── JusticeCast_Final.ipynb                          # SUBMISSION: full comparative narrative
├── reports/
│   ├── proposal.md                                      # for the prof, due 5/7
│   ├── checkpoint1_summary.md
│   ├── ml_canvas.pdf
│   ├── JusticeCast_Pitch.pdf
│   └── results/
│       ├── baseline_results.csv                         # BoW Phase 3
│       ├── gridsearch_results.csv                       # BoW Phase 4
│       ├── embedding_baseline_results.csv               # NEW Phase 4.5
│       ├── embedding_gridsearch_results.csv             # NEW Phase 4.5
│       └── comparative_summary.csv                      # NEW Phase 5: side-by-side
├── tests/                                               # 25+ passing
├── requirements.in / requirements.txt
├── README.md
├── CLAUDE.md
└── project-state.md
```

## Non-Negotiables

These rules apply across every phase. Violations are bugs, not preferences.

1. **No data leakage.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id`. Fold 0 = test for the baseline sweep. **Both modeling tracks (BoW and embeddings) use IDENTICAL splits** — same fold definitions, same test rows. Comparison is apples-to-apples. The split logic lives in `src/modeling/splits.py` and is consumed by both pipelines.
2. **Stratified splits** on the binary vote label. `random_state=42` everywhere.
3. **Vectorizers/encoders fit on train only**, via `sklearn.pipeline.Pipeline`.
4. **No post-hoc features.** Only information available at the moment the Justice finished speaking.
5. **Reproducibility.** Fixed seed (42), pinned dependencies, notebook runs top-to-bottom on a fresh kernel.
6. **Class imbalance handled explicitly.** 62.4/37.6 balance. `class_weight='balanced'`. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged** with per-fit wall-clock time.
8. **Cache aggressively.** Oyez calls cached at both layers; SCDB downloaded once; **sentence embeddings cached to disk** (~30-min compute, painful to redo).
9. **Frame as Option 1 stance classification.**
10. **Hand-verify before bulk operations.**
11. **EDA the input, not just the labels.**
12. **Per-Justice baselines, not the global, are the comparison reference.**
13. **Honesty pass on author-identity vs stance.** Per-Justice ROC AUC on contested cases (minVotes > 0) is the cleanest test. Reported for BOTH modeling tracks in Phase 5.
14. **Honest framing of null and weak results.** When the data does not support a strong claim, the report says so plainly. The CRISP-DM business-understanding loop accommodates revising the question based on the finding.
15. **Comparative tracks use identical evaluation harness.** Phase 5 must produce a single side-by-side comparison table. Same test fold, same metrics, same per-Justice breakdown, same contested-cases test. The comparison itself is the project's contribution — anything that makes the comparison apples-to-oranges (different splits, different metrics, different evaluation code paths) is a bug.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE

⚠️ **Reminder for Saurav:** `reports/proposal.md` due to professor by **5/7** (~5 days). The proposal stays as-is — Option 1 stance classification with BoW + TF-IDF + n-grams + three classifiers + GridSearchCV. The embeddings track is over-delivery, not over-promise. Submit the existing draft.

### Phase 1: Data Acquisition ✅ COMPLETE

10,308 joined rows post-rescue. 16 Justices validated. Bulk fetch 54 min, cache 377 MB.

### Phase 2: EDA & Inclusion/Exclusion Decisions ✅ COMPLETE

10,039 rows × 20 cols, 1,293 cases. B1–B6 done. 424-term `STOPWORDS_FOR_VECTORIZER`. Length not a confound.

### Phase 3: BoW Baseline Sweep ✅ COMPLETE — null result documented

All 9 (vectorizer × classifier) combinations land ROC AUC 0.507–0.528. Top 6 linear combos statistically tied. 15 of 16 Justices show negative lift over per-Justice majority-class baseline. Top features post-stopwording are still thematic legal vocabulary, not stance markers. Two leakage findings: advocate-name fragments (`frederick`, `mr frederick`, `fisher`, `mr fisher`) leaked through.

### Phase 3.5: Pre-Phase-4 Due Diligence

Brief sanity-check pause — confirm null finding isn't a downstream pipeline bug; clean up advocate-name leakage before BoW tuning.

- [ ] **Label correctness re-verification.** Sample 10 random rows from `modeling_table.parquet` (`random_state=42`). For each, hand-look-up the case via Oyez or scotusblog: case name, actual outcome (which side won), the Justice and which side they voted with, the `voted_petitioner` value in the parquet. **Pass criterion:** 10 of 10 correct.
- [ ] **Stopword expansion for advocate-name patterns.** Add `mr <surname>`, `ms <surname>`, `general <surname>` patterns to `STOPWORDS_FOR_VECTORIZER`. Update `src/text_clean.py`. No parquet rebuild — vectorizer-side filtering only.
- [ ] **Quick re-run of `tfidf_bigram__logreg`** with expanded stopword list. Confirm advocate names are gone from top features. Expect AUC drift ≤ 0.01.
- [ ] Update `project-state.md` with the Phase 3 finding and Phase 3.5 sanity-check result.

- **CHECKPOINT 3.5:** Report 10/10 label result with case lookups documented; new stopword count; cleaned top features for `tfidf_bigram__logreg`; new AUC. Brief stop — ~30 min, not a phase.

### Phase 4: BoW GridSearchCV — Sequential Strategy

Run in full per the rubric requirement. Expectation calibrated: tuning likely lifts AUC by 1–3 points (e.g., 0.528 → ~0.55), confirming a ceiling for BoW representations.

**Stage 4A: Two `GridSearchCV` runs (LogReg, SVM-Linear) sharing the vectorizer parameter grid**

- [ ] LogReg grid: `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']` with appropriate solver
- [ ] SVM grid (LinearSVC, since BoW is high-dim sparse): `C ∈ [0.01, 0.1, 1, 10]`
- [ ] Vectorizer (joint with both): `min_df ∈ [2, 5]`, `max_df ∈ [0.9, 0.95]`, `ngram_range ∈ [(1,1), (1,2), (1,3)]`. `stop_words=STOPWORDS_FOR_VECTORIZER` (post-Phase-3.5) fixed.
- [ ] `StratifiedGroupKFold(n_splits=5)` with `groups=case_id`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] **Report mean ± std AUC across all 5 folds.** Single-fold AUC (Phase 3) was one realization of fold variance.

**Stage 4B: RF with fixed vectorizer**

- [ ] Vectorizer fixed at Stage 4A's winner
- [ ] `n_estimators ∈ [100, 300, 500]`, `max_depth ∈ [None, 20, 50]`, `min_samples_split ∈ [2, 5, 10]`

**Both stages:**

- [ ] Log to `reports/results/gridsearch_results.csv` with per-fit timing
- [ ] Refit best per model on full train, evaluate on held-out test (fold 0)
- [ ] **Compute per-Justice ROC AUC with bootstrap CIs.** Wide CIs on small-n Justices (O'Connor 30 rows, KBJackson 174) are the finding for those Justices.
- [ ] Identify final winning BoW model

**Compute budget:** ~45–60 min wall-clock.

- **CHECKPOINT 4:** Best hyperparameters per model, 5-fold CV mean ± std, CV vs test gap, final winning BoW model, per-Justice AUC table with CIs, total Phase 4 time. **Stop and wait.**

### Phase 4.5: Sentence-Embeddings Track — Peer Pipeline

This phase exists because the Phase 3 BoW finding diagnosed the ceiling and the methodologically-appropriate next step is semantic representation. **Phase 4.5 is a peer to Phase 4, not a footnote.** Same rigor, same test fold, same evaluation harness.

**Setup (one-time):**

- [ ] Pin `sentence-transformers` and `torch` (CPU) versions in `requirements.in`/`.txt`. Use CPU build of torch — no GPU dependency.
- [ ] Smoke-test: encode 5 sample utterances with `all-MiniLM-L6-v2`, verify shape `(5, 384)`, no errors.
- [ ] Implement `src/modeling/splits.py` as the canonical fold definition. Both `bow_pipeline.py` and `embedding_pipeline.py` import from here. (Refactor BoW Phase 4 code to consume this if it doesn't already — apples-to-apples is non-negotiable.)

**Embedding generation:**

- [ ] Implement `src/compute_embeddings.py`:
  - Load `modeling_table.parquet`, encode the `text` column with **two models in parallel** (separate runs):
    - `all-MiniLM-L6-v2` (384-dim, fast — ~10 min on CPU for 10K rows)
    - `all-mpnet-base-v2` (768-dim, stronger — ~30 min on CPU)
  - Cache outputs to `data/processed/embeddings/{model}.npy` plus `row_index.parquet` (preserves `(case_id, justice_id)` ordering)
  - Idempotent: skip encoding if cache exists and shape matches
  - Pytest: cache-roundtrip test, shape test, sample-text encoding test
- [ ] Compute both embedding sets. Total wall-clock: ~40 min.

**Baseline sweep on embeddings (parallel to BoW Phase 3):**

For EACH of the two embedding models:

- [ ] Run all three classifiers as untuned baselines:
  - `LogisticRegression(class_weight='balanced', max_iter=2000)`
  - `SVC(kernel='rbf', class_weight='balanced')` — RBF kernel becomes appropriate on dense vectors (was inappropriate on 200K-dim sparse BoW; this is a real methodological upgrade enabled by the representation)
  - `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
- [ ] Same fold-0 test split as BoW (via `splits.py`)
- [ ] Log to `reports/results/embedding_baseline_results.csv` with per-fit timing
- [ ] Identify top performer per embedding model

Total: 2 embedding models × 3 classifiers = 6 baseline combos. Logged side-by-side with BoW's 9.

**GridSearchCV on embeddings (parallel to BoW Phase 4):**

For the better-performing embedding model from the baseline sweep:

- [ ] **LogReg grid:** `C ∈ [0.01, 0.1, 1, 10, 100]`, `penalty ∈ ['l1','l2']`
- [ ] **SVM-RBF grid:** `C ∈ [0.1, 1, 10]`, `gamma ∈ ['scale', 0.01, 0.001]`
- [ ] **RF grid:** `n_estimators ∈ [100, 300, 500]`, `max_depth ∈ [None, 20, 50]`, `min_samples_split ∈ [2, 5, 10]`
- [ ] `StratifiedGroupKFold(n_splits=5)`, `scoring='roc_auc'`, `n_jobs=-1`
- [ ] 5-fold mean ± std reported, same as BoW
- [ ] Log to `reports/results/embedding_gridsearch_results.csv` with per-fit timing
- [ ] Per-Justice ROC AUC with bootstrap CIs
- [ ] Identify final winning embedding model

**Compute budget:** ~30–60 min wall-clock for tuning (RBF SVM is the slow part).

**What NOT to do in this phase:**
- Do NOT fine-tune the embedding model. We're using pre-trained off-the-shelf encoders — that's the comparison: out-of-the-box semantic representation vs hand-engineered BoW. Fine-tuning would be a different (and out-of-scope) project.
- Do NOT use the BoW stopword list on embedding inputs. Sentence-transformers consume natural language.
- Do NOT change the train/test split. Same fold 0 = test, identical row membership.

- **CHECKPOINT 4.5:** Report:
  - Both embedding-model baseline results (3 classifiers × 2 models = 6 combos)
  - Best classifier × embedding model after tuning, with 5-fold CV mean ± std
  - Per-Justice AUC table with bootstrap CIs for the winning embedding model
  - Side-by-side comparison BoW Phase 4 winner vs embeddings Phase 4.5 winner — preview of Phase 5's comparative table
  - Total Phase 4.5 compute time
  - **Stop and wait.**

### Phase 5: Comparative Evaluation & Interpretability

Phase 5 evaluates BOTH tracks side-by-side on the identical test fold. The comparison itself is the central narrative.

- [ ] **Comparative summary table** (`reports/results/comparative_summary.csv`) — one row per track winner, columns:
  - Representation (BoW best vectorizer config / embedding model name)
  - Classifier (winning per track)
  - Test accuracy, balanced accuracy, precision, recall, F1, ROC AUC, ROC AUC 5-fold mean±std
  - Per-Justice contested-cases ROC AUC mean (the honesty metric)
  - Wall-clock time to fit (full train) and to predict (test)
- [ ] **For each track's winner, full evaluation suite:** confusion matrix, ROC curve, PR curve, calibration curve. Both rendered in the notebook side-by-side.
  - SVM-RBF (embeddings) supports `predict_proba` natively when initialized with `probability=True`, OR via `decision_function` for AUC; calibration curve uses `CalibratedClassifierCV(method='sigmoid', cv=5)` if needed.
  - LinearSVC (BoW) wraps with `CalibratedClassifierCV` only for the calibration curve.
- [ ] **The honesty triad, BOTH tracks:**
  - (a) Per-Justice global ROC AUC with bootstrap CIs
  - (b) Per-Justice ROC AUC split by unanimity
  - (c) Per-Justice ROC AUC on contested cases only (minVotes > 0) — the strictest test
- [ ] **Per-Justice lift over individual baselines, BOTH tracks** — bar chart with both tracks shown per Justice. Visualizes: did embeddings recover real signal or just shift around?
- [ ] **Top features per track:**
  - BoW: top 30 positive coefficients, top 30 negative (LogReg/Linear-SVM); top 30 importances (RF). Compare against B1 EDA pre-stopwording features.
  - Embeddings: feature interpretation is harder for dense vectors. Approach: identify the 20 utterances with the highest predicted probability of `voted_petitioner=1` and the 20 with the lowest, paste a few representative ones in the notebook with prose interpretation. The "feature" is the semantic neighborhood, not individual tokens.
- [ ] **The comparative finding paragraph** — the deck's headline. Prose answers:
  - How much AUC did embeddings recover over BoW?
  - Did the per-Justice contested-cases AUC move? (The strictest test of bench-reading signal.)
  - Did the lift come uniformly across Justices, or concentrated in some?
  - What does the magnitude of the gap tell us about how much of bench-reading is semantic vs lexical?
- [ ] **Reframed business interpretation paragraph** — pivots to the methodology study angle:
  - The standard text-classification toolkit (BoW/TF-IDF) hits a ceiling at AUC ~X
  - Pre-trained semantic embeddings move the ceiling to AUC ~Y
  - Gap of Y−X quantifies how much signal lives in semantics that lexical features can't access
  - For a legal-tech firm: don't sell a TF-IDF question-classifier. The right product uses semantic representations at minimum, ideally combined with case features (issue area, lower-court holding, prior voting record)
  - Audio features (tone, pace, hesitation) and sequence-aware models are the next methodological frontier
- [ ] **Honest interpretation pass:**
  - What did each track actually learn? (BoW: thematic legal vocabulary + per-Justice priors. Embeddings: semantic neighborhoods + ?)
  - What does the contested-cases AUC tell us for each? (The cleanest test of true bench-reading signal.)
  - What can we honestly claim? (Don't overclaim. Don't underclaim. Report what's there.)
- [ ] All in `JusticeCast_Final.ipynb` with prose around each cell

- **CHECKPOINT 5:** Full comparative evaluation complete. CAI reviews for storytelling, honesty, and pitch-deck alignment. **Stop and wait.**

### Phase 6: ML Canvas + Notebook Polish

- [ ] Fill the Machine Learning Canvas v0.4. Frame the project as a *comparative methodology study* — the canvas's "Goal" and "Value Propositions" describe rigorous evaluation of two representation strategies, not a confidence claim about a single model.
- [ ] Export `reports/ml_canvas.pdf`
- [ ] Polish `JusticeCast_Final.ipynb`: clean markdown, smooth narrative, every section maps to a rubric line item plus the comparative-study deliverables
- [ ] Polish `README.md`: one-line description = *"Comparative study of text representations for stance classification on SCOTUS oral arguments — bag-of-words vs sentence-transformer embeddings."* Reproduce-from-fresh-clone instructions including the embedding-encoding step. Team credits.
- [ ] `Restart & Run All` on a fresh kernel — must succeed end-to-end. **Note:** the notebook should load cached embeddings from `data/processed/embeddings/`, NOT re-encode (otherwise notebook run takes 40+ min). Document this in the README under reproduction.
- [ ] `pytest` green
- **CHECKPOINT 6:** Final notebook + canvas PDF ready. **Stop and wait.**

### Phase 7: Pitch Deck (Part A) — Comparative Methodology Narrative

- [ ] 8–12 slide deck (~10 target):
  1. **Title** — JusticeCast: A Comparative Study of Text Representations for SCOTUS Vote Prediction. Team names, date.
  2. **The Hypothesis** — litigators believe they can "read the bench" from oral-argument questioning. Legal-tech firms (Lex Machina, Bloomberg Law, Westlaw Edge) are starting to monetize this intuition. Is the intuition empirically true, and which text representation captures it best?
  3. **The Data** — SCDB + Oyez, 10,039 Justice-utterance blobs across 1,293 cases (2005–2024), no leakage, careful preprocessing
  4. **Two Modeling Tracks** — pipeline diagram showing BoW track (rubric-required) and embeddings track (methodologically-appropriate alternative). Both use identical splits, classifiers, and evaluation.
  5. **ML Canvas summary**
  6. **BoW Results** — best ROC AUC ~0.55 after tuning. Per-Justice lift mostly negative. Top features are thematic legal vocabulary, not stance markers. The lexical representation has a real ceiling.
  7. **Embeddings Results** — best ROC AUC ~Y after tuning. Per-Justice lift table side-by-side with BoW. Comparative chart: BoW vs Embeddings vs per-Justice baseline.
  8. **The Comparative Finding** — gap of Y−0.55 quantifies how much bench-reading signal lives in semantics. The per-Justice contested-cases AUC tells us how much is *real* bench-reading vs author-identity-plus-priors. Headline number for the deck.
  9. **What This Means for Legal-Tech** — (a) don't sell a TF-IDF question-classifier; the lexical representation is insufficient. (b) Semantic representations are necessary baseline, not optional upgrade. (c) The remaining signal — tone, sequence, interruption patterns, audio — is the next product frontier. 2–3 concrete recommendations for a legal-tech firm.
  10. **Methodological Recommendations & Next Steps** — fine-tuned Legal-BERT, sequence-aware models on full transcripts, multimodal (audio: tone/pace/hesitation). What we couldn't test, where signal might still live.
  11. **Outro / Q&A**
- [ ] Storytelling: open with a vivid case (Citizens United, Heien fallback). Close with: *"Litigators have read the bench by gut for 200 years. We tested two computational approaches — the standard one and the methodologically-appropriate one. The gap between them tells us where the real signal lives. That's the actionable finding."*
- [ ] Export `reports/JusticeCast_Pitch.pdf`
- **CHECKPOINT 7:** Both deliverables ready for Canvas submission.

## Definition of Done

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with zero errors, loading cached embeddings rather than re-encoding
- EDA includes per-class vocabulary differences, per-Justice baseline table, sample text inspection, vocab statistics, length-vs-label check, per-Justice signature check
- BoW track: 9 baseline combinations + GridSearchCV (Stages 4A and 4B) + 5-fold CV mean ± std AUC
- Embeddings track: 6 baseline combinations (3 classifiers × 2 embedding models) + GridSearchCV on the better embedding model + 5-fold CV mean ± std AUC
- Both tracks evaluated on IDENTICAL test fold via shared `src/modeling/splits.py`
- Final winning model per track has: confusion matrix, precision, recall, F1, ROC AUC, ROC curve, PR curve, calibration curve
- **Comparative summary table** showing both tracks side-by-side
- Per-Justice performance reported as **lift over each Justice's individual baseline**, both tracks
- **Per-Justice contested-cases ROC AUC reported as a primary metric** for both tracks
- Top features for BoW (lexical) and exemplar utterances for embeddings (semantic neighborhoods) documented
- **Honest interpretation pass** comparing what each track learned
- **Reframed business interpretation paragraph** addressing the comparative finding
- Machine Learning Canvas v0.4 reframed as comparative-study and exported
- **Pitch deck reframed** as comparative methodology study
- README documents project as comparative study, with cached-embeddings reproduction
- pytest suite runs green
- All artifacts committed with clean history
- Proposal submitted to professor by **5/7**
- Both deliverables submitted to Canvas by **5/28**

## Constraints

- **Hard deadlines:** proposal 5/7 (~5 days, ⚠️ awaiting submission); both deliverables 5/28
- **Oyez API:** ≤ 1 req/sec
- **SCDB:** Latin-1 encoded, release 2025_01
- **Label derivation: locked.** `(partyWinning == 1) == (majority == 2)`
- **BoW preprocessing + stopwords: locked, with Phase 3.5 advocate-name expansion**
- **Embeddings: pre-trained only, no fine-tuning** — fine-tuning is out of scope
- **Identical splits across tracks: non-negotiable.** Apples-to-apples comparison is the project's contribution.
- **Framing:** comparative methodology study. Option 1 stance classification on the original proposal; embeddings track is over-delivery.
- **Team size:** 6. Distribution lives in chat with CAI.

## Current Instruction

**Status:** Phase 3 complete with major empirical finding. Project pivots from "single-track null result" to "comparative methodology study." Phase 3.5 is the immediate next step — brief due diligence before Phase 4. Phase 4.5 (embeddings track) is a peer to Phase 4 in scope and rigor.

**Resolutions from project-level discussion:**

- **Project reframed as comparative study.** Updated Objective, README description, Phase 7 deck narrative. Non-Negotiable #15 codifies apples-to-apples evaluation across tracks.
- **Embeddings track approved as Phase 4.5.** Same scope and rigor as Phase 4. Two embedding models (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`), same three classifiers, same hyperparameter tuning, same evaluation harness, same test fold.
- **Pre-trained embeddings only, no fine-tuning.** Out-of-the-box semantic representation is the comparison; fine-tuning would be a different study.
- **Proposal stays as-is for 5/7.** Embeddings work is over-delivery. Don't over-promise.
- **Identical splits enforced via shared `src/modeling/splits.py`.** Refactor BoW Phase 4 to consume this if not already.

**What to produce this turn (Phase 3.5 due diligence — brief):**

1. **Label correctness check (10 random rows).** Sample with `random_state=42`. For each row, paste:
   - Case name (look up via SCDB `caseName` field, cross-check on Oyez or scotusblog)
   - Actual outcome (which side won)
   - Justice name and which side they voted with
   - The `voted_petitioner` value in the parquet
   - Pass/fail
2. **Stopword expansion** — add advocate-name patterns (`mr <surname>`, `ms <surname>`, `general <surname>`) to `STOPWORDS_FOR_VECTORIZER`. Update `src/text_clean.py`. Confirm tests pass.
3. **Re-run `tfidf_bigram__logreg` baseline** with expanded stopword list. Confirm advocate names are gone from top 30 features. Report new AUC (expected within ±0.01 of original 0.528).
4. Update `project-state.md` with Phase 3 finding and Phase 3.5 sanity-check result.

**This is a brief stop — ~30 min, not a phase.** After CAI confirms 10/10 labels and cleaned features, CC proceeds directly to Phase 4 (BoW tuning) and then Phase 4.5 (embeddings track), with checkpoints between each.

**Pushback welcome on:**

- The label correctness check method — if there's a faster authoritative source, use it. Goal: rule out the boring explanation.
- The advocate-name expansion implementation — token-list vs regex, whatever's cleaner.
- The choice of two embedding models — if one of them is broken on Python 3.14 / current torch, drop it and document why. `all-MiniLM-L6-v2` is the priority; `all-mpnet-base-v2` is the optional richer comparison.
- Anything else that looks off in the Phase 3 finding that a fresh sanity-check might reveal.
