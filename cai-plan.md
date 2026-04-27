# Project Plan: JusticeCast — Forecasting Supreme Court Votes

## Objective

A **comparative study of text representations** for predicting Supreme Court Justice votes from oral-argument questions. Two parallel modeling tracks: (1) the rubric-required bag-of-words pipeline (BoW + TF-IDF + n-grams × LogReg + SVM + RF + GridSearchCV), and (2) a methodologically-appropriate sentence-embeddings track (pre-trained sentence-transformers × the same three classifiers + GridSearchCV). The contribution is the rigorous comparison itself — quantifying how much of "bench-reading" signal lives in lexical features vs semantic representations. The contested-cases test (Phase 5) confirms the embedding lift is real bench-questioning signal, not just author-identity-plus-priors recovery. Final deliverables are a polished, reproducible Jupyter notebook (Part B, 20 pts) and a pitch deck (Part A, 15 pts) framing the work as a legal-tech methodology study, **structured explicitly around the CRISP-DM Data Science Process Model** (Business Understanding → Data Understanding → Data Preparation → Modeling → Model Evaluation → Model Deployment) per the course rubric. Total: 35 pts.

## Tech Stack

- Python 3.14.3, Jupyter Notebook (primary deliverable)
- `pandas==3.0.2`, `numpy==2.4.4`
- `scikit-learn==1.8.0` — vectorizers, classifiers, GridSearchCV, CV splitters, metrics
- `sentence-transformers` + CPU `torch` — `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- `requests==2.33.1`, `beautifulsoup4==4.14.3`, `tenacity==9.1.4`
- `matplotlib==3.10.9`, `seaborn==0.13.2`
- `nltk==3.9.4`
- `joblib==1.5.3`, `pyarrow==24.0.0`
- `pytest==9.0.3` — 90 tests passing

## Data Sources

1. **Supreme Court Database (SCDB)** — release `2025_01`, Justice-Centered file, 83,644 vote rows × 61 columns, **Latin-1 encoding**.
2. **Oyez.org API** — TWO-STEP FETCH (case metadata then transcript audio JSON), multi-audio cases concatenated, list-response failure mode handled via `CaseNotFound`.

Joined on `(term, docket_number)`. Final modeling table: 10,039 rows, 1,293 cases, 16 Justices.

## SCDB Field Semantics

| Field | Semantics |
|---|---|
| `partyWinning` | `0`=petitioner LOST, `1`=petitioner WON, `2`=unclear (EXCLUDE) |
| `majority` | `1`=dissent, `2`=majority, `NaN`=did not participate (EXCLUDE) |

**Final label derivation (locked):** `voted_petitioner = (partyWinning == 1) == (majority == 2)`.

⚠️ SCDB's `majority` is encoded `1=dissent, 2=majority` — counterintuitive. Heien spot-check at Stop A caught the inversion.

## Text Preprocessing & Stopwords (BoW track only)

- **Preprocessing:** `preprocess_text` strips bracketed transcription annotations, collapses whitespace, idempotent
- **Stopwords:** `STOPWORDS_FOR_VECTORIZER` = sklearn defaults (318) + state names (40) + agency abbreviations (35) + famous case names (21) + court-procedural terms (10). Phase 3.5 added advocate-name patterns via vectorizer `preprocessor=` callable.
- Deliberately preserves thematic legal vocabulary (`officer`, `jury`, `religious`)
- **Embeddings track does NOT use this stopword list.**

## Architecture

```
JusticeCast/
├── data/
│   ├── raw/...
│   └── processed/
│       ├── justice_id_map.csv
│       ├── justice_case_rows.parquet
│       ├── modeling_table.parquet                       # 10,039 rows
│       └── embeddings/
│           ├── minilm_l6_v2.npy                         # (10039, 384)
│           ├── mpnet_base_v2.npy                        # (10039, 768)
│           └── row_index.parquet
├── src/
│   ├── fetch_scdb.py
│   ├── fetch_oyez.py
│   ├── run_bulk_fetch.py
│   ├── checkpoint1_analysis.py
│   ├── rescue_failed_dockets.py
│   ├── build_dataset.py
│   ├── build_modeling_table.py
│   ├── text_clean.py
│   ├── compute_embeddings.py
│   └── modeling/
│       ├── splits.py
│       ├── bow_pipeline.py
│       └── embedding_pipeline.py
├── notebooks/
│   ├── 01_eda.ipynb                                     # working
│   ├── 02_modeling_bow.ipynb                            # working
│   ├── 03_modeling_embeddings.ipynb                     # working
│   ├── 02_phase5_comparative.ipynb                      # working
│   └── JusticeCast_Final.ipynb                          # SUBMISSION (CRISP-DM structured)
├── reports/
│   ├── proposal.md
│   ├── checkpoint1_summary.md
│   ├── ml_canvas.pdf
│   ├── JusticeCast_Pitch.pdf
│   └── results/                                         # 11+ CSV artifacts
├── tests/                                               # 90 passing
├── requirements.in / requirements.txt
├── README.md
├── CLAUDE.md
└── project-state.md
```

## Non-Negotiables

These rules apply across every phase. Violations are bugs, not preferences.

1. **No data leakage.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id`. Both tracks use IDENTICAL splits via shared `src/modeling/splits.py`.
2. **Stratified splits** on the binary vote label. `random_state=42` everywhere.
3. **Vectorizers/encoders fit on train only**, via `sklearn.pipeline.Pipeline`.
4. **No post-hoc features.** Only information available at the moment the Justice finished speaking.
5. **Reproducibility.** Fixed seed (42), pinned dependencies, notebook runs top-to-bottom on a fresh kernel.
6. **Class imbalance handled explicitly.** 62.4/37.6 balance. `class_weight='balanced'`. Report ROC AUC and balanced accuracy alongside raw accuracy.
7. **Every experiment logged** with per-fit wall-clock time.
8. **Cache aggressively.** Embeddings cached to disk — notebook loads cached arrays, does NOT re-encode (saves ~12 min).
9. **Frame as Option 1 stance classification.**
10. **Hand-verify before bulk operations.**
11. **EDA the input, not just the labels.**
12. **Per-Justice baselines, not the global, are the comparison reference.**
13. **Honesty pass on author-identity vs stance.** Per-Justice ROC AUC on contested cases (minVotes > 0) is the primary honesty metric. Phase 5 confirmed the embedding lift survives the strict test (+4 pp on contested cases vs +5 pp unanimous).
14. **Honest framing of null and weak results.** Absolute AUC of 0.569 is modest — frame as "lower bound on bench-reading from text alone."
15. **Comparative tracks use identical evaluation harness.**
16. **CRISP-DM is the visible structure of the submission notebook.** The course rubric explicitly asks for "Apply Data Science Process Model as a guide." The submission notebook uses CRISP-DM's six phases (Business Understanding → Data Understanding → Data Preparation → Modeling → Model Evaluation → Model Deployment) as primary top-level section headers, so the framework is visible to the grader at first glance, not implicit. Same applies to the ML Canvas wording and the pitch deck flow.

## Implementation Phases

### Phase 0: Proposal & Repo Init ✅ COMPLETE

Repo: https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes
Proposal sent to Professor Sharad Gupta on 2026-04-26 (well ahead of 5/7 deadline).

### Phase 1: Data Acquisition ✅ COMPLETE

10,308 joined rows post-rescue. 16 Justices validated. Bulk fetch 54 min, cache 377 MB.

### Phase 2: EDA & Cleanup ✅ COMPLETE

10,039 rows × 20 cols, 1,293 cases. B1–B6 done. Custom 424-term stopword list. Length not a confound.

### Phase 3: BoW Baseline Sweep ✅ COMPLETE

All 9 combinations land ROC AUC 0.507–0.528. Top features dominated by thematic legal vocabulary.

### Phase 3.5: Pre-Phase-4 Due Diligence ✅ COMPLETE

10/10 label correctness verified. Advocate-name preprocessor added.

### Phase 4: BoW GridSearchCV ✅ COMPLETE

Tuning lifted ROC AUC to 0.5323 test (+0.4 pp, exactly the predicted ceiling). Best linear: LinearSVC, C=0.01 (max regularization), unigrams. Only 2 of 15 Justices have CI lower bound above 0.5. KBJackson worst-predicted at AUC 0.406.

### Phase 4.5: Sentence-Embeddings Track ✅ COMPLETE

LogReg on MiniLM-L6-v2 lifts ROC AUC to 0.5691 — **+3.7 pp over BoW**. KBJackson flips: BoW 0.406 → Embeddings 0.635. Thomas gains +0.193. Kennedy regresses (-0.101).

### Phase 5: Comparative Evaluation ✅ COMPLETE

Contested-cases test passed: embeddings retain +4 pp lift on the strict slice. KBJackson contested: BoW 0.405 → Embeddings 0.643 (+0.238). The deck's centerpiece anecdote.

### Phase 6: ML Canvas + Final Notebook + README

Build the polished Canvas-submission deliverables. **Notebook structured around CRISP-DM's six phases per Non-Negotiable #16.**

#### 6.1 — Build `JusticeCast_Final.ipynb` as the CRISP-DM-structured submission notebook

Working notebooks (`01_eda.ipynb`, `02_modeling_bow.ipynb`, `03_modeling_embeddings.ipynb`, `02_phase5_comparative.ipynb`) stay in the repo as the development trail. The Canvas submission is a fresh, polished, top-to-bottom narrative — six top-level sections matching CRISP-DM exactly, with subsections folded inside. Each top-level phase opens with a one-sentence framing line ("This phase focuses on...") so the grader sees the framework explicitly.

**Top-level structure (six sections, matching the six CRISP-DM phases):**

##### Section 1 — Business Understanding

*This phase focuses on understanding the project objectives from a business perspective.* (Per Class 2 slides: determine business objectives, develop success criteria, determine actions to be taken on predictions.)

- [ ] **1.1 Project framing & hypothesis.** Title, team, the question ("can a model read the bench from oral-argument text?"), why it matters (legal-tech firms monetize adjacent products; no widely-available product turns transcripts into per-Justice forecasts).
- [ ] **1.2 Success criteria.** Empirical: ROC AUC meaningfully above the per-Justice majority-class baseline on contested cases. Methodological: rigorous apples-to-apples comparison of BoW vs semantic representations. Honest: report what the data shows, not what we hoped to find.
- [ ] **1.3 Actions to be taken on predictions.** A per-Justice vote forecast informs (a) pre-argument prep targeting, (b) amicus brief targeting within hours of argument, (c) historical bench-reading benchmarks for litigation press. False-positive cost: prep wasted on a Justice who'll vote the other way. False-negative cost: missed sympathetic Justice.
- [ ] **1.4 Stakeholders.** Appellate litigators, amicus brief authors, legal-tech vendors (Lex Machina, Bloomberg Law, Westlaw Edge, SCOTUSblog), litigation press.
- [ ] **1.5 Comparative-study design.** Two parallel tracks (BoW + sentence embeddings), identical evaluation harness, contested-cases honesty test. The comparison is the contribution.
- [ ] **1.6 Rubric mapping.** Brief table showing where each rubric requirement (3 vectorizers, 3 classifiers, GridSearchCV, ML Canvas, business interpretation) lives in this notebook.

##### Section 2 — Data Understanding

*This phase focuses on initial data collection, becoming familiar with the data, and identifying data quality problems.* (Per Class 2 slides: collect relevant data, profile for quality issues, explore and visualize, identify transformation needs.)

- [ ] **2.1 Data sources.** SCDB (Justice-Centered file, release 2025_01, Latin-1, 83,644 vote rows) and Oyez API (two-step fetch: case metadata → transcript audio JSON). Both free, both public.
- [ ] **2.2 Data flow diagram.** SCDB rows + Oyez transcripts → joined on (term, docket) → per (case, Justice) blob → modeling table.
- [ ] **2.3 Coverage profile.** 1,470 unique (term, docket) pairs in 2005–2024 → 1,322 with valid oral argument → 1,307 cases successfully fetched + parsed. Justice ID mapping covers all 16 Justices in the window. Cache footprint: 377 MB.
- [ ] **2.4 Data quality findings.** Bracketed transcription annotations (1,499 occurrences in 1,078 rows), advocate-name leakage (`Mr. Frederick`, `Mr. Fisher`), list-response failure mode in Oyez API, mid-utterance dashes (handled by tokenizer naturally), original-jurisdiction cases missing oral-argument coverage, Medellin docket duplication.
- [ ] **2.5 Exploration highlights (from B1–B6).** Per-Justice petitioner-vote rate range (50–80%); word-count distribution per Justice with Thomas (median 233) and KBJackson (median 1,205) named as outliers; per-class vocabulary differences (B1 finding: pre-stopwording top discriminative tokens were dominated by case-topic words, not stance markers); per-Justice vocabulary signatures detectable (B6 finding); length is not a confound (B5: Mann-Whitney p=0.255).
- [ ] **2.6 Identified transformation needs.** Custom calibrated stopword list (case-topic words removed, thematic legal vocabulary preserved); preprocessing for transcription artifacts; codebook verification for label semantics; multi-audio aggregation strategy. (Each addressed in Section 3.)

##### Section 3 — Data Preparation

*This phase focuses on constructing the dataset from raw data and cleaning and transforming data.* (Per Class 2 slides: select data subset, clean and transform, perform feature engineering and feature selection.)

- [ ] **3.1 Label derivation & codebook verification.** SCDB codebook semantics (`partyWinning`: 0/1/2; `majority`: 1=dissent, 2=majority — counterintuitive). Final formula: `voted_petitioner = (partyWinning == 1) == (majority == 2)`. Heien spot-check at Stop A verified the formula on a known 8-1 case (Sotomayor=1, others=0); all matched. Phase 3.5 re-verified on 10 random rows from the modeling table; 10/10 correct.
- [ ] **3.2 Multi-audio aggregation.** 13 cases have multiple oral-argument sessions (NFIB v. Sebelius has 4 — the ACA case). Justice utterances concatenated across sessions; `n_audio_sessions` stored as metadata.
- [ ] **3.3 Justice ID mapping.** Hand-built `justice_id_map.csv` with 16 rows. SCDB uses numeric IDs and short codes (`80180`, `HHBurton`); Oyez uses slugs (`john_g_roberts_jr`). All 16 slugs validated empirically at Checkpoint 1 (all returned non-zero coverage at credible ratios).
- [ ] **3.4 Cleanup decisions.** From the joined table (10,308 rows), dropped: 171 NaN-label rows (24 from `partyWinning == 2`, 127 from `majority NaN`, 20 unmatched); 17 original-jurisdiction cases (no cert grant, Oyez doesn't catalog them); 81 rows with word_count < 30 (truncated half-utterances with no signal). Final modeling table: 10,039 rows × 20 cols, 1,293 cases, 16 Justices.
- [ ] **3.5 Train/test split discipline.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=case_id`. Fold 0 = test, folds 1–4 = train. **All Justices for a given case stay in the same fold** — this is the no-leakage guarantee. Both modeling tracks consume splits via shared `src/modeling/splits.py`.
- [ ] **3.6 BoW track features.** Custom calibrated stopword list (424 terms: sklearn defaults + states + agencies + famous case names + court-procedural terms + advocate-name patterns). Three vectorizers: BoW (`CountVectorizer`), TF-IDF unigram, TF-IDF bigram. Vectorizers live inside `Pipeline` so vocabulary is built on train fold only.
- [ ] **3.7 Embeddings track features.** Pre-trained sentence-transformers, no fine-tuning. Two models compared: `all-MiniLM-L6-v2` (384-dim, ~80 MB, 1 min CPU encode) and `all-mpnet-base-v2` (768-dim, 11 min CPU encode). Embeddings cached to `data/processed/embeddings/`. No stopword list applied — sentence-transformers consume natural language.

##### Section 4 — Modeling

*This phase focuses on selecting and applying algorithms and techniques and calibrating parameters.* (Per Class 2 slides: select algorithms, build models, assess performance.)

- [ ] **4.1 Track 1: BoW pipeline.** All 9 combinations (3 vectorizers × 3 classifiers): LogisticRegression, LinearSVC, RandomForestClassifier — all with `class_weight='balanced'`. Baseline sweep table with per-fit timing.
- [ ] **4.2 BoW hyperparameter tuning (sequential strategy).** Stage 4A: joint GridSearchCV over linear models + vectorizer params (`C`, `penalty`, `min_df`, `max_df`, `ngram_range`). Stage 4B: RF with vectorizer fixed at Stage 4A's winner. Avoids the 1,600+ fit blowup that a full joint grid on RF would cause. Best BoW model: **LinearSVC + TF-IDF unigram, C=0.01, test ROC AUC 0.5323, 5-fold CV mean 0.540 ± [std]**.
- [ ] **4.3 Track 2: Sentence-embeddings pipeline.** Same three classifiers (with `SVC(kernel='rbf')` instead of `LinearSVC` — RBF is appropriate on dense vectors and was inappropriate on the 200K-dim sparse BoW). Baseline sweep across two embedding models × three classifiers (6 combos). Lightweight MiniLM (384-dim) edged MPNet (768-dim) by 0.003 — within noise but selected for the 10× encode-speed advantage.
- [ ] **4.4 Embeddings hyperparameter tuning.** GridSearchCV over LogReg, SVM-RBF, and RF on the winning MiniLM embeddings. Best embeddings model: **LogReg + MiniLM-L6-v2, C=100, test ROC AUC 0.5691, 5-fold CV mean 0.540 ± [std]**.
- [ ] **4.5 Compute discipline.** Per-fit timing logged to `reports/results/`. BoW Phase 4: 6.9 min total wall-clock (after `n_jobs=4` + Pipeline caching to avoid OOM). Embeddings Phase 4.5: ~32 min total (12 min encoding + 20 min sweep + tuning).

##### Section 5 — Model Evaluation

*This phase focuses on evaluating the model, testing it, and determining list of possible actions.* (Per Class 2 slides: evaluate against business criteria, test in the real application, review with stakeholders for next steps.)

- [ ] **5.1 Comparative summary table.** One row per track winner. Columns: representation, classifier, accuracy, balanced accuracy, precision, recall, F1, ROC AUC, 5-fold CV mean ± std, contested-cases AUC. Headline: BoW 0.5323 vs Embeddings 0.5691 (+3.7 pp).
- [ ] **5.2 Standard metrics suite, both tracks side-by-side.** Confusion matrix, ROC curve, PR curve, calibration curve. (LinearSVC wrapped with `CalibratedClassifierCV(method='sigmoid', cv=5)` only for the calibration curve.)
- [ ] **5.3 Per-Justice lift over individual baselines.** Bar chart with both tracks. BoW: only 2 of 15 Justices statistically distinguishable from chance (Kennedy, Alito). Embeddings: 4 of 15 (Thomas, Kavanaugh, Alito, Roberts).
- [ ] **5.4 The honesty triad — per-Justice contested-cases AUC, both tracks.** This is the strict test of true bench-reading signal vs author-identity-plus-priors recovery. **Result: embedding lift survives at +4 pp on contested cases (vs +5 pp unanimous), so the gain is real bench-questioning signal.** 9 of 15 Justices above 0.5 with BoW; 13 of 15 with Embeddings.
- [ ] **5.5 What did each track learn?** BoW: top features are thematic legal vocabulary (`officer`, `arrest`, `religious`, `jury`, `sentence`) — topic proxies, not stance markers. Embeddings: dense-vector dimensions are uninterpretable individually, so the proxy is highest- and lowest-predicted utterances surfaced as semantic neighborhoods. Mixed correctness at the extremes — model is not infallible.
- [ ] **5.6 Evaluation against business criteria (loop back to Section 1).** Section 1 set the success criterion as "ROC AUC meaningfully above the per-Justice majority-class baseline on contested cases." **Result on contested cases: BoW met this for 9/15 Justices; Embeddings for 13/15. The success criterion is partially met — broadly with embeddings, narrowly with BoW.** False-positive vs false-negative cost asymmetry: depends on whether the legal-tech consumer is preparing arguments (FN costlier) vs targeting amicus briefs (FP costlier). Threshold can be tuned per use case.
- [ ] **5.7 Honest interpretation pass.** What can we honestly claim? (1) Pre-trained sentence embeddings extract more vote-relevant signal than the standard TF-IDF/linear toolkit (+3.7 pp ROC AUC overall, +4 pp on contested cases). (2) The per-Justice gain is broadly distributed but concentrated in specific personalities (Thomas, Barrett, KBJackson, Kavanaugh). (3) Absolute level is modest — "lower bound on bench-reading from text alone." (4) The KBJackson flip (BoW 0.406 → Embeddings 0.643 globally; 0.405 → 0.643 contested) is the project's sharpest single finding: same Justice, same data, same labels, opposite conclusions about predictability depending solely on the representation. Stevens & Kennedy regressions reported honestly — not every Justice improves with embeddings.
- [ ] **5.8 Stakeholder review & next steps.** Reviewable findings, list of recommended actions for the next CRISP-DM iteration. Sets up Section 6.

##### Section 6 — Model Deployment

*This phase focuses on deploying the model and defining the monitoring and maintenance plan.* (Per Class 2 slides: develop deployment plan, ongoing support including operating model and monitoring, develop final reports and presentations.)

- [ ] **6.1 Deployment plan.** What shipping JusticeCast as a legal-tech product would look like. API endpoint that consumes a transcript, returns per-Justice vote probabilities. Pre-trained MiniLM embeddings at the core (lightweight: 80 MB, ~12 min CPU encoding for a corpus this size). Hybrid product extends the text model with structured case features (issue area, lower-court holding, Justice's prior voting record).
- [ ] **6.2 Operating model.** Who consumes this, when, how. Appellate litigators (pre-argument prep, week before argument). Amicus brief authors (post-argument, hours after). Legal-tech vendors (embedded in research workflow). Litigation press (post-argument forecasts).
- [ ] **6.3 Monitoring & re-training cadence.** Per-term re-training after each SCOTUS term ends (June). Monitor per-term performance drift — if AUC drops >5 pp term-over-term, audit the bench composition and case mix changes. Justice composition changes (retirements, appointments) require justice_id_map updates and per-Justice retraining; flag any case where a parsed Justice utterance fails to map to a known Justice.
- [ ] **6.4 Ongoing support.** Oyez API politeness (≤1 req/sec), cached transcripts, pinned dependencies in `requirements.txt`. Annual SCDB release pull. The codebook semantics may shift in major SCDB releases — re-verify the label derivation each release.
- [ ] **6.5 Methodological frontier (next CRISP-DM iteration).** Out of scope for this study, listed as recommended next directions: fine-tuned Legal-BERT on SCOTUS oral argument corpus, sequence-aware models on full transcripts (capture turn-taking, who interrupts whom), multimodal audio features (tone, pace, hesitation from the Oyez .mp3 files — the signal litigators actually pick up on that lexical features cannot capture), structured case feature integration.
- [ ] **6.6 Final reports & presentations (this notebook + deck + canvas).** Reproducibility instructions: cached embeddings live in `data/processed/embeddings/`, notebook loads cached arrays and does NOT re-encode. Fresh-clone reproduction requires running `src/compute_embeddings.py` once (~12 min CPU). pytest suite: 90 tests green.

**Discipline notes for the build:**

- Each top-level section opens with the one-sentence framing line in italics (so the grader can confirm CRISP-DM mapping at a glance)
- Each subsection has prose around the cells, not bare code dumps
- Charts are fresh-rendered with titles, axis labels, legends, color-blind-safe palettes
- All printouts formatted (no raw `print(df)` dumps)
- No leftover working-notebook noise (debug cells, `# TODO`, commented-out code)
- The notebook reads as a polished report, not a transcript of working sessions

#### 6.2 — Machine Learning Canvas v0.4

Fill the 12 quadrants per the BAX 453 template. **Map each quadrant explicitly to the corresponding CRISP-DM phase** so the framework is visible in the canvas too:

- [ ] **Goal box** (CRISP-DM Business Understanding): "Quantify how much of bench-reading signal lives in lexical features vs semantic representations on SCOTUS oral arguments."
- [ ] **Decisions** (Business Understanding): per-Justice vote forecasts inform pre-argument prep targeting, amicus brief targeting, historical bench-reading benchmarks
- [ ] **ML Task** (Modeling): binary stance classification (with petitioner / against petitioner)
- [ ] **Value Propositions** (Business Understanding): "Empirical lower bound on text-only bench-reading; methodological evidence that pre-trained semantics outperform tuned bag-of-words by ~4 pp on the strict contested-cases test."
- [ ] **Data Sources** (Data Understanding): SCDB + Oyez API
- [ ] **Collecting Data** (Data Understanding): per-term refresh after each SCOTUS term
- [ ] **Features** (Data Preparation): TF-IDF / n-grams (BoW track) + sentence-transformer embeddings (embeddings track)
- [ ] **Building Models** (Modeling): annual retraining after each term ends
- [ ] **Offline Evaluation** (Model Evaluation): ROC AUC, per-Justice contested-cases AUC, 5-fold CV mean ± std, confusion matrix
- [ ] **Making Predictions** (Modeling): per-Justice per-case forecast within hours of oral argument
- [ ] **Live Evaluation and Monitoring** (Model Deployment): track per-term performance drift, flag ≥5 pp drops
- [ ] Export as `reports/ml_canvas.pdf`

#### 6.3 — README polish

- [ ] One-line description: *"Comparative study of text representations for stance classification on SCOTUS oral arguments — bag-of-words vs sentence-transformer embeddings, structured around CRISP-DM."*
- [ ] Headline finding (one paragraph): the +3.7 pp gap, the contested-cases survival, the KBJackson flip
- [ ] Method overview: explicit CRISP-DM phase list with brief one-line description of each
- [ ] How to reproduce from a fresh clone (including the `compute_embeddings.py` step)
- [ ] Team credits
- [ ] Repo layout pointer

#### 6.4 — Final reproducibility check

- [ ] `Restart & Run All` on `JusticeCast_Final.ipynb` on a fresh kernel — must succeed end-to-end with cached embeddings
- [ ] `pytest` — all 90 tests green
- [ ] Commit clean

- **CHECKPOINT 6:** Final notebook + canvas PDF + polished README ready. **Stop and wait** for CAI review before pitch deck.

### Phase 7: Pitch Deck (Part A) — CRISP-DM-aligned Comparative Methodology Narrative

10-slide deck (8–12 range), exported as PDF. Deck flow loosely follows CRISP-DM (without naming the phases in slide headers — the deck is a pitch, not a process document — but each phase is implicitly covered).

- [ ] **Slide 1 — Title.** JusticeCast: A Comparative Study of Text Representations for SCOTUS Vote Prediction. Team names, date.
- [ ] **Slide 2 — The Hypothesis.** *(Business Understanding)* Open with **Citizens United v. FEC** (the rescued landmark case) — 5-4 ideological split, recognizable to any audience. Frame the question: *"Litigators have read the bench by gut for 200 years. Can a model do it from the words alone?"* Mention Lex Machina, Bloomberg Law, Westlaw Edge as firms monetizing this intuition.
- [ ] **Slide 3 — The Data.** *(Data Understanding)* 10,039 Justice-utterance blobs across 1,293 cases (2005–2024). SCDB (Justice-Centered file) + Oyez API. No leakage (StratifiedGroupKFold by `case_id`). Custom calibrated stopword list preserves thematic legal vocabulary while filtering case-identity terms. 2,940 API calls, 54-min bulk fetch.
- [ ] **Slide 4 — Two Modeling Tracks.** *(Data Preparation + Modeling)* Pipeline diagram. Track 1: BoW (rubric-required) — 3 vectorizers × 3 classifiers + GridSearchCV. Track 2: Embeddings (methodologically-appropriate alternative) — pre-trained sentence-transformers + same 3 classifiers + GridSearchCV. Both tracks use identical test fold.
- [ ] **Slide 5 — ML Canvas summary.** Mini canvas with 4-6 most important boxes filled.
- [ ] **Slide 6 — BoW Results.** *(Modeling + Evaluation)* Best ROC AUC 0.5323 after tuning. Only 2 of 15 Justices statistically distinguishable from chance. Top features are thematic legal vocabulary, *not* stance markers. The lexical representation has a real ceiling.
- [ ] **Slide 7 — Embeddings Results.** *(Modeling + Evaluation)* Best ROC AUC 0.5691 after tuning. **Lightweight MiniLM (384-dim, 80MB, no fine-tuning) beats tuned 200K-feature TF-IDF + LinearSVC by 3.7 pp.** Per-Justice lift bar chart with both tracks.
- [ ] **Slide 8 — The Comparative Finding (the headline slide).** *(Evaluation)* Side-by-side contested-cases AUC: BoW 9/15 above 0.5; Embeddings 13/15 above 0.5. The lift survives the strict test (+4 pp contested vs +5 pp unanimous), so the embedding gain is real bench-questioning signal, not author-identity-plus-priors recovery. **Pivot: KBJackson centerpiece.** Same Justice, same data — BoW 0.405 → Embeddings 0.643 (+0.238 lift). Stevens & Kennedy regress, plausibly because their votes tracked thematic case content closely (BoW exploited that, embeddings collapsed it in semantic space). Mixed evidence, honestly reported.
- [ ] **Slide 9 — What This Means for Legal-Tech.** *(Deployment recommendations)* Don't sell a TF-IDF question-classifier — the lexical representation is insufficient. Pre-trained semantic embeddings are necessary baseline, not optional upgrade. Marginal infrastructure cost: ~12 min CPU encoding once. The remaining signal — tone, sequence, interruption patterns, audio — is the next product frontier. 2-3 concrete recommendations.
- [ ] **Slide 10 — Methodological Recommendations & Honest Caveats.** *(Deployment + Next Iteration)* Absolute AUC 0.569 is modest — we're reporting a *lower bound* on bench-reading from text alone. Out-of-scope frontiers: fine-tuned Legal-BERT, sequence-aware transformers, multimodal audio features, structured case features.
- [ ] **Slide 11 — Outro / Q&A.** Bookend by returning to Citizens United. Close with: *"Litigators have read the bench by gut for 200 years. We tested two computational approaches — the standard one and the methodologically-appropriate one. The gap between them tells us where the real signal lives. That's the actionable finding."*
- [ ] Storytelling discipline: open with Citizens United → KBJackson centerpiece on Slide 8 → Citizens United bookend close.
- [ ] Export `reports/JusticeCast_Pitch.pdf`

- **CHECKPOINT 7:** Both deliverables (`JusticeCast_Final.ipynb`, `reports/ml_canvas.pdf`, `reports/JusticeCast_Pitch.pdf`) ready for Canvas submission. **Stop and wait** for final CAI review before submission.

## Definition of Done

- `JusticeCast_Final.ipynb` reads as a polished top-to-bottom narrative organized around **CRISP-DM's six phases as primary section headers** (Business Understanding → Data Understanding → Data Preparation → Modeling → Model Evaluation → Model Deployment), executes `Restart & Run All` clean on fresh kernel with cached embeddings
- All 9 BoW combinations + 6 embeddings combinations evaluated and logged with per-fit timing
- GridSearchCV applied to both tracks via the sequential strategy
- 5-fold CV mean ± std AUC reported for both track winners
- Comparative summary table showing both tracks side-by-side
- Per-Justice performance reported as lift over each Justice's individual baseline, both tracks
- Per-Justice contested-cases ROC AUC reported as a primary metric for both tracks (the embedding lift survives the strict test — confirmed)
- Top features for BoW (interpretable) and exemplar utterances for embeddings (semantic neighborhoods) documented
- Honest interpretation pass comparing what each track learned
- Reframed business interpretation paragraph addressing the comparative finding
- **Machine Learning Canvas v0.4 explicitly maps each box to its CRISP-DM phase** and exported as PDF
- Pitch deck reframed as comparative methodology study with Citizens United → KBJackson → Citizens United narrative arc, slide flow loosely follows CRISP-DM
- README documents project as comparative study with explicit CRISP-DM method overview
- pytest suite runs green (90 tests)
- All artifacts committed with clean history
- Proposal submitted to professor (✅ sent 4/26)
- Three deliverables submitted to Canvas by **5/28**:
  - `JusticeCast_Final.ipynb` (Part B notebook)
  - `reports/ml_canvas.pdf` (Part B canvas)
  - `reports/JusticeCast_Pitch.pdf` (Part A deck)

## Constraints

- **Hard deadlines:** proposal sent 4/26 (✅); deliverables 5/28
- **Oyez API:** ≤ 1 req/sec
- **SCDB:** Latin-1 encoded, release 2025_01
- **Label derivation: locked.** `(partyWinning == 1) == (majority == 2)`
- **Pre-trained embeddings only, no fine-tuning** — out of scope
- **Identical splits across tracks: non-negotiable.**
- **CRISP-DM is the visible structure of the submission notebook** (Non-Negotiable #16)
- **Framing:** comparative methodology study with empirically-grounded business recommendations
- **Team size:** 6

## Current Instruction

**Status:** Phase 5 ✅ complete. CRISP-DM-aligned restructuring of the Phase 6 notebook spec is now locked in (Non-Negotiable #16). CC is approved to execute Phase 6.

**Resolutions from Checkpoint 5 + post-Checkpoint review:**

- **Notebook structure changed.** The 12-section spec from v11 collapses into six top-level sections matching CRISP-DM exactly, with the original 12 sections folded as subsections inside. Same content, framework now visible at the top level.
- **ML Canvas wording explicitly maps boxes to CRISP-DM phases.**
- **README adds CRISP-DM method overview** so the framework is named in the project's front door.
- **Pitch deck flow loosely tracks CRISP-DM** with phase labels in italic captions per slide (not in headers).
- All previous Checkpoint 5 resolutions still hold: contested-cases test result is the project's central methodological win; KBJackson centerpiece locked; lightweight-MiniLM-beats-tuned-BoW framing locked; Citizens United arc locked; Stevens & Kennedy regressions reported honestly.

**What to produce this turn (Phase 6):**

1. **Build `JusticeCast_Final.ipynb`** as a fresh polished six-section narrative organized around CRISP-DM phases. Each top-level section opens with a one-sentence framing line in italics. Subsections per the spec above. Pull figures and tables from working notebooks but re-render cleanly.
2. **Fill the Machine Learning Canvas v0.4** with each box mapped to its CRISP-DM phase. Export `reports/ml_canvas.pdf`.
3. **Polish `README.md`** with the comparative-study one-liner, headline finding paragraph, explicit CRISP-DM method overview, reproduction instructions including cached-embeddings note.
4. **Run `Restart & Run All`** on `JusticeCast_Final.ipynb` from a fresh kernel — must succeed with cached embeddings, not re-encode.
5. **Run `pytest`** — all 90 tests green.

**What to stop and report back on (Checkpoint 6):**

- `JusticeCast_Final.ipynb` rendered in CRISP-DM six-section structure, `Restart & Run All` clean
- `reports/ml_canvas.pdf` generated with explicit CRISP-DM box mapping
- Polished `README.md` with CRISP-DM method overview
- Confirmation pytest is green
- Any narrative sections where you had to make a judgment call on tone, emphasis, or CRISP-DM mapping — flag for CAI review

**Pushback welcome on:**

- Subsection ordering within any of the six CRISP-DM phases if a different order tells the story better
- Any chart or table from the working notebooks that doesn't translate cleanly to the polished narrative — propose a substitute or omission
- ML Canvas wording for any of the 12 boxes — if a box doesn't fit the comparative-study framing, propose a reframe rather than forcing it
- The README headline paragraph — if a different framing of the +3.7 pp finding lands cleaner, propose it
- The italic phase-mapping captions on the deck slides — if they read as cluttered or distracting, drop them and rely on the narrative flow alone (the ML Canvas + notebook already make CRISP-DM explicit)
