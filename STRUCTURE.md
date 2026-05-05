# Repository Structure

A one-page reference. Every meaningful file in the repo with a one-line description of what it does. For the project narrative + reproduction instructions, see [`README.md`](README.md).

## Top-level

| Path | Purpose |
| --- | --- |
| `README.md` | Project orientation document — what JusticeCast is, the findings, method, reproduction paths |
| `STRUCTURE.md` | This file — file-by-file repo reference |
| `requirements.in` | Direct dependencies (unpinned), human-edited |
| `requirements.txt` | Pinned snapshot via `pip freeze` (Python 3.14) |
| `.gitignore` | Excludes `data/raw/`, `data/processed/*` (with exception for `justice_id_map.csv`), embedding caches, internal collaboration artifacts, OS/IDE junk |

## `notebooks/`

| Path | Purpose |
| --- | --- |
| `notebooks/JusticeCast_Final.ipynb` | **SUBMISSION NOTEBOOK.** CRISP-DM-structured (six top-level sections matching the six CRISP-DM phases). 83 cells. Reads pre-computed CSVs from `reports/results/` and cached embeddings; Restart-and-Run-All completes in seconds. |
| `notebooks/01_eda.ipynb` | Working notebook for Phase 2 EDA + the B1–B6 expansion (per-class log-odds, per-Justice baselines, sample-text artifact audit, vocab stats, length-vs-label, per-Justice signature) |

## Project workflow phases

The repo uses two parallel "phase" systems. **CRISP-DM phases (6)** are the methodological framework — they answer *"what kind of work is this?"* and are documented in [`README.md`](README.md) §4. **Internal project phases (1–8)** are the build-sequence checkpoints used in filenames (`phase4_gridsearch.py`, `phase5_*`, etc.) — they answer *"where in the build are we?"* Multiple internal phases can map to a single CRISP-DM phase (Modeling spans Phases 3, 4, and 4.5).

| Internal phase | What it does | CRISP-DM phase |
| --- | --- | --- |
| Phase 1 | Data acquisition + initial join (SCDB + Oyez bulk fetch → joined parquet) | Data Understanding |
| Phase 2 | Cleanup → modeling table (drop NaN labels, original-jurisdiction, low-word-count rows) | Data Preparation |
| Phase 3 | BoW baseline sweep — 9 (vectorizer × classifier) combinations | Modeling |
| Phase 4 | BoW GridSearchCV — joint vectorizer + classifier tuning | Modeling |
| Phase 4.5 | Embeddings track — MiniLM/MPNet baselines + GridSearchCV | Modeling |
| Phase 5 | Honesty-triad evaluation — per-Justice contested-cases AUC | Model Evaluation |
| Phase 6 | Final notebook + ML Canvas synthesis | Business Understanding + Deployment |
| Phase 7 | Pitch-deck assembly (chart PNGs + asset bundle + .ppt) | Model Deployment |
| Phase 8 | Handoff polish (README, STRUCTURE, repro verification) | Model Deployment |

## `src/`

Source modules. Every file has a polished header docstring stating what it does, what it produces, and which CRISP-DM phase it serves.

### Data acquisition + dataset construction (Data Understanding + Data Preparation)

| Path | Purpose |
| --- | --- |
| `src/fetch_scdb.py` | Download + Latin-1 read of the SCDB Justice-Centered file (release 2025_01). Idempotent. |
| `src/fetch_oyez.py` | Two-step Oyez fetcher with global ≤ 1 req/sec rate limit, `tenacity` retries, on-disk cache at both layers. Defines `CaseNotFound` for non-standard dockets. |
| `src/run_bulk_fetch.py` | Bulk-fetch driver for Phase 1 Stop B — iterates all `(term, docket)` pairs in the 2005–2024 window through `fetch_oyez.fetch_case_full`. |
| `src/rescue_failed_dockets.py` | Stop C rescue pass — for each standard-format failure from the bulk-fetch log, tries `term ± 1` (Oyez sometimes files re-argued cases under the prior term). Recovered Citizens United and Kiobel. |
| `src/build_dataset.py` | Parses cached Oyez transcripts, filters to Justice utterances, multi-audio aggregation, joins SCDB via `justice_id_map.csv`, derives `voted_petitioner` label + `unanimous` flag, writes `data/processed/justice_case_rows.parquet`. Includes the codebook-cited `derive_voted_petitioner`. |
| `src/build_modeling_table.py` | Phase 2 cleanup — drops NaN-label rows, original-jurisdiction cases, low-word-count rows; applies `preprocess_text()`; writes `data/processed/modeling_table.parquet` + audit CSV. |
| `src/text_clean.py` | Defines `preprocess_text()` (strips bracketed transcription artifacts), `vectorizer_preprocessor()` (advocate-name regex stripping for vectorizers), and `STOPWORDS_FOR_VECTORIZER` (424-term list = sklearn defaults + custom domain-identifying terms). |
| `src/phase1_data_audit.py` | Generates `reports/phase1_data_audit.md` from cached SCDB + bulk-fetch log + parquet — Phase 1 detailed report. |

### Modeling (Modeling phase)

| Path | Purpose |
| --- | --- |
| `src/modeling/__init__.py` | Package marker |
| `src/modeling/splits.py` | **Canonical fold-0 train/test split, shared by both modeling tracks** (Non-Negotiable #15). `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, `groups=case_id`. Includes `assert_no_case_leakage()` guard. |
| `src/phase3_baseline_sweep.py` | BoW 9-combo baseline sweep (3 vectorizers × 3 classifiers) on the canonical fold-0 split. Logs per-fit timing. |
| `src/phase4_gridsearch.py` | BoW GridSearchCV — sequential strategy. Stage 4A: two `GridSearchCV` runs (LogReg + LinearSVC) sharing the same vectorizer parameter grid. Stage 4B: RF with vectorizer fixed at Stage 4A's winner. `n_jobs=4` + `Pipeline(memory=...)` caching for memory safety. Reports per-Justice AUC with bootstrap CIs. |
| `src/compute_embeddings.py` | Encodes `text` with `all-MiniLM-L6-v2` (384-dim) and `all-mpnet-base-v2` (768-dim). CPU-only torch backend. Caches `.npy` files + row-index parquet. Idempotent. |
| `src/phase45_baseline_sweep.py` | Embeddings 6-combo baseline sweep (2 embedding models × 3 classifiers — including `SVC(kernel='rbf')` as the methodologically-appropriate dense-vector classifier). Same fold-0 test rows as the BoW track. |
| `src/phase45_gridsearch.py` | Embeddings GridSearchCV on the better-performing embedding model from the baseline sweep. LogReg + SVM-RBF + RF grids. Same memory discipline as Phase 4. Reports per-Justice AUC with bootstrap CIs. |

### Evaluation + reporting (Model Evaluation + reporting artifacts)

| Path | Purpose |
| --- | --- |
| `src/phase5_evaluation.py` | Refits both Phase 4 (BoW) and Phase 4.5 (Embeddings) winners on full canonical training fold; computes the **honesty triad** (per-Justice ROC AUC under three slicings: global, unanimous-only, contested-only). Writes 9 result CSVs spanning summary metrics, ROC/PR/calibration data, confusion matrices, per-Justice lift, test predictions, and extreme-score utterances. Per-review fixes: per-Justice baseline computed train-only (no test leakage); CalibratedClassifierCV uses StratifiedGroupKFold (case-grouped). |
| `src/phase5_delong.py` | Paired AUC hypothesis test on the BoW-vs-Embeddings gap. Implements DeLong's two-sided test (Sun & Xu 2014 vectorized form) plus a paired bootstrap CI. Reports Z, p-value, 95% CI on `reports/results/phase5_delong_test.csv`. Added in peer-review pass. |
| `src/phase5_kfold_eval.py` | 5-fold cross-validated test AUC sweep — refits each winner on the 4-fold training portion of each StratifiedGroupKFold fold and scores on the held-out fold. Reports per-fold AUC for both tracks plus the paired t-test on per-fold diffs. Added in peer-review pass to address single-fold-luck concern. Supports `--n-reps N` for repeated CV (load-bearing 10×5 repeated-CV result writes to `phase5_repeated_cv.csv`). |
| `src/build_comparative_summary.py` | Builds `comparative_summary.csv` (top-line, one row per track winner) and `comparative_per_justice.csv` (long form, BoW vs Embeddings per Justice). Side-by-side artifacts for Phase 5 narrative. |
| `src/build_ml_canvas.py` | Renders `reports/ml_canvas.pdf` — the 12-box Machine Learning Canvas v0.4 with each box CRISP-DM-tagged. Matplotlib + PdfPages. |
| `src/build_deck_charts.py` | Phase 7 — renders 8 deck-quality chart PNGs into `reports/deck_assets/` with the locked deck theme (deep navy + warm gold + cream). Source of every chart asset in the pitch-deck bundle. |

## `tests/`

9 pytest files, 104 tests total. Run with `pytest` or `pytest -q`.

| Path | What it tests |
| --- | --- |
| `tests/test_fetchers.py` | SCDB schema + Latin-1 encoding; Oyez Step 1/Step 2 shapes on Heien; rate-limiter timing; cache idempotency |
| `tests/test_builders.py` | Label-derivation truth table; parser filters Justices vs advocates; **Heien spot-check (Sotomayor=1, others=0)**; Justice-mapping coverage; multi-audio aggregation |
| `tests/test_modeling_table.py` | Original-jurisdiction docket detection; cleanup drops NaN-label / low-word-count / original-jurisdiction rows; label dtypes are `int8`; audit CSV is monotonically decreasing |
| `tests/test_text_clean.py` | Bracket-annotation stripping; advocate-name regex (preserves "Justice Roberts"); idempotence; sklearn stopword inclusion + custom additions; thematic-vocab non-overstrip |
| `tests/test_splits.py` | Split parameters locked (n_splits=5, random_state=42, fold 0 = test); deterministic across calls; no caseId leakage; stratification preserves label balance; 20% test fraction |
| `tests/test_phase4_gridsearch.py` | Pipeline construction (LogReg + LinearSVC + RF); grid sizes (12 vec configs × clf options); CV result CSV row counts (120 + 48 + 27); no anti-predictive combos |
| `tests/test_compute_embeddings.py` | MODELS dict contents; CPU-only device; `_matches_cache()` shape check; cache shape (10039, 384) for MiniLM + (10039, 768) for MPNet; row-index aligned to modeling_table |
| `tests/test_phase45.py` | Baseline-sweep classifier set (LogReg + SVM-RBF + RF, all class_weight='balanced'); GridSearchCV grid sizes (10 + 9 + 27); n_jobs=4 settings; result CSV shape and content |
| `tests/test_phase5_evaluation.py` | Best-params constants match Phase 4 / 4.5 winners; honesty triad has all three slicings per track; global slice matches existing per-Justice CSVs; contested lift > 0.02 (guard against regression); extremes table has 40 rows |
| `tests/test_deck_charts.py` | `ALL_PNGS` matches §7.1 spec (8 files); theme color constants locked; every PNG present in `reports/deck_assets/` and ≥ 30 KB; every spec markdown present and ≥ 1 KB |

## `data/`

| Path | Purpose |
| --- | --- |
| `data/raw/` | gitignored — SCDB CSV + zipped source + Oyez JSON cache (~377 MB after full bulk fetch) |
| `data/raw/scdb_codebook/` | Cached SCDB codebook HTML pages for the 5 label-relevant fields (`partyWinning`, `majority`, `vote`, `direction`, `caseDisposition`) |
| `data/raw/oyez/cases/` | Cached case metadata JSON, one per `(term, docket)` |
| `data/raw/oyez/transcripts/` | Cached transcript audio JSON, one per `audio_id` |
| `data/processed/justice_id_map.csv` | **TRACKED** — hand-built SCDB ↔ Oyez Justice key (16 rows for the 2005–2024 window). Canonical input. |
| `data/processed/justice_case_rows.parquet` | gitignored — joined parquet, post-build_dataset, pre-cleanup |
| `data/processed/modeling_table.parquet` | gitignored — final modeling table, 10,039 rows × 20 cols |
| `data/processed/embeddings/minilm.npy` | gitignored — cached MiniLM embeddings, shape (10039, 384), float32, ~15 MB |
| `data/processed/embeddings/mpnet.npy` | gitignored — cached MPNet embeddings, shape (10039, 768), float32, ~30 MB |
| `data/processed/embeddings/row_index.parquet` | gitignored — preserves modeling-table row order so positional split indices align |

## `reports/`

### Course deliverables

| Path | Purpose |
| --- | --- |
| `reports/proposal.md` | 1-page proposal sent to Professor Sharad Gupta on 4/26 (well ahead of 5/7 deadline) |
| `reports/ml_canvas.pdf` | Machine Learning Canvas v0.4 with each box CRISP-DM-tagged. Generated by `src/build_ml_canvas.py`. |
| `reports/JusticeCast_Pitch.ppt` | Phase 7 pitch deck (PowerPoint, 11 slides; team-editable). Assembled externally by the PowerPoint Claude extension from the asset bundle in `reports/deck_assets/`. |
| `reports/phase1_data_audit.md` | Auto-generated Phase 1 detailed report (pipeline stages, Justice coverage, label distribution, multi-audio cases). Produced by `src/phase1_data_audit.py`. |

### Phase 7 deck-asset bundle (`reports/deck_assets/`)

| Path | Purpose |
| --- | --- |
| `reports/deck_assets/theme_spec.md` | Locked visual identity — palette (`#1A2E47` navy + `#C9A961` gold + `#FAF7F2` cream), typography (serif/sans pairing), layout density rules, standard slide chrome |
| `reports/deck_assets/slide_content_spec.md` | Per-slide content for all 11 slides — layout type, header bar label, title, subtitle, visual asset, body content, takeaway |
| `reports/deck_assets/headline_numbers.md` | Every numeric claim that appears in the deck, with pointer to the source result CSV row |
| `reports/deck_assets/prompt_for_powerpoint_extension.md` | Drop-in prompt the user pastes into the PowerPoint Claude extension to assemble the deck (with the 11 attachments listed) |
| `reports/deck_assets/chart_bow_vs_embeddings_3slice.png` | Slide 8 headline — three grouped bar pairs (Global, Unanimous, Contested), BoW vs Embeddings |
| `reports/deck_assets/chart_per_justice_lift.png` | Per-Justice lift bar chart, both tracks side-by-side, KBJackson/Thomas highlighted |
| `reports/deck_assets/chart_kbjackson_flip.png` | Slide 8 KBJackson spotlight — single-Justice contested AUC, +0.238 in big gold |
| `reports/deck_assets/chart_bow_baselines.png` | Slide 6 — 9-combo BoW baseline sweep |
| `reports/deck_assets/chart_embeddings_baselines.png` | Slide 7 — 6-combo embeddings baseline sweep with BoW-baseline-best reference line |
| `reports/deck_assets/chart_data_pipeline_funnel.png` | Slide 3 — case-level + row-level data attrition funnel |
| `reports/deck_assets/data_flow_diagram.png` | Slide 4 — data flow diagram (SCDB ↔ Oyez ↔ joined parquet ↔ modeling table) |
| `reports/deck_assets/ml_canvas_summary.png` | Slide 5 — ML Canvas re-rendered as PNG for embedding in the deck |

### Result CSVs (`reports/results/`)

27+ CSV files spanning Phases 1–5. Notebook reads from these at render time; we don't re-run modeling sweeps to re-knit the notebook (Non-Negotiable #7).

| Path pattern | Purpose |
| --- | --- |
| `reports/results/baseline_results.csv` | Phase 3 BoW 9-combo baseline metrics |
| `reports/results/per_justice_lift.csv` | Phase 3 per-Justice lift over baseline |
| `reports/results/top_features_best_linear.csv` | Phase 3 top features for the best linear combo |
| `reports/results/bulk_fetch_log.csv` | Phase 1 per-case bulk-fetch outcome (term, docket, success, n_audio_sessions, error) |
| `reports/results/rescue_log.csv` | Stop C per-case rescue outcome |
| `reports/results/modeling_table_audit.csv` | Phase 2 cleanup audit — per-stage row counts |
| `reports/results/gridsearch_results.csv` | Phase 4 BoW GridSearchCV all-fits log (195 rows = 120 LogReg + 48 SVM + 27 RF) |
| `reports/results/phase4_test_eval.csv` | Phase 4 best-of-each on the test set (3 rows) |
| `reports/results/phase4_per_justice_auc.csv` | Phase 4 per-Justice AUC with bootstrap CIs |
| `reports/results/phase4_top_features.csv` | Phase 4 top 30 ± features for the BoW winner |
| `reports/results/embedding_baseline_results.csv` | Phase 4.5 baseline sweep (6 combos) |
| `reports/results/embedding_baseline_per_justice.csv` | Phase 4.5 per-Justice baseline metrics |
| `reports/results/embedding_gridsearch_results.csv` | Phase 4.5 GridSearchCV all-fits log (46 rows) |
| `reports/results/phase45_test_eval.csv` | Phase 4.5 best-of-each on the test set |
| `reports/results/phase45_per_justice_auc.csv` | Phase 4.5 per-Justice AUC with bootstrap CIs |
| `reports/results/phase45_top_features.csv` | Phase 4.5 top features (embedding-dim coefficients) |
| `reports/results/comparative_summary.csv` | Top-line BoW vs Embeddings winners |
| `reports/results/comparative_per_justice.csv` | Per-Justice BoW vs Embeddings, long form |
| `reports/results/phase5_summary_metrics.csv` | Phase 5 standard metrics suite (both tracks) |
| `reports/results/phase5_honesty_triad.csv` | Per-Justice ROC AUC under all three slicings (global / unanimous / contested) for both tracks |
| `reports/results/phase5_per_justice_lift.csv` | Phase 5 per-Justice lift over individual baselines, both tracks |
| `reports/results/phase5_roc_curve_data.csv` | ROC curve points for plotting (both tracks) |
| `reports/results/phase5_pr_curve_data.csv` | PR curve points for plotting (both tracks) |
| `reports/results/phase5_calibration_data.csv` | Calibration curve points (both tracks; BoW sigmoid-calibrated for the curve only) |
| `reports/results/phase5_confusion_matrices.csv` | 2x2 confusion matrix per track |
| `reports/results/phase5_test_predictions.csv` | One row per test utterance with both tracks' scores + predictions |
| `reports/results/phase5_extreme_utterances.csv` | Top 20 highest-predicted + 20 lowest-predicted by `emb_proba`, with case context |
| `reports/results/phase5_delong_test.csv` | DeLong's paired AUC test + paired bootstrap CI for the BoW-vs-Embeddings gap on fold 0. Single row. |
| `reports/results/phase5_kfold_evaluation.csv` | Per-fold AUC for both tracks across all 5 StratifiedGroupKFold folds (5 rows). |
| `reports/results/phase5_repeated_cv.csv` | 10×5 repeated CV — 50 fold-realizations across both tracks, used to power the load-bearing paired t-test on the AUC lift (n=50). |

### Other (`reports/figures/`)

| Path | Purpose |
| --- | --- |
| `reports/figures/justicecast_data_flow.png` | Source PNG for Slide 4 of the deck (copied to `reports/deck_assets/data_flow_diagram.png` by `src/build_deck_charts.py`) |
