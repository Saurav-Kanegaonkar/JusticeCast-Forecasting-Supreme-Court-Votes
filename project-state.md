# Project State: JusticeCast
Last updated: 2026-04-26 by CC at Checkpoint 0

## Project Context

- **What**: A binary text-classification system that predicts how a Supreme
  Court Justice will vote in a given case (with petitioner vs. with respondent)
  using only the text of that Justice's questions during oral argument.
- **Why**: A graded course project (Part A = 15 pt pitch deck, Part B = 20 pt
  reproducible Jupyter notebook; total 35 pt) framed as a legal-tech product
  for appellate litigators, amicus brief authors, and legal-tech platforms
  (Lex Machina, Bloomberg Law, Westlaw Edge, SCOTUSblog). Team of 6.
- **Type**: Data science (NLP / classical ML). Framed under Option 1
  (custom topic) as **stance classification**, not Option 2 (sentiment).
- **Tech Stack**: Python 3.14, pandas, numpy, scikit-learn ≥ 1.6, requests,
  beautifulsoup4, nltk, joblib, matplotlib, seaborn, JupyterLab, pytest.
  All deps pinned in `requirements.txt`.
- **Repo Structure** (per `cai-plan.md`):
  - `data/raw/` — SCDB CSV + cached Oyez JSONs (gitignored)
  - `data/processed/` — joined parquet tables (gitignored)
  - `src/` — `fetch_scdb.py`, `fetch_oyez.py`, `build_dataset.py`, `text_clean.py`
  - `notebooks/` — `01_eda.ipynb`, `02_modeling.ipynb`, `JusticeCast_Final.ipynb`
  - `reports/` — `proposal.md`, `ml_canvas.pdf`, `JusticeCast_Pitch.pdf`,
    `results/` (per-experiment CSVs)
  - `tests/` — `test_fetchers.py`, `test_builders.py`
  - Root: `requirements.txt`, `requirements.in`, `README.md`, `CLAUDE.md`,
    `cai-plan.md`, `project-state.md`, `.gitignore`

## What Exists (ground truth — Checkpoint 0)

- Git repo initialized (branch `main`).
- `.gitignore` excludes `.venv/`, `data/raw/`, `data/processed/`, `.env*`,
  `.ipynb_checkpoints/`, IDE/OS junk, `*.joblib`, `*.pkl`.
- Directory scaffold present (empty `data/raw`, `data/processed`, `src/`,
  `notebooks/`, `reports/results/`, `tests/`). `src/__init__.py` and
  `tests/__init__.py` placeholders created.
- `.venv/` created from system Python 3.14.3; deps installed from
  `requirements.in`; pinned snapshot in `requirements.txt`.
- **SCDB downloaded once** to `data/raw/SCDB_2025_01_justiceCentered_Citation.csv`
  (28 MB, 83,644 rows, 61 columns, Latin-1 encoded). Source URL recorded in
  `CLAUDE.md`. Sample row and column list verified.
- **Oyez API verified** with smoke-test on Heien v. North Carolina
  (`cases/2014/13-1314`). Confirmed transcript structure lives at
  `case_media/oral_argument_audio/{audio_id}` (NOT at the case endpoint —
  see Key Decisions below).
- `CLAUDE.md` written (CC working notes — env, data sources, conventions,
  open questions).
- `reports/proposal.md` drafted (1-page proposal for the professor).
- No source modules, notebooks, or tests written yet.
- No GitHub remote attached yet (will be created at Checkpoint 0 commit).

## Key Decisions Made

- **Splitting primitive: `StratifiedGroupKFold`**, fold 0 = test, folds 1–4 =
  train, `groups=case_id`. `train_test_split(stratify=y)` is forbidden for
  the primary split because it ignores groups and would leak cases across
  train/test (CC pushback, accepted into Non-Negotiable #1, Phase 0).
- **`LinearSVC` ROC AUC via `decision_function`**; `CalibratedClassifierCV`
  used only for the Phase 5 calibration curve. Avoids unnecessary calibration
  cross-validation overhead during the baseline sweep (CC pushback, accepted
  into Phases 3 + 5, Phase 0).
- **Unanimous cases are KEPT**, flagged via a `unanimous` metadata column
  (derived from SCDB `minVotes == 0`). The model never sees unanimity as a
  feature, but Phase 5 reports per-Justice metrics split by unanimity as a
  sensitivity analysis (CC pushback, accepted into Phases 1/2/5, Phase 0).
- **Sequential GridSearchCV (Phase 4 rewritten)**: Stage 4A jointly tunes the
  two linear models (LogReg, SVM) with vectorizer hyperparams; Stage 4B fixes
  the best vectorizer config from 4A and tunes RF only. Avoids a 1,620+ fit
  blowup that the original joint-grid plan would have triggered for RF
  (CC pushback, accepted into Phase 4 rewrite, Phase 0).
- **Stage 4A is two `GridSearchCV` runs sharing the vectorizer grid**, not
  literally one — sklearn `GridSearchCV` operates on a single estimator.
  CC clarification noted in `cai-plan.md` resolutions; plan wording to be
  tightened next cycle (Phase 0).
- **Oyez fetcher is a 2-step pull** (case JSON → case_media JSON). The
  original plan said "Oyez returns full transcripts at the case endpoint" —
  empirical check shows transcript turns live at the linked
  `case_media/oral_argument_audio/{id}` endpoint instead. Both layers will
  be cached on disk (Phase 0 verification finding, to be incorporated into
  Phase 1 fetcher design).
- **No `Co-Authored-By: Claude` trailer** on commits in this repo
  (user preference, recorded in CC memory).

## Metrics / Results So Far

- SCDB: 83,644 vote rows × 61 columns (release 2025_01).
- Oyez sample case (Heien): ~85–90 turns in the transcript section,
  with `speaker.roles[].type == "scotus_justice"` flagging Justices.
- No model results yet — Phase 1 has not begun.

## Current Status

- **Completed phases**: Phase 0 (proposal & repo init).
- **Current phase**: Awaiting Checkpoint 0 approval before Phase 1
  (data acquisition / bulk Oyez fetch).
- **Blockers**: None.

## What's Left

- Phase 1: build Oyez fetcher (2-step, rate-limited, retried, cached);
  bulk-fetch 2005–2024 transcripts; parse into `(case, justice, text)` rows
  with `unanimous` flag; write `data/processed/justice_case_rows.parquet`;
  3+ pytest tests.
- Phase 2: EDA, inclusion/exclusion decisions, build modeling table.
- Phase 3: 9 baseline (vectorizer × classifier) combos with per-fit timing.
- Phase 4: Sequential GridSearchCV (4A linear + vectorizer; 4B RF only).
- Phase 5: Evaluation, interpretability, per-Justice + unanimity sensitivity.
- Phase 6: ML Canvas PDF, polished `JusticeCast_Final.ipynb`, README, pytest.
- Phase 7: 8–12 slide pitch deck.

### Known risks / open questions

- SCDB `partyWinning` and `majority` codebook semantics need verification
  against the codebook before locking in label derivation (Phase 1).
- Oyez 2005–2024 window is provisional; actual coverage rate empirical
  in Phase 1. CAI invited pushback to extend the window if coverage is
  solid further back.
- Phase 4 Stage 4B RF compute on TF-IDF bigrams will be the longest single
  block of the project. Per-fit timings logged in Phase 3 baseline sweep
  set the budget; CAI to approve at Checkpoint 3.

## Non-Negotiables (carried forward from `cai-plan.md`)

1. No data leakage — split by `case_id` using `StratifiedGroupKFold`
   (fold 0 test, folds 1–4 train), pass `groups=case_id` to nested CV
   inside `GridSearchCV`. `train_test_split(stratify=y)` is forbidden
   for the primary split.
2. Stratified splits on the binary vote label, `random_state=42` everywhere.
3. Vectorizers fit on train only — enforce via `sklearn.pipeline.Pipeline`.
4. No post-hoc features — only information available the moment the Justice
   stops speaking. Vote label is the only future signal.
5. Reproducibility — fixed seed 42, pinned deps, `Restart & Run All` clean.
6. Class imbalance handled explicitly — `class_weight='balanced'` or
   documented; report ROC AUC and balanced accuracy alongside accuracy.
7. Every experiment logged — `reports/results/` CSVs, one row per
   (vectorizer, classifier, hyperparams), with per-fit wall-clock time.
   Notebook reads these CSVs, does not re-run sweeps.
8. Cache aggressively — Oyez calls cached on disk; SCDB downloaded once.
9. Frame as **Option 1 stance classification**, not sentiment.

## Definition of Done (carried forward from `cai-plan.md`)

- Notebook runs top-to-bottom on a fresh kernel (`Restart & Run All`) with
  zero errors and zero unjustified warnings.
- All 9 vectorizer × classifier baseline combinations evaluated and logged
  with per-fit timing.
- Sequential GridSearchCV applied (Stage 4A linear models joint with
  vectorizer; Stage 4B RF with fixed vectorizer).
- Final winning model has confusion matrix (rendered figure), precision,
  recall, F1, ROC AUC, ROC curve, PR curve, calibration curve.
- Per-Justice performance breakdown with prose discussion.
- Unanimity sensitivity analysis (per-Justice metrics split by unanimous
  vs contested) in the notebook.
- Top n-grams per class extracted and visualized for at least one model.
- Business interpretation paragraph (FN vs FP cost) in notebook prose.
- Machine Learning Canvas v0.4 filled and exported to PDF.
- Pitch deck 8–12 slides, exported to PDF, follows the storytelling arc.
- README documents how to reproduce from a fresh clone.
- pytest suite green.
- All artifacts committed with clean history.
- Proposal submitted to professor by **2026-05-07** (Phase 0).
- Both deliverables submitted to Canvas by **2026-05-28**.
