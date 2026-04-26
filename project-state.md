# Project State: JusticeCast
Last updated: 2026-04-26 by CC at Phase 2 Checkpoint 2

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

## What Exists (ground truth — Phase 1 Stop A)

### From Phase 0 (carried forward)
- Git repo on `main`, public GitHub remote at
  https://github.com/Saurav-Kanegaonkar/JusticeCast-Forecasting-Supreme-Court-Votes
- `.venv/` (Python 3.14.3), deps pinned in `requirements.txt`
- Directory scaffold + `cai-plan.md`, `CLAUDE.md`, `README.md`,
  `reports/proposal.md`

### Added in Phase 1 Stop A
- **SCDB**: cached at `data/raw/SCDB_2025_01_justiceCentered_Citation.csv`
  (28 MB, 83,644 rows × 61 cols, Latin-1).
- **SCDB codebook pages** cached at `data/raw/scdb_codebook/*.html` for
  `partyWinning`, `majority`, `vote`, `direction`, `caseDisposition`.
  Field semantics documented in `CLAUDE.md`.
- **Oyez Heien**: cached at `data/raw/oyez/cases/2014_13-604.json` and
  `data/raw/oyez/transcripts/23272.json` (1 audio session, 8 Justices spoke,
  Thomas silent as usual).
- **Source modules** in `src/`:
  - `fetch_scdb.py` — download + Latin-1 load, idempotent
  - `fetch_oyez.py` — 2-step fetch with global ≤1 req/sec rate limiter,
    `tenacity` exponential backoff (max 5 attempts, retries on 5xx/429/timeout),
    cache at both layers
  - `build_dataset.py` — parse cached transcripts → filter to Justices →
    aggregate per (case, Justice) with multi-audio concatenation →
    join SCDB → derive `voted_petitioner` + `unanimous` → write parquet
  - `text_clean.py` — minimal whitespace helper (Phase 2/3 will populate)
- **`data/processed/justice_id_map.csv`** — hand-built, 16 Justices in the
  2005–2024 window, ungitignored exception in `.gitignore`. 8 of 16 slugs
  empirically verified against the Heien transcript; the other 8 follow
  identical Oyez `firstname_middleinitial_lastname` convention.
- **`data/processed/justice_case_rows.parquet`** — 8 rows (Heien only)
  produced end-to-end by the Heien-only smoke pipeline.
- **Tests** at `tests/test_fetchers.py` + `tests/test_builders.py`,
  **11 tests passing** (`pytest`):
  - SCDB shape + Latin-1 encoding
  - Oyez Step 1 + Step 2 shapes on Heien
  - Rate limiter timing
  - Oyez cache idempotency
  - Label-derivation truth table (4 cases + 3 missing/unclear cases)
  - Parser filters Justices vs advocates
  - Heien end-to-end label spot-check (Sotomayor=1, others=0)
  - Justice→SCDB mapping coverage on Heien
  - Multi-audio synthetic concatenation

## Codebook-Verified Field Semantics (Phase 1 Stop A)

Source: `scdb.wustl.edu/documentation.php?var=<field>`, cached HTML in
`data/raw/scdb_codebook/`.

- **`partyWinning`**: 0 = no favorable disposition for petitioner (lost);
  1 = petitioner received favorable disposition (won); 2 = unclear.
  Rows with value 2 are EXCLUDED from labels.
- **`majority`**: **1 = dissent, 2 = majority** (note: this is opposite of
  CC's Phase 0 assumption). Justice-level only. Missing → did not participate.
  Rows with NaN are EXCLUDED from labels.
- **`vote`**: 8 categorical values capturing concurrence types (regular vs
  special, dissent, jurisdictional, etc.). For our binary label we collapse
  via `majority`, not `vote` directly.
- **`direction`**: 1 = conservative, 2 = liberal. Ideological coding —
  not used in our label (we want `voted_petitioner`, not ideology).
- **`caseDisposition`**: 11-value taxonomy (affirmed, reversed, vacated, etc.)
  that already governs `partyWinning`. Per the codebook: *"The entry in this
  variable governs whether the individual justices voted with the majority or
  in dissent."* We use `partyWinning` (the derived field) directly.

**Final binary label** (codified in `src/build_dataset.py::derive_voted_petitioner`):
```python
voted_petitioner = (partyWinning == 1) == (majority == 2)
```
Returns `None` if either field is missing or `partyWinning == 2`.

**Heien spot-check passed**: 8 spoken Justices in the parsed transcript,
Sotomayor (lone dissent) → 1, all others → 0.

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
- **Phase 1 split into Stop A / Stop B**: codified as Non-Negotiable #10
  (hand-verify before any irreversible/expensive operation). Stop A
  produces a Heien-only proof of correctness; Stop B does the bulk fetch
  only after CAI signs off on Stop A.
- **Heien is at docket 13-604, NOT 13-1314.** Earlier `cai-plan.md`
  versions cited 13-1314 as Heien — that's actually *Arizona State
  Legislature v. AIRC*. Both are OT2014 cases. The mistake was caught at
  Stop A by SCDB lookup. All tests and smoke probes use the correct docket.
- **`majority` field encoding is `1=dissent, 2=majority`**, opposite of
  CC's Phase 0 assumption. Verified against the SCDB codebook in Stop A.
  Original XNOR formula would have inverted every label. Heien spot-check
  caught this — exactly what it was designed to do (Non-Negotiable #10
  was the right call).

## Metrics / Results So Far

- **SCDB**: 83,644 vote rows × 61 columns (release 2025_01).
- **`partyWinning` distribution**: 0=29,819 (35.6%); 1=53,627 (64.1%);
  2=54 (0.1%); NaN=144 (0.2%). Empirical petitioner-win rate ≈ 64%
  (matches the ~65–70% expectation in Non-Negotiable #6).
- **`majority` distribution**: 1=14,709 (17.6%); 2=65,952 (78.9%);
  NaN=2,983 (3.6%, mostly recusals).
- **2005–2024 window**: 13,149 justice-vote rows, 1,471 unique cases,
  1,470 unique (term, docket) pairs (one case has duplicate docket entry).
- **Heien (2014/13-604) end-to-end**: 8 Justice rows produced, Sotomayor=1,
  others=0. Word counts range 291 (Breyer) to 1,131 (Scalia). Sotomayor
  had 20 turns — the most engagement, despite being the lone dissent.
- **Bulk fetch complete (Stop B)**: 1,470 cases attempted, 1,322 succeeded
  with audio, 98 had no audio, 50 failed. Total wall-clock: 54 min, 377 MB cache.
- **Stop C rescue**: 2 high-value re-argued cases recovered at term-1
  (Citizens United, Kiobel). 25 standard-format failures confirmed as
  legitimate gaps (per-curiam reversals, summary dispositions, in-re).
- **Joined parquet (pre-cleanup)**: `data/processed/justice_case_rows.parquet`
  — 10,308 rows, 1,309 distinct cases.
- **Modeling table (post-cleanup, Phase 2)**:
  `data/processed/modeling_table.parquet` — **10,039 rows × 20 cols**,
  1,293 distinct cases, 16 Justices.
- **Label distribution (modeling table)**: 62.4% with-petitioner /
  37.6% against. **Majority-class baseline = 62.4%**.
- **Unanimity (modeling table)**: 4,207 rows (41.9% of labeled) in
  unanimous cases.
- **Multi-audio cases**: 15 after rescue (added Citizens United and Kiobel,
  both famously re-argued). Concatenation worked correctly.
- **Justice coverage cross-check**: all 16 slugs returned non-zero matches.
  Lowest meaningful ratio: Thomas at 20.5% (his real silence pattern, not
  a bug). All 8 previously unverified slugs validated with no fix needed.
- **Word counts per Justice**: tail at O'Connor (median 148, n=35),
  head at KBJackson (median 1,204, n=174). Thomas median 233. Top
  questioners (median): KBJackson > Breyer > Kagan/Souter/Gorsuch/Scalia.
- **No model results yet** — modeling begins in Phase 3.

## Phase 2 Cleanup Decisions (codified in `src/build_modeling_table.py`)

1. **Drop NaN-label rows** — `partyWinning ∈ {2, NaN}` (24 rows, "unclear
   winner" or missing) or `majority NaN` (147 rows, Justice did not
   participate). Sweeps up the 45 OT2015 unmatched rows from Phase 1
   (post-Scalia-death cases, Oyez/SCDB term-encoding mismatches, and the
   Kagan-as-SG / Souter-retired Citizens United artifacts).
2. **Drop original-jurisdiction cases** — docket patterns `* ORIG`,
   `*, Orig.`, `22O*`. Substantively different (state-vs-state, no cert
   grant); Oyez doesn't catalog them under SCDB-style dockets anyway.
   Belt-and-suspenders for any that slipped through.
3. **Drop rows with `word_count < 30`** — empirical floor chosen from the
   distribution (1st percentile = 35 words). Below 30 words almost every
   row is a truncated half-utterance like `"What --"` or `"Counsel --"` with
   no stance signal.
4. **Keep unanimous cases** (Phase 1 decision, carried forward). Flagged
   via `unanimous` metadata for the Phase 5 sensitivity split.
5. **Keep Thomas** with low-n caveat (Phase 1 decision, carried forward).
   302 → 295 rows after cleanup; enough for stable per-Justice estimates.

Cleanup audit (per-stage row counts in `reports/results/modeling_table_audit.csv`):

  input (justice_case_rows.parquet)                    10,308
  after drop NaN-label rows                            10,137  (-171)
  after drop original-jurisdiction cases               10,120  (- 17)
  after drop word_count < 30                           10,039  (- 81)

## Current Status

- **Completed phases**: Phase 0; Phase 1 Stop A; Phase 1 Stop B (bulk fetch);
  Stop C (rescue, +2 cases incl. Citizens United); Phase 2 (cleanup + EDA
  notebook + modeling table).
- **Current phase**: Awaiting Checkpoint 2 approval before Phase 3 (baseline
  sweep — 9 vectorizer × classifier combos with per-fit timing).
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
8. Cache aggressively — Oyez calls cached on disk (both layers);
   SCDB downloaded once.
9. Frame as **Option 1 stance classification**, not sentiment.
10. Hand-verify before bulk operations — any irreversible/expensive step
    (bulk API fetch, multi-hour grid search) gets a smoke test on a
    hand-checked sample first.

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
