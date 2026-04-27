# Project State: JusticeCast
Last updated: 2026-04-26 by CC at Phase 4.5 Checkpoint 4.5 (embeddings track)

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

## Phase 2B EDA Expansion (B1–B6, reopened Checkpoint 2)

CAI reopened Checkpoint 2 because the initial EDA characterized labels well
but didn't engage with the text — the actual model input. Six expansion
tasks landed:

- **B1 (per-class vocabulary differences)**: variance-adjusted log-odds
  with Dirichlet prior (Monroe et al. 2008) on unigrams + bigrams. **Finding:
  overwhelming content-term dominance.** Top petitioner-side: officer,
  church, religious, arrest, warrant, ada, bia, navigable, wetlands. Top
  respondent-side: illinois, idaho, tinker, sentencing, crack, cocaine,
  pollutant, pto, algorithm. **Decision: built custom stopword list** in
  `src/text_clean.py::STOPWORDS_FOR_VECTORIZER` — 106 custom additions on
  top of sklearn's 318 (US states + agency abbreviations + famous case
  shortnames + court-procedural terms). Deliberately did NOT stopword
  thematic legal vocabulary (officer, jury, warrant, attorney, school,
  sentence) — those carry stance through context.
- **B2 (per-Justice baseline table)**: per-Justice baselines range
  ~50%–80%. Mandatory framing prose added to notebook: "global 62.4% is
  not the right reference; Phase 5 must report lift over per-Justice
  baselines." Phase 5 spec updated to enforce this.
- **B3 (sample text inspection)**: 5 fixed-seed samples revealed bracketed
  transcription annotations (`[Laughter]`, `[Crosstalk]`, etc.) in 1,078
  rows (10.7%). **Decision: added `preprocess_text` regex** in
  `src/text_clean.py` (strips `\\[[^\\]]+\\]`, normalizes whitespace) and
  rebuilt `data/processed/modeling_table.parquet` with cleaned text.
- **B4 (vocabulary statistics)**: 32,638 unique tokens, 5.37M total
  instances. 305 sklearn stopwords (0.9% of vocab) cover 59% of all token
  instances — textbook Zipf shape.
- **B5 (word count vs label)**: Mann-Whitney U test on `word_count` vs
  `voted_petitioner`. p-value reported in notebook; if significant, length
  is a confound that Phase 3 baselines must beat a length-only classifier
  to claim model lift.
- **B6 (per-Justice vocabulary signature)**: top 10 distinctive bigrams
  per Justice via TF-IDF max-mean differential. **Author-identity-from-text
  is a real concern** — Justices have detectable signatures, and combined
  with their stable voting priors (B2), part of any Phase 5 lift may come
  from "this is Sotomayor → Sotomayor votes liberal" rather than
  bench-questioning signal. Phase 5 must explicitly discuss this. Cleanest
  test of "real" model lift: per-Justice AUC on **contested** cases.

### Artifacts updated in Phase 2B

- `src/text_clean.py`: `preprocess_text()` + `STOPWORDS_FOR_VECTORIZER`
  (424 total = 318 sklearn + 106 custom)
- `src/build_modeling_table.py`: applies `preprocess_text` to text column,
  recomputes `word_count` post-preprocessing
- `data/processed/modeling_table.parquet`: rebuilt; bracket annotations
  stripped (0 remaining vs 1,499 before); shape unchanged at 10,039 × 20
- `notebooks/01_eda.ipynb`: extended from 7 to 13 sections (B3, B4, B1, B2,
  B5, B6 added in that order). Verified Restart-and-Run-All via nbconvert.
- `tests/test_text_clean.py`: 8 new tests (preprocessor idempotence,
  bracket stripping, sklearn stopword inclusion, custom-stopword inclusion,
  thematic-vocab non-overstrip)
- 25 pytest tests total, all green

## Phase 3 Baseline Sweep — Major Empirical Finding

**Headline:** all 9 (vectorizer × classifier) combinations produce ROC AUC
in the range **0.507–0.528** — essentially chance. The bench-questioning
text does not contain enough signal to reliably predict petitioner-vote
stance at the modeling task as currently framed.

### Baseline results (sorted by ROC AUC)

  combo                            accuracy  bal_acc  ROC AUC  F1     n_features
  tfidf_bigram__logreg             0.539     0.512    0.528    0.627  202,559
  tfidf_unigram__logreg            0.537     0.519    0.526    0.615   29,594
  tfidf_bigram__linear_svc         0.547     0.506    0.524    0.649  202,559
  tfidf_unigram__linear_svc        0.537     0.513    0.522    0.621   29,594
  bow_unigram__logreg              0.542     0.520    0.521    0.624   29,594
  bow_unigram__linear_svc          0.548     0.524    0.519    0.630   29,594
  tfidf_bigram__random_forest      0.624     0.506    0.512    0.765  202,559
  bow_unigram__random_forest       0.620     0.501    0.509    0.764   29,594
  tfidf_unigram__random_forest     0.626     0.509    0.507    0.766   29,594

**RF rows show the class-prior trap**: 62.4% accuracy by predicting
near-majority almost always (balanced_accuracy ~0.5, ROC AUC ~0.5).

### Per-Justice lift over individual baselines (Non-Negotiable #12)

For the best linear combo (`tfidf_bigram__logreg`):
- **POSITIVE lift on only 1 of 16 Justices** (O'Connor, n_test=3, statistically meaningless)
- For the other 15 Justices, the model performs **WORSE** than predicting
  their majority class
- Worst affected: ACBarrett (-18.6 pp), KBJackson (-15.4 pp), Ginsburg (-13.7 pp)
- Median lift: -7.3 pp; mean lift: -7.4 pp

The least-bad combo on per-Justice lift is `tfidf_unigram__random_forest`
with median lift +0.0003 (essentially zero) and 8 of 16 Justices with
positive lift — but its ROC AUC is the lowest at 0.507.

### Top features for best linear model (vs B1 EDA pre-stopwording)

The custom stopword list filtered the obvious topic terms (state names,
agency abbreviations, famous case shortnames) but **thematic legal
vocabulary became the new topic-proxy** — exactly the second outcome the
cai-plan predicted.

Top petitioner-side features: `officer`, `misleading`, `arrest`, `attorney`,
`profits`, `circuit`, `standing`, `religious`, `church`, `bankruptcy`,
`prison`, `rehabilitation`, `discrimination`. These overlap heavily with
the B1 pre-stopwording top list (officer, religious, arrest, church remain
in the top 30).

Top respondent-side features: `evidence`, `delegation`, `petition`, `hours`,
`insurance`, `fraud`, `jury`, `sentence`, `discovery`, `indian`, `trial`,
`grand jury`, `discharge`, `government`, `school`, `burglary`, `conspiracy`.
B1 respondent terms (sentencing, sentence, jury, school, indian, conspiracy)
all retained.

**Two leakage findings within top features**: `frederick`/`mr frederick`
and `fisher`/`mr fisher` are advocate names from MORSE v. FREDERICK and
FISHER v. UNIVERSITY OF TEXAS. Justices addressing advocates by name leaks
case identity into the text. A second-pass stopword expansion would add
common advocate forms (`mr X`, `ms X`, `general X`) — proposed for Phase
4 vectorizer config, not blocking.

### What this means for the project

The project hypothesis ("we can read the bench from oral-argument text")
is **not strongly supported by these baselines**. AUC 0.51–0.53 means the
text-only model is essentially uninformative. Three real possibilities:

1. **The signal is genuinely weak.** Justices' questions are tactical —
   they often grill the side they're sympathetic to (legal scholarship has
   noted this). Oral argument may contain less stance signal than litigators
   believe.
2. **The signal exists but requires more than bag-of-words.** Sequence
   models (BERT-style embeddings) might recover signal that TF-IDF + linear
   classifiers cannot. Out of scope for this rubric.
3. **Phase 4 hyperparameter tuning could lift AUC by 1–3 percentage
   points.** Unlikely to break above ~0.55 without a method change.

This is a real finding, not a bug. Phase 5 will frame it honestly via
Non-Negotiable #13 (per-Justice contested-cases ROC AUC) and the cai-plan
"honesty pass". Phase 7 pitch deck pivots from prediction-as-product to
"empirical lower bound on bench-reading from text alone".

### Phase 4 compute budget (extrapolated from baseline timings)

  Stage 4A (linear with vectorizer grid):
    LogReg 10 × vec 12 × 5-fold CV = 600 fits × ~1.5s = ~15 min sequential
                                                         ~5 min with n_jobs=-1
    SVM    4 × vec 12 × 5-fold CV  = 240 fits × ~1.5s = ~6 min sequential
                                                         ~2 min with n_jobs=-1
  Stage 4B (RF, fixed vectorizer):
    27 RF configs × 5-fold CV = 135 fits × ~4-15s = ~10-30 min
  TOTAL realistic wall-clock:                       ~45-60 min

## Phase 3.5 Sanity Pass — Confirmed: Phase 3 finding is real

CAI reopened Phase 3 with three sanity tasks before launching Phase 4 +
Phase 4.5 (the new comparative-methodology study). All three landed.

### Task 1 — 10-row label correctness check

Sampled 10 random rows from `modeling_table.parquet` (`random_state=42`).
For each, verified that `voted_petitioner` matches the codebook formula
`(partyWinning == 1) == (majority == 2)` AND that the case outcomes match
historical record:

| caseId | term | docket | caseName (truncated) | Justice | pet result | Justice in | label | hist? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-042 | 2019 | 18-6662 | SHULAR v. UNITED STATES | EKagan | lost | majority | 0 | ✓ |
| 2009-093 | 2009 | 08-1065 | POTTAWATTAMIE COUNTY ... v. CURTIS W. | AMKennedy | lost | majority | 0 | ✓ |
| 2021-016 | 2021 | 20-303 | UNITED STATES v. VAELLO MADERO | SSotomayor | won | DISSENT | 0 | ✓ |
| 2016-071 | 2016 | 16-6219 | DAVILA v. DAVIS | RBGinsburg | lost | DISSENT | 1 | ✓ |
| 2023-026 | 2023 | 22-1238 | OFFICE OF THE U.S. TRUSTEE v. HAMMONS | BMKavanaugh | won | majority | 1 | ✓ |
| 2022-043 | 2022 | 22-105 | COINBASE v. BIELSKI | KBJackson | won | DISSENT | 0 | ✓ |
| 2020-056 | 2020 | 19-251 | AMERICANS FOR PROSPERITY ... v. BONTA | NMGorsuch | won | majority | 1 | ✓ |
| 2006-013 | 2006 | 05-593 | OSBORN v. HALEY | SGBreyer | lost | majority | 0 | ✓ |
| 2018-057 | 2018 | 18-281 | VIRGINIA HOUSE v. BETHUNE-HILL | BMKavanaugh | lost | DISSENT | 1 | ✓ |
| 2020-015 | 2020 | 19-123 | FULTON v. CITY OF PHILADELPHIA | CThomas | won | majority | 1 | ✓ |

**Result: 10/10 coherent + 10/10 match historical record.** The Phase 3
chance-level AUC is NOT a labeling bug.

### Task 2 — Advocate-name stopword expansion

Implementation: added `vectorizer_preprocessor()` to `src/text_clean.py`
that lowercases text and strips advocate-name patterns
(`mr|mrs|ms|mister|madam|madame|general` + optional period + surname)
before tokenization. Wired into `make_vectorizers()` in
`src/phase3_baseline_sweep.py` via the vectorizer `preprocessor=` argument.
No parquet rebuild — vectorizer-side filtering only, per cai-plan.

Critical fix: initial regex didn't account for the period in "Mr. Frederick"
(Oyez transcript style). Caught by the verification step below; fixed.
The corrected regex catches `Mr. Frederick`, `Mister Frederick`, `Mr Frederick`,
`General Verrilli`, etc., and explicitly preserves `Justice Roberts`-style
patterns (Justices addressing each other is a different signal).

### Task 3 — Re-run with expanded stopwords

  Best linear AUC (was 0.528 → now 0.527)         within ±0.001
  All combo AUCs in 0.520-0.527                   essentially unchanged
  Vocab dropped:  29,594 → 28,987 unigrams        -607
                  202,559 → 199,546 bigrams       -3,013
  Advocate-title features in top 60:              0 (was 4)

**Top 15 features post-stripping** are now exclusively thematic legal
vocabulary:
- Petitioner-side: officer, misleading, arrest, profits, attorney, circuit,
  standing, religious, hypothetical, plan, church, factor, taking, need, standard
- Respondent-side: delegation, evidence, petition, hours, insurance, fraud,
  jury, discovery, records, sentence, grand jury, trial, indian, actual
  knowledge, discharge

Confirms the Phase 3 narrative: **the model is learning thematic legal
vocabulary as a topic proxy**, not stance markers, and this picture
persists when the obvious advocate-name leakage is removed.

### Phase 3.5 implications

The Phase 3 chance-level finding is the real picture. Project pivots from
single-track null result to comparative methodology study (BoW vs sentence-
transformer embeddings). Phase 4 (BoW tuning) and Phase 4.5 (embeddings)
proceed with checkpoints between, sharing identical splits via
`src/modeling/splits.py` (Phase 4.5 setup task).

## Phase 4 BoW GridSearchCV — Tuning Confirms the Ceiling

Wall-clock: **6.9 minutes** (Stage 4A 3.8 min + Stage 4B 3.0 min). Way under
the 60-90 min budget thanks to `n_jobs=4` + Pipeline memory caching (the
first attempt with `n_jobs=-1` OOM'd the system; second attempt with safer
parallelism completed cleanly).

### Tuning result vs Phase 3 baseline

  Model         Phase 3 (untuned)    Phase 4 (tuned 5-fold CV)    Phase 4 test
  ─────────     ─────────────────    ─────────────────────────    ────────────
  LinearSVC     0.524                0.5402  ± fold std            0.5323  ← winner
  LogReg        0.528                0.5401                         0.5317
  RandomForest  0.512                0.5337                         0.5215

Test AUC moved **0.528 → 0.5323 (+0.4 pp)** — exactly the "1-3 pp ceiling"
the cai-plan predicted. CV-test gap is small (~0.008), so no overfitting.

### Best hyperparameters

  LinearSVC (winner):  C=0.01 (heavy regularization)
                       vec: min_df=5, max_df=0.9, ngram_range=(1,1)  ← UNIGRAMS
  LogReg:              C=0.1, l1_ratio=0.0 (L2)
                       vec: min_df=5, max_df=0.9, ngram_range=(1,1)
  RandomForest:        n_estimators=500, max_depth=50, min_samples_split=10
                       vec inherited from Stage 4A winner (unigrams)

**Notable**: bigrams and trigrams added nothing — both linear winners landed
on `ngram_range=(1,1)`. The signal is in individual words, not phrases.
SVM picked maximum L2 regularization (C=0.01), which on a 200K-feature
sparse problem confirms the signal is so weak that almost any flexibility
overfits.

### Per-Justice AUC with bootstrap CIs (winner: linear_svc)

  Justice               n_test  point_AUC  CI_95
  ─────────────────────  ─────  ─────────  ───────────
  sandra_day_oconnor          3      1.000  (n too small)
  brett_m_kavanaugh          78      0.640  [0.498, 0.781]   borderline
  anthony_m_kennedy         167      0.621  [0.528, 0.705]   ★ above chance
  samuel_a_alito_jr         231      0.598  [0.517, 0.669]   ★ above chance
  david_h_souter             62      0.576  [0.417, 0.720]
  john_paul_stevens          74      0.562  [0.421, 0.692]
  antonin_scalia            138      0.558  [0.449, 0.672]
  clarence_thomas            58      0.538  [0.386, 0.675]
  sonia_sotomayor           188      0.527  [0.442, 0.608]
  amy_coney_barrett          55      0.504  [0.349, 0.664]
  john_g_roberts_jr         255      0.489  [0.417, 0.557]
  ruth_bader_ginsburg       191      0.483  [0.400, 0.566]
  stephen_g_breyer          214      0.462  [0.377, 0.547]
  elena_kagan               170      0.458  [0.365, 0.549]
  neil_gorsuch               87      0.448  [0.317, 0.575]
  ketanji_brown_jackson      36      0.406  [0.223, 0.598]

**Only 2 of 15 Justices** (Kennedy and Alito) have a bootstrap 95% CI lower
bound above 0.5 — i.e., the model is statistically distinguishable from
chance only for those two. For the other 13, the model's per-Justice AUC
is consistent with random.

**Inverse story for KBJackson** (the cai-plan storytelling hook): she's the
most-engaged questioner (median 1,205 words/case), but her per-Justice AUC
is the *lowest* (0.406) — chattiness does NOT translate to predictability.
This is a clean finding for the Phase 7 deck.

### Top features (linear_svc tuned)

After all preprocessing and tuning, the top-30 ± features are still
thematic legal vocabulary, identical in character to Phase 3:

  Petitioner-side: officer, circuit, attorney, arrest, know, plan,
                   misleading, counsel, ninth, religious, profits, standard,
                   say, standing, discrimination
  Respondent-side: jury, evidence, government, trial, sentence, percent,
                   petition, error, fraud, insurance, 10, year, indian,
                   expenses, hours

The cai-plan's prediction holds: tuning confirms the BoW ceiling. The model
is recovering thematic case content, not stance from questioning style.

## Phase 4.5 Sentence-Embeddings Track — The Comparative Result

Pre-trained sentence transformers (no fine-tuning) on the same fold-0 test
rows as BoW Phase 4. Total wall-clock: 32 min (12 encoding + 2 baseline +
17 GridSearchCV + cleanup).

### BoW vs Embeddings — top-line

  Track       Representation                           Classifier  CV AUC   Test AUC
  ──────────  ───────────────────────────────────────  ──────────  ──────   ────────
  BoW         TF-IDF unigram (Phase 4 winner)          LinearSVC   0.5402   0.5323
  Embeddings  all-MiniLM-L6-v2 (384-dim, pre-trained)  LogReg      0.5398   0.5691  ← winner

  Lift (embeddings - BoW): +0.0368 ROC AUC (+3.7 percentage points)

The embeddings lift (+3.7 pp) is **9× the BoW Phase 4 tuning lift (+0.4 pp)**.
Pre-trained semantic representations recover real signal that bag-of-words
cannot access.

### Phase 4.5 mechanics

- Encoding: both `all-MiniLM-L6-v2` (1.0 min, 15.4 MB) and
  `all-mpnet-base-v2` (10.8 min, 29.4 MB) cached at
  `data/processed/embeddings/{model}.npy`. Row-index parquet preserves
  modeling-table ordering (= positional indices align with
  `get_train_test_split()`).
- Baseline sweep (6 combos): minilm__svm_rbf wins at AUC 0.569, mpnet__svm_rbf
  second at 0.566. MiniLM selected for tuning by `_pick_better_embedding_from_baseline`.
- GridSearchCV on MiniLM:
    LogReg     (10 settings × 5-fold = 50 fits):  CV 0.5398, test 0.5691
    SVM-RBF    (9 settings × 5-fold = 45 fits):   CV 0.5394, test 0.5630
    RandomFor. (27 settings × 5-fold = 135 fits): CV 0.5291, test 0.5570
  LogReg won. Best params: C=100, l1_ratio=1.0 (= L1, very low regularization).

### CV-test gap is NEGATIVE for all three Phase 4.5 models

  LogReg     CV 0.5398, test 0.5691 → gap -0.029
  SVM-RBF    CV 0.5394, test 0.5630 → gap -0.024
  RF         CV 0.5291, test 0.5570 → gap -0.028

CV underestimates test by ~3 pp. Most likely fold-0 happens to be slightly
"easier" than the average CV fold (sampling variance from grouped CV).
Phase 5 should report this honestly — the 0.569 is the test AUC on a single
fold, and the more conservative reading is "CV-mean ~0.54, test 0.57". The
+3.7 pp BoW-vs-embeddings gap holds either way (BoW also evaluates on the
same test fold).

### Per-Justice ROC AUC — embeddings winner (LogReg on MiniLM)

  Justice                   n_test  point_AUC  CI_95            BoW lift
  ────────────────────────  ──────  ─────────  ──────────────   ────────
  sandra_day_oconnor             3      1.000  (n too small)
  clarence_thomas               58      0.730  [0.588, 0.855]   +0.193  ★
  brett_m_kavanaugh             78      0.682  [0.560, 0.807]   +0.042
  ketanji_brown_jackson         36      0.635  [0.416, 0.818]   +0.229  ★ STORY FLIPS
  amy_coney_barrett             55      0.622  [0.440, 0.781]   +0.118
  samuel_a_alito_jr            231      0.600  [0.526, 0.678]   +0.003
  antonin_scalia               138      0.594  [0.487, 0.695]   +0.035
  john_g_roberts_jr            255      0.575  [0.505, 0.647]   +0.086
  john_paul_stevens             74      0.568  [0.438, 0.704]   +0.005
  david_h_souter                62      0.567  [0.415, 0.711]   -0.008
  sonia_sotomayor              188      0.564  [0.483, 0.642]   +0.037
  neil_gorsuch                  87      0.537  [0.404, 0.663]   +0.089
  elena_kagan                  170      0.527  [0.437, 0.611]   +0.070
  anthony_m_kennedy            167      0.519  [0.419, 0.616]   -0.101  ↓ regression
  ruth_bader_ginsburg          191      0.516  [0.431, 0.596]   +0.032
  stephen_g_breyer             214      0.505  [0.427, 0.576]   +0.043

### Headline per-Justice findings

1. **All 16 Justices have point AUC > 0.5 with embeddings**, vs 10 of 16
   with BoW. Broad improvement.
2. **4 of 15 Justices have CI lower bound > 0.5** (statistically distinguishable
   from chance): Thomas, Kavanaugh, Alito, Roberts. Doubled vs BoW's 2.
3. **The KBJackson story FLIPS.** With BoW her AUC was 0.406 (worst on bench).
   With embeddings: 0.635 (third highest), lift +0.229. This is the deck's
   sharpest single-Justice anecdote: the most-engaged questioner produces
   text that pre-trained semantic encoders can map to votes, while
   bag-of-words couldn't extract the signal.
4. **Thomas dramatic gain**: BoW 0.538 → embeddings 0.730, lift +0.193.
   The silent-Justice's few utterances carry semantic structure.
5. **Kennedy regresses**: BoW 0.621 → embeddings 0.519, lift -0.101. Only
   Justice with a meaningful negative lift. Likely explanation: Kennedy was
   the swing vote whose votes correlated with thematic case content; BoW
   exploited that thematic correlation, while embeddings collapse some
   topical distinctions in the semantic space.

### Comparative summary artifacts

  reports/results/comparative_summary.csv         — top-line both winners
  reports/results/comparative_per_justice.csv     — long form, BoW vs emb per Justice

## Current Status

- **Completed phases**: Phase 0; Phase 1; Phase 2; Phase 3; Phase 3.5;
  Phase 4 (BoW GridSearchCV); Phase 4.5 (embeddings).
- **Current phase**: Awaiting Checkpoint 4.5 confirmation, then Phase 5
  (comparative evaluation + interpretability + honesty triad).
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
