# JusticeCast — CC Working Notes

@cai-plan.md

## What this file is

CC (Claude Code) working notes — local technical state, conventions, and gotchas
that should not live in `cai-plan.md` (which CAI owns) or `project-state.md`
(which is the human-readable ground truth, refreshed at handoffs).

Read `cai-plan.md` (imported above) for the full plan and the live
**Current Instruction** at the bottom of that file.

## Environment

- **Python**: 3.14.3 (system Homebrew)
- **venv**: `.venv/` in project root (gitignored). Activate with
  `source .venv/bin/activate`. Direct deps in `requirements.in`,
  pinned via `pip freeze > requirements.txt`.
- **Notebook kernel**: launch JupyterLab from inside the venv
  (`.venv/bin/jupyter lab`) so the notebook picks up the right Python.
- **Reproducibility seed**: 42 everywhere (Non-Negotiable #5).

## Repo layout

See `cai-plan.md` Architecture section. `data/raw/` and `data/processed/`
are gitignored — re-fetch via the `src/fetch_*` modules.

## Data sources — verified

### SCDB (Supreme Court Database)

- **Site**: http://scdb.wustl.edu (HTTP only — HTTPS not configured)
- **Latest release**: `2025_01`
- **Justice-Centered, Cases by Citation** download URL:
  `http://scdb.wustl.edu/_brickFiles/2025_01/SCDB_2025_01_justiceCentered_Citation.csv.zip`
- **File**: 1.6 MB zipped, 28 MB unzipped, 83,644 vote rows, 61 columns
- **Encoding**: Latin-1 / Windows-1252 (the file shows `Â§` mojibake when
  read as UTF-8). Use `pd.read_csv(..., encoding='latin-1')`.
- **Key columns for our task**:
  - Case keys: `caseId`, `term`, `docket`, `caseName`
  - Outcome: `partyWinning` (1 = petitioner wins, 0 = respondent wins, per codebook)
  - Vote split: `majVotes`, `minVotes` (unanimous ⇒ `minVotes == 0`)
  - Justice: `justice` (numeric ID), `justiceName` (string)
  - Justice's actual vote: `vote`, `direction`, `majority` (1 = in majority, 2 = in minority)
- **Label derivation** (binary, "Justice voted with petitioner"):
  - `voted_petitioner = (partyWinning == 1) == (majority == 1)`
  - i.e., Justice voted petitioner if they were in the majority of a
    petitioner-winning case OR in the minority of a respondent-winning case.
  - **Verify against codebook before locking this in** (Phase 1 task).

### Oyez

- **Case metadata endpoint**: `https://api.oyez.org/cases/{term}/{docket}`
  - Example: `https://api.oyez.org/cases/2014/13-1314` (Heien v. North Carolina)
  - Returns case-level metadata. **Does NOT contain transcript turns.**
  - Has `oral_argument_audio[].href` pointing to the case_media endpoint.
- **Transcript endpoint** (the actual data we need):
  - Pattern: `https://api.oyez.org/case_media/oral_argument_audio/{audio_id}`
  - Example: `https://api.oyez.org/case_media/oral_argument_audio/23270`
  - JSON path: `transcript.sections[].turns[].speaker` and
    `transcript.sections[].turns[].text_blocks[].text`
- **Speaker fields** in each turn:
  - `speaker.identifier` — slug (e.g., `john_g_roberts_jr`) — use as the
    stable Justice ID for joining to SCDB `justiceName`
  - `speaker.name` — full name
  - `speaker.roles` — array; `roles[].type == "scotus_justice"` flags
    Justices. Advocates have `roles == null`.
  - `speaker.ID` — numeric (Oyez-internal)
- **Fetcher must be a 2-step pull**: case JSON → for each
  `oral_argument_audio[].href`, fetch the case_media JSON → extract turns.
  Cache both layers under `data/raw/oyez/`.

## Conventions

- **Splits**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
  with `groups=case_id` (Non-Negotiable #1). Fold 0 = test, folds 1–4 = train.
  Never `train_test_split(stratify=y)` for the primary split — it ignores groups.
- **Vectorizers** live inside an `sklearn.pipeline.Pipeline` so vocabulary
  is built only on train fold (Non-Negotiable #3).
- **`LinearSVC` + ROC AUC**: pass `decision_function(X_test)` scores to
  `roc_auc_score`. Wrap with `CalibratedClassifierCV(method='sigmoid', cv=5)`
  **only** for the calibration curve in Phase 5.
- **Experiment log**: every fit appends a row to `reports/results/*.csv`
  including per-fit wall-clock time. Notebook reads from these CSVs at render
  time — do not re-run sweeps to re-knit.
- **Commit style**: NO `Co-Authored-By: Claude` trailer on this repo
  (per user, recorded in memory).

## Open questions / verifications outstanding

- [ ] Confirm SCDB `partyWinning` codebook semantics (1 = petitioner?
      respondent? both? other?). Codebook is at scdb.wustl.edu under
      Documentation. Cross-check with one known case before bulk derivation.
- [ ] Confirm SCDB `majority` codebook (1 = majority, 2 = minority?).
- [ ] Establish Oyez coverage rate empirically over 2005–2024 in Phase 1.
      If coverage is solid pre-2005, propose extending the window (CAI
      explicitly invited this pushback).
- [ ] Decide unanimity column derivation: SCDB `minVotes == 0` is a
      first-pass; verify against `decisionType` (special vote types).

## Pushback log (resolved)

1. **`StratifiedGroupKFold` over `train_test_split`** — accepted, codified
   in Non-Negotiable #1.
2. **`LinearSVC` decision_function for AUC, calibration only when needed**
   — accepted, Phases 3 + 5 updated.
3. **Keep unanimous cases, flag as metadata, sensitivity-analyze in Phase 5**
   — accepted, Phases 1/2/5 updated.
4. **Sequential GridSearchCV (4A linear+vectorizer, 4B RF with fixed vec)**
   — accepted; CC clarified Stage 4A is two `GridSearchCV` runs sharing the
   vectorizer grid, not literally one (sklearn API constraint). Plan wording
   to be tightened next cycle.
