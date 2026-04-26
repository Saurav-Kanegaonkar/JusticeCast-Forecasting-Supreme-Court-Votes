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
- **Key columns for our task** (codebook-verified, Phase 1 Stop A):
  - Case keys: `caseId`, `term`, `docket`, `caseName`
  - Outcome: `partyWinning` —
    `0` = petitioner lost, `1` = petitioner won, `2` = unclear (excluded)
    Source: scdb.wustl.edu/documentation.php?var=partyWinning
  - Vote split: `majVotes`, `minVotes` (unanimous ⇒ `minVotes == 0`)
  - Justice: `justice` (numeric ID), `justiceName` (short SCDB code, e.g. `JGRoberts`)
  - Justice's actual vote: `vote`, `direction`, `majority` —
    **`majority`: `1` = dissent, `2` = majority** (NOT the other way around).
    Source: scdb.wustl.edu/documentation.php?var=majority
- **Label derivation** (binary, "Justice voted with petitioner") — VERIFIED:
  - `voted_petitioner = (partyWinning == 1) == (majority == 2)`
  - Implemented in `src/build_dataset.py::derive_voted_petitioner` with codebook
    citation in the docstring.
  - Excluded rows: `partyWinning == 2` (unclear winner) or `majority is NaN`
    (Justice did not participate).
  - **Heien spot-check passed end-to-end** (term=2014, docket=13-604):
    Sotomayor=1, all 7 other Justices who spoke=0. Thomas (silent) excluded
    because he produced no text. See `tests/test_builders.py::test_heien_label_spotcheck_end_to_end`.

### Oyez

- **Case metadata endpoint**: `https://api.oyez.org/cases/{term}/{docket}`
  - Verified example: `https://api.oyez.org/cases/2014/13-604` (Heien v. North
    Carolina). NOTE: an earlier version of `cai-plan.md` cited docket 13-1314
    as Heien — that's actually *Arizona State Legislature v. Arizona
    Independent Redistricting Commission*, also OT2014. The real Heien is
    docket **13-604**, caseId **2014-001**. The docket-vs-case-name confusion
    was caught at Stop A by SCDB lookup.
  - Returns case-level metadata. **Does NOT contain transcript turns.**
  - Has `oral_argument_audio[].href` pointing to the case_media endpoint.
- **Transcript endpoint** (the actual data we need):
  - Pattern: `https://api.oyez.org/case_media/oral_argument_audio/{audio_id}`
  - Verified example: `https://api.oyez.org/case_media/oral_argument_audio/23272`
    (Heien transcript)
  - JSON path: `transcript.sections[].turns[].speaker` and
    `transcript.sections[].turns[].text_blocks[].text`
- **Speaker fields** in each turn:
  - `speaker.identifier` — slug (e.g., `john_g_roberts_jr`) — used as the
    stable Justice key for joining via `data/processed/justice_id_map.csv`
  - `speaker.name` — full name
  - `speaker.roles` — array; `(role or {}).get("type") == "scotus_justice"`
    flags Justices. Advocates have `roles == []` or null.
  - `speaker.ID` — numeric (Oyez-internal)
- **Fetcher is a 2-step pull**: case JSON → for each
  `oral_argument_audio[].href`, fetch the case_media JSON → extract turns.
  Both layers cached under `data/raw/oyez/{cases,transcripts}/`. Implemented
  in `src/fetch_oyez.py`. Global rate limiter shared across both layers.

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

- [x] ~~Confirm SCDB `partyWinning` codebook semantics~~ — Stop A: 0 = lost,
      1 = won, 2 = unclear (excluded). Documented above.
- [x] ~~Confirm SCDB `majority` codebook (1 = majority, 2 = minority?)~~ —
      Stop A: codebook says **1 = dissent, 2 = majority** (opposite of my
      Phase 0 assumption). Label formula corrected accordingly. Heien spot-check
      passed end-to-end.
- [ ] Establish Oyez coverage rate empirically over 2005–2024 in Phase 1.
      If coverage is solid pre-2005, propose extending the window (CAI
      explicitly invited this pushback). Deferred to Stop B.
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
   tightened by CAI in the Phase 1 plan revision.
5. **Two-stop Phase 1 split** — accepted; codified as Non-Negotiable #10
   (hand-verify before bulk operations).
6. **Hand-built Justice ID map** — accepted; ~16 rows for 2005–2024 window
   (`data/processed/justice_id_map.csv`, ungitignored exception).

## Self-corrections logged

- **Phase 0 column-counting bug.** I read `partyWinning=6` for the Halliburton
  sample row using `awk -F','`, which broke on the embedded comma in
  `caseName`. Real value was `1`. Fix: always use pandas / `csv.reader` on
  quoted CSVs. Don't grep-walk SCDB rows.
- **Phase 0 `majority` field encoding mistake.** I'd assumed `1 = majority`,
  `2 = minority`. The codebook says the opposite (`1 = dissent`, `2 = majority`).
  Original XNOR formula would have inverted every label. Heien spot-check
  caught this — exactly what it was designed to catch.
- **`cases/2014/13-1314` is not Heien.** The cai-plan repeatedly cited that
  docket as Heien v. North Carolina; SCDB lookup shows it's *Arizona State
  Legislature v. Arizona Independent Redistricting Commission*. Real Heien
  is `2014/13-604`, caseId `2014-001`. Both Stop A smoke test and the
  fetcher tests now use the correct docket.
