# JusticeCast

Forecasting Supreme Court Justice votes from oral-argument questions.

A binary text-classification project: given the verbatim questions a Supreme
Court Justice asks during a single case's oral argument, predict whether
that Justice will vote with the petitioner or with the respondent.

Course project deliverables — Part A (pitch deck) + Part B (reproducible
Jupyter notebook).

> **Status:** Phase 0 (proposal & repo init) complete. See
> [`project-state.md`](project-state.md) for ground-truth status and
> [`cai-plan.md`](cai-plan.md) for the full plan.

## Reproduce

```sh
# 1. Clone, then:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch data (will be wired up in Phase 1)
python -m src.fetch_scdb
python -m src.fetch_oyez

# 3. Build the modeling table
python -m src.build_dataset

# 4. Run the submission notebook end-to-end
jupyter lab notebooks/JusticeCast_Final.ipynb
# then: Kernel → Restart & Run All
```

## Data sources

- **SCDB** (Washington University) — Justice-Centered file, release 2025_01.
  Free CSV. Used for vote labels and case metadata.
- **Oyez.org** — public REST API, no auth required. Used for oral-argument
  transcripts with each utterance tagged by speaker name and role.

Both layers cache to disk under `data/raw/` (gitignored). Reruns hit the
cache, not the network.

## Repo layout

```
JusticeCast/
├── data/
│   ├── raw/        # gitignored — SCDB CSV + Oyez JSON cache
│   └── processed/  # gitignored — joined parquet tables
├── src/            # fetch_scdb, fetch_oyez, build_dataset, text_clean
├── notebooks/      # 01_eda, 02_modeling, JusticeCast_Final (submission)
├── reports/        # proposal.md, ml_canvas.pdf, JusticeCast_Pitch.pdf, results/
├── tests/          # pytest suite for fetchers + builders
├── cai-plan.md     # the plan (CAI-owned)
├── CLAUDE.md       # CC working notes
├── project-state.md
├── requirements.in / requirements.txt
└── README.md
```

## Tests

```sh
pytest
```

## Team

6-person course team. Member assignments live in CAI chat.
