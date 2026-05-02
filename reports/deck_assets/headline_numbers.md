# Headline Numbers

Every number cited in `slide_content_spec.md` traces back to a row in `reports/results/`. **Do not invent numbers in the deck beyond what appears here.**

## Comparative AUC (Phase 5)

| Number | Value | Source CSV |
| --- | --- | --- |
| BoW best test ROC AUC | **0.5323** | `phase4_test_eval.csv` (model=linear_svc) |
| BoW best params | LinearSVC + TF-IDF unigram, C=0.01, min_df=5, max_df=0.9 | `phase4_test_eval.csv` (best_params) |
| Embeddings best test ROC AUC | **0.5691** | `phase45_test_eval.csv` (model=logreg) |
| Embeddings best params | LogReg + MiniLM-L6-v2, C=100, l1_ratio=1.0 | `phase45_test_eval.csv` (best_params) |
| Overall lift | **+0.0368** = +3.7 pp | computed: 0.5691 − 0.5323 |
| BoW tuning lift over baseline | +0.4 pp (0.528 → 0.532) | Phase 3 best vs Phase 4 winner |
| BoW 5-fold CV mean AUC | 0.5402 | `phase4_test_eval.csv` (best_cv_score) |
| Embeddings 5-fold CV mean AUC | 0.5398 | `phase45_test_eval.csv` (best_cv_score) |
| BoW CV-test gap | +0.008 (CV slightly higher) | computed |
| Embeddings CV-test gap | −0.029 (test fold slightly easier) | computed |

## Honesty triad — per-Justice mean AUC by slice

| Slice | BoW | Embeddings | Lift |
| --- | --- | --- | --- |
| Global (per-Justice mean) | 0.554 | 0.609 | +0.055 |
| Unanimous-only | 0.566 | 0.615 | +0.049 |
| **Contested-only (the strict test)** | **0.532** | **0.576** | **+0.044** |

Source: `phase5_honesty_triad.csv`, mean of `point_auc` grouped by `(track, slice)` over Justices with point AUC defined.

## Per-Justice count above 0.5 by slice

| Slice | BoW | Embeddings |
| --- | --- | --- |
| Global | 10 / 16 | 16 / 16 |
| Unanimous | 11 / 16 | 14 / 16 |
| **Contested** | **9 / 15** | **13 / 15** |

Source: `phase5_honesty_triad.csv`. (15 not 16 on contested because some Justices have no contested test rows.)

## KBJackson centerpiece

| Number | Value | Source |
| --- | --- | --- |
| BoW global AUC | 0.406 (worst on bench, below random) | `phase4_per_justice_auc.csv` |
| Embeddings global AUC | 0.635 (3rd best on bench) | `phase45_per_justice_auc.csv` |
| BoW contested AUC | 0.405 | `phase5_honesty_triad.csv` (track=bow, slice=contested) |
| Embeddings contested AUC | 0.643 | `phase5_honesty_triad.csv` (track=embeddings, slice=contested) |
| **Contested-cases lift** | **+0.238** | computed |
| Word-count median | 1,205 words/case (highest on bench) | EDA, `phase5_test_predictions.csv` |
| Speaking rate | 96% of her cases | EDA |
| Test rows in fold-0 | 36 | `phase45_per_justice_auc.csv` |

## Other per-Justice findings

| Justice | Track | Notable | Source |
| --- | --- | --- | --- |
| Thomas | both | global lift +0.193, contested lift +0.282 | `comparative_per_justice.csv` |
| Thomas | — | speaking rate 20.5% (lowest on bench), 295 cases in modeling table | EDA |
| Kennedy | both | global lift −0.101, contested lift −0.128 (only meaningful negative) | `comparative_per_justice.csv` |
| Stevens | both | contested lift −0.190 (small n=74 test rows) | `comparative_per_justice.csv` |
| Barrett | both | global lift +0.118, contested lift +0.208 | `comparative_per_justice.csv` |
| Kavanaugh | both | global lift +0.042, contested lift +0.023 | `comparative_per_justice.csv` |

## Encoder + classifier sizes

| Model | Dim | Cache size | CPU encode time (10K rows) |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 384 | 15.4 MB (.npy) | ~1 minute |
| `all-mpnet-base-v2` | 768 | 29.4 MB (.npy) | ~11 minutes |
| MiniLM model files (HuggingFace) | — | ~80 MB | — |
| MPNet model files | — | ~420 MB | — |

## Compute discipline (per-fit timings)

| Phase | Wall-clock | Notes |
| --- | --- | --- |
| Phase 1 Stop B (bulk Oyez fetch) | 54 min | ≤ 1 req/sec, 2,940 API calls |
| Phase 4 BoW GridSearchCV | 6.9 min | n_jobs=4 + Pipeline memory caching |
| Phase 4.5 baseline sweep (6 combos) | 2.2 min | Dense vectors, fast classifiers |
| Phase 4.5 GridSearchCV | 17.2 min | RBF SVM with `probability=True` is the slow part |
| Total reproducibility from scratch | ~95 min | Dominated by 54-min Oyez fetch |

## Data scale

| Number | Value |
| --- | --- |
| SCDB Justice-vote rows in 2005-2024 window | 13,149 |
| Unique cases in window | 1,471 |
| Unique `(term, docket)` pairs to fetch | 1,470 |
| Cases with valid oral argument | 1,322 |
| Cases successfully parsed | 1,307 |
| **Joined parquet rows** | **10,308** |
| **Final modeling table rows** | **10,039** |
| Distinct cases in modeling table | 1,293 |
| Justices represented | **16** |
| Term range | OT 2005 - OT 2024 (20 terms) |
| Class balance (modeling table) | **62.4% petitioner / 37.6% respondent** |
| Unanimous-case rows | 4,207 (41.9%) |
| Test fold size | **2,007 rows / 258 cases** |
| Train fold size | 8,032 rows / 1,035 cases |

## Stopword + preprocessor bookkeeping

| Number | Value |
| --- | --- |
| Total `STOPWORDS_FOR_VECTORIZER` size | 424 |
| sklearn ENGLISH_STOP_WORDS subset | 318 |
| Custom additions (states, agency abbrevs, famous case shortnames) | 106 |
| Bracket annotations stripped at preprocess time | 1,499 occurrences across 1,078 rows (10.7% of corpus) |

## Tests

| Number | Value |
| --- | --- |
| Total pytest tests passing | 90 |
| Test files | 9 (now 10 with `test_deck_charts.py`) |
| Total notebook execution wall-clock (cached) | < 30 sec for `JusticeCast_Final.ipynb` Restart-Run-All |
