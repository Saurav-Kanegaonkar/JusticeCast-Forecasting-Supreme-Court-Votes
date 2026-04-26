# JusticeCast — Project Proposal

**Course Project — Option 1 (custom topic), Part A + Part B**
**Proposal date:** 2026-04-26 — **Due:** 2026-05-07 — **Final due:** 2026-05-28
**Team:** 6 members

---

## 1. Project name

**JusticeCast** — *Forecasting Supreme Court Votes from Oral-Argument Questions*

## 2. Business problem

Appellate litigators spend hours after every Supreme Court oral argument
"reading the bench" — inferring, from the questions Justices ask, how each
Justice is likely to vote. The inference is currently done by gut intuition
and senior-partner experience. Legal-tech vendors (Lex Machina, Bloomberg
Law, Westlaw Edge, SCOTUSblog) monetize adjacent prediction products
(judge-by-judge ruling history, motion-success rates), but **no widely
available product turns the verbatim transcript of an oral argument into a
per-Justice vote forecast**. The opportunity is to convert what every
litigator does informally into a measurable, repeatable signal.

## 3. Proposed solution

A binary text-classification system. **Input**: the concatenated text of all
questions a single Justice asks during a single case's oral argument.
**Output**: probability that the Justice will vote with the petitioner
(positive class) vs. with the respondent (negative class). Unit of analysis
is one `(case, Justice)` row.

Productized, this delivers three concrete legal-tech artifacts:

1. **Pre-argument prep tool** — given prior cases, recommend which Justices'
   questioning patterns to anticipate.
2. **Post-argument forecast** — within hours of an argument, deliver a
   per-Justice vote forecast for amicus brief authors and litigation press.
3. **Historical bench-reading benchmark** — quantify how predictable each
   Justice has been over time, by topic and by court composition.

## 4. Framing under Option 1: stance classification (not sentiment)

The course rubric mentions "Sentiment Analysis" because **Option 2** is
sentiment classification. Our project is **Option 1 (custom topic)**, and
we frame the task as **stance classification** toward the petitioner's
position. The machinery — bag-of-words, TF-IDF, n-grams, linear and
ensemble classifiers — is identical to sentiment work; only the label
definition changes. If the professor prefers a sentiment framing, the
fallback is "stance toward petitioner = positive sentiment toward the
petitioner's argument," which preserves the rubric mapping without altering
the model.

## 5. Data sources (verified live, 2026-04-26)

- **Supreme Court Database (SCDB)**, Washington University —
  scdb.wustl.edu. Latest release **2025_01**. Justice-Centered file:
  83,644 vote rows × 61 columns. Provides every recorded SCOTUS vote with
  petitioner/respondent winner, majority/dissent flags, and case metadata.
  Free CSV.
- **Oyez.org API** — `https://api.oyez.org/`. Case metadata at
  `/cases/{term}/{docket}`; oral-argument transcripts (with each utterance
  tagged by speaker name and `roles[].type == "scotus_justice"`) at the
  linked `/case_media/oral_argument_audio/{audio_id}` endpoint. Free,
  public, no auth. We will rate-limit at ≤ 1 request/second.

The two sources join on `(term, docket_number)`. Final modeling table:
one row per `(case_id, justice_id, concatenated_question_text, vote_label)`
plus metadata columns including a `unanimous` flag.

## 6. Method — direct mapping to rubric requirements

| Rubric requirement                       | Implementation in JusticeCast                                                          |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| Bag-of-Words vectorization               | `CountVectorizer(ngram_range=(1,1))`                                                   |
| TF-IDF vectorization                     | `TfidfVectorizer(ngram_range=(1,1))`                                                   |
| n-grams                                  | `TfidfVectorizer(ngram_range=(1,2))` baseline; `(1,3)` searched in tuning              |
| Three classifiers                        | `LogisticRegression`, `LinearSVC`, `RandomForestClassifier` — all with `class_weight='balanced'` |
| GridSearchCV                             | Sequential strategy: Stage 4A jointly tunes the two linear models with vectorizer params; Stage 4B fixes the best vectorizer and tunes RF only (avoids a 1,600+ fit blowup on RF × bigrams) |
| Train/test discipline                    | `StratifiedGroupKFold(n_splits=5, random_state=42)`, `groups=case_id`. Fold 0 = test, folds 1–4 = train. Vectorizers live inside `Pipeline` so vocabulary is built on train fold only. |
| Evaluation                               | Confusion matrix, precision, recall, F1, ROC AUC, balanced accuracy, ROC curve, PR curve, calibration curve, per-Justice breakdown, **per-Justice metrics split by unanimous vs contested cases** |
| Business interpretation (FN vs FP cost)  | Discussed in notebook prose: FP = over-prepares for a petitioner-leaning Justice (wasted prep, mistargeted amicus); FN = under-prepares for a sympathetic Justice (lost opportunity). Asymmetry informs whether to tune the operating threshold. |
| ML Canvas v0.4                           | Filled across Goal / Predict / Learn / Evaluate quadrants; exported to `reports/ml_canvas.pdf` |
| Pitch deck                               | 8–12 slides, problem → insight → market → solution → ML Canvas → data → approach → results → recommendations |

## 7. Engineering non-negotiables

1. **No data leakage** — all of a case's Justices stay in the same fold.
2. **Vectorizers fit on train only** — enforced by `Pipeline`, not discipline.
3. **No post-hoc features** — only information available at the moment the
   Justice finished speaking. The vote label is the only future signal we
   touch. No decision text, no opinion text, no outcome-derived features.
4. **Reproducibility** — fixed seed 42, pinned dependencies, notebook runs
   top-to-bottom on a fresh kernel via `Restart & Run All`.
5. **Class imbalance handled explicitly** — petitioner-side wins ~65–70%
   of SCOTUS cases. `class_weight='balanced'`; report ROC AUC and balanced
   accuracy alongside raw accuracy.
6. **Every experiment logged** — `reports/results/*.csv` rows for each
   (vectorizer, classifier, hyperparams) combination including per-fit
   wall-clock time.

## 8. Deadline structure

- **2026-05-07** — proposal submitted to professor (this document).
- **2026-05-21** *(target, internal)* — modeling complete, notebook
  near-final, pitch deck draft.
- **2026-05-28** — both deliverables submitted to Canvas: polished
  reproducible notebook (`JusticeCast_Final.ipynb`, Part B, 20 pt) and
  pitch deck (`JusticeCast_Pitch.pdf`, Part A, 15 pt).

## 9. Risks and mitigations

- **Oyez transcript coverage gaps for older terms.** The window is
  provisional at 2005–2024; an empirical coverage check in Phase 1 will
  set the final window.
- **Oyez API politeness.** Rate-limited at ≤ 1 request/second with on-disk
  caching, so reruns hit the cache, not the API.
- **Compute budget for Random Forest tuning on TF-IDF bigrams.** Avoided a
  1,600+ fit blowup by separating Stage 4B (RF only, fixed vectorizer)
  from the cheap linear-model joint search in Stage 4A. Per-fit timings
  from the Phase 3 baseline sweep set the actual budget.
- **SCDB label semantics.** `partyWinning` and `majority` codebook
  semantics will be verified against the official SCDB codebook before
  the binary label is locked in.

## 10. Sign-off requested from professor

We request explicit sign-off on the **Option 1 stance-classification
framing** so that the rubric's "sentiment analysis" wording does not
penalize a substantively richer task. The pitch deck (Part A) and notebook
(Part B) will both make the framing decision visible.
