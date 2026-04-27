# Phase 6 Polish Pass — Instructions for CC

**Status:** Checkpoint 6 substantively cleared. Final notebook, ML Canvas, and README all shipped. Before Phase 7 (pitch deck), execute a polish pass on `notebooks/JusticeCast_Final.ipynb`. **No analysis changes, no rebuilds, no new compute.** Five surgical edits to sharpen what's already there.

## Why these five

Reviewed the rendered notebook end-to-end. Strong CRISP-DM structure, right charts in the right places for the moments that demand visuals, prose density appropriate. Five spots where the TA's reading experience can be measurably better with under an hour of work.

## The five changes

### 1. Add an executive summary cell at the very top

Insert one new markdown cell **between the title cell (cell 0) and Section 1**. Three to four sentences, no header, just a TL;DR paragraph the TA reads before anything else. Suggested wording (adjust as fits):

> *We built two parallel models predicting SCOTUS Justice votes from oral-argument questions, structured around CRISP-DM. The standard bag-of-words approach (TF-IDF + LinearSVC, tuned via GridSearchCV) hit a ceiling at ROC AUC 0.532. A pre-trained sentence-embeddings approach (MiniLM-L6-v2 + LogisticRegression) reached 0.569, a 3.7 percentage-point lift. The gap survived the strict contested-cases test (+4 pp on cases where the bench was genuinely split, where author-identity-plus-priors recovery cannot account for the lift). The result quantifies how much bench-reading signal lives in semantics versus lexical features, and recommends pre-trained embeddings as the necessary baseline for legal-tech products in this space.*

Do not add a heading — keep it as a standalone paragraph that reads like an abstract. Italicize the whole paragraph if that fits your existing notebook styling.

### 2. Add three charts where the existing tables make the reader work harder than necessary

All three use data already in `reports/results/` — no new computation, just visualization.

**Chart A — Section 2.3 (Coverage profile): pipeline funnel.**
Add a horizontal bar chart showing the row-count funnel: 1,470 cases attempted → 1,420 fetched → 1,322 with valid oral argument → 1,307 parsed → 10,308 joined → 10,039 cleaned. Insert immediately after the existing text wall in 2.3. The visual lets the TA see the data-loss profile in one glance instead of parsing numbers from prose.

**Chart B — Sections 4.1 and 4.3: baseline-sweep bar charts.**
- In Section 4.1, add a horizontal bar chart of the 9 BoW combinations sorted by ROC AUC, color-coded by classifier family. The "all 9 cluster around 0.51-0.53" finding is visceral as a chart, abstract as a printed DataFrame.
- In Section 4.3, add the parallel chart for the 6 embedding combinations sorted by ROC AUC, color-coded by embedding model.

Keep the existing DataFrame outputs — the chart supplements, doesn't replace.

**Chart C — Section 5.1 (Comparative summary): side-by-side bar chart.**
Add a grouped bar chart comparing BoW vs Embeddings on the three headline metrics: test ROC AUC, contested-cases ROC AUC, balanced accuracy. This is the project's headline finding — it deserves a chart, not just a table.

All three charts: titles, axis labels, legends, color-blind-safe palettes, consistent with the existing chart styling.

### 3. Trim Section 6.6 (Reproducibility)

Current section is structurally correct but operationally verbose. The full six-step shell block belongs in `README.md`, not the submission notebook.

**Keep:**
- The deliverable path table (clear, useful, fast to scan)
- The framing of why reproducibility matters

**Replace:**
- The six-step `python -m src.*` shell block

**With:** a one-paragraph operational summary along the lines of:

> *The full pipeline reproduces from a fresh clone in roughly 95 minutes — 54 minutes for the bulk Oyez fetch, 12 minutes for embedding encoding, and approximately 25 minutes for the modeling sweeps and hyperparameter tuning. Cached embeddings and pre-computed result CSVs let this notebook itself execute `Restart & Run All` in seconds without re-running modeling. Full reproduction commands are documented in `README.md`.*

This keeps the integrity signal (we built something reproducible) without rehearsing operational detail the TA does not need.

### 4. Verify Section 1.6 (Rubric mapping) explicitly names every rubric requirement

The TA will read this section first and treat it as the contract for what to grade. Open the section and verify that the following words appear by name in the mapping table:

- "Bag of Words" / "BoW"
- "TF-IDF"
- "n-grams"
- "Logistic Regression" / "LogisticRegression"
- "SVM" or "Support Vector Machine" or "LinearSVC"
- "Random Forest" / "RandomForestClassifier"
- "GridSearchCV"
- "ML Canvas" or "Machine Learning Canvas"
- "confusion matrix"
- "precision"
- "recall"
- "ROC AUC"
- "false positive" / "false negative" or "FN/FP business interpretation"

If any are missing, add a row or extend an existing row to name them explicitly. The TA hunts when terms are paraphrased; the TA finds the section instantly when the rubric vocabulary appears verbatim.

### 5. Sharpen Section 5.6 (Evaluation against business criteria)

This is the most under-rewarded CRISP-DM step in most student work and the easiest "yes they did it" win when done well. Open Section 5.6 and verify two things:

**First, it quotes the success criterion verbatim from Section 1.2.** If Section 1.2 says "ROC AUC meaningfully above the per-Justice majority-class baseline on contested cases," then Section 5.6 should literally repeat that phrase before answering it. Do not paraphrase across the loop-back.

**Second, it answers with concrete numbers, not prose.** The right structure is:

> *Section 1.2 set the success criterion as "[verbatim quote]." Result on contested cases: BoW model met this for 9 of 15 Justices (60%); Embeddings met it for 13 of 15 (87%). The success criterion is partially met — broadly with embeddings, narrowly with BoW. The +4 percentage-point gap on the strict contested-cases test indicates the embedding lift is real bench-questioning signal rather than author-identity-plus-priors recovery.*

If the current section is closer to general-purpose evaluation prose, rewrite it to follow this quote-then-answer-with-numbers pattern.

## What NOT to change

- Do not redo any analysis
- Do not re-encode embeddings
- Do not re-run any GridSearchCV
- Do not change the CRISP-DM section structure (six top-level phases)
- Do not change the ML Canvas (already shipped)
- Do not change the README (already polished)

## Reproducibility check after the polish

- [ ] `Restart & Run All` on `JusticeCast_Final.ipynb` from a fresh kernel — must still succeed end-to-end
- [ ] `pytest` — all 90 tests still green
- [ ] Commit clean

## Stop signal

Brief polish pass — estimated 30 to 45 minutes total. Stop and confirm completion. After this clears, proceed to Phase 7 (pitch deck) per the v12 spec.
