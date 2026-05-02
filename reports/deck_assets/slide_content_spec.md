# JusticeCast Pitch Deck — Per-Slide Content Spec

11 slides. Each section below is the lay-out instruction for one slide. The PowerPoint Claude extension should follow these specs strictly.

Theme constants come from `theme_spec.md`. Numbers come from `headline_numbers.md`. Visual assets are PNGs in this directory.

---

## Slide 1: Title

**Layout type:** title slide (full-bleed navy)
**Section label (header bar):** none — title slide has no header bar
**Slide title (large gold serif, centered, ~54pt):**
> JusticeCast

**Subtitle (cream serif italic, centered, ~22pt):**
> A Comparative Study of Text Representations
> for Predicting Supreme Court Justice Votes from Oral-Argument Questions

**Visual asset:** none
**Body content:** below the subtitle, three-line metadata block in light grey sans-serif (~14pt):
> Course: BAX 453
> Team: [team members — fill in]
> Date: 2026-05-28

**Speaker notes / takeaway:** Open the deck. Pause. Set up that this is a *methodology study*, not a single-track null result.

---

## Slide 2: The Hypothesis

**Layout type:** two-column-prose with case-study card
**Section label (header bar):** "THE HYPOTHESIS"
**Slide title:** *For 200 years, litigators have read the bench by gut.*
**Subtitle:** Can a model do it from the words alone?

**Visual asset:** none — left column is a "case study card" rendered as a small navy-bordered panel with cream fill

**Body content:**

LEFT COLUMN (case study card, ~40% width, navy border, cream fill, gold corner accent):

> **Citizens United v. FEC, 2009**
>
> Argued twice — March 2009 (Stewart for the FEC) and re-argued September 2009 (Kagan for the FEC). Justices peppered both sides with questions about corporate personhood, statutory limits on issue ads, and stare decisis.
>
> Decided 5–4 in January 2010.
>
> *Could a litigator have read this 5–4 split from the questions on the bench? Could a model?*

RIGHT COLUMN (three flowing prose blocks, ~55% width):

> Litigator intuition is real. Senior partners spend hours after every argument inferring the bench's leanings from the texture of the questioning. Junior associates absorb this as a craft.

> Legal-tech firms monetize adjacent products — Lex Machina, Bloomberg Law, Westlaw Edge, SCOTUSblog all sell judge-by-judge ruling history and motion-success rates.

> But no widely-available product converts oral-argument transcripts into per-Justice vote forecasts. We tested whether one *could*.

**Speaker notes / takeaway:** "Citizens United" sets up a famous, recognizable case. The deck will return to it on Slide 11.

---

## Slide 3: The Data

**Layout type:** chart-and-panel
**Section label (header bar):** "THE DATA"
**Slide title:** Two free public sources, joined at the case docket
**Subtitle:** Twenty SCOTUS terms, sixteen Justices, ten thousand utterance rows

**Visual asset:** `chart_data_pipeline_funnel.png` (centered, ~75% slide width)

**Body content (compact panel below or beside the chart):**

- **SCDB** (Supreme Court Database, Wash. U.) — release 2025_01, Justice-Centered file, 83,644 vote rows × 61 columns. Latin-1 encoded. Provides every recorded SCOTUS vote with petitioner/respondent winner, majority/dissent flags.
- **Oyez.org** REST API — two-step fetch (case metadata → transcript audio JSON). Polite limit ≤ 1 request/sec.
- **No leakage:** `StratifiedGroupKFold(n_splits=5, random_state=42)` with `groups=case_id` — all of a case's Justices stay in the same fold. Codified in `src/modeling/splits.py`.
- **Custom calibrated stopword list** preserves thematic legal vocabulary while filtering case-identity terms.

**Speaker notes / takeaway:** Anchor the audience: this is empirical, the data discipline is real, the splits don't leak.

---

## Slide 4: Two Modeling Tracks

**Layout type:** chart-and-panel
**Section label (header bar):** "THE COMPARATIVE DESIGN"
**Slide title:** Two parallel tracks, identical evaluation harness

**Visual asset:** `data_flow_diagram.png` (centered, full slide width)

**Body content (compact side panel, right side):**

- **Track 1 — BoW (rubric-required):**
  3 vectorizers (Count, TF-IDF unigram, TF-IDF bigram) × 3 classifiers (LogReg, LinearSVC, RandomForest) + GridSearchCV.
- **Track 2 — Embeddings (methodologically appropriate alternative):**
  Pre-trained `all-MiniLM-L6-v2` and `all-mpnet-base-v2`, no fine-tuning, same three classifier families + GridSearchCV.
- **Both tracks use the identical fold-0 test rows** (Non-Negotiable #15). Same 2,007 rows, 258 cases.

**Speaker notes / takeaway:** Apples-to-apples is the methodological backbone. Without it, the comparison is meaningless.

---

## Slide 5: ML Canvas

**Layout type:** full-bleed image
**Section label (header bar):** "THE PRODUCT FRAMING"
**Slide title:** Machine Learning Canvas, CRISP-DM-tagged

**Visual asset:** `ml_canvas_summary.png` (centered, ~92% slide width)

**Body content:** small caption below the canvas (~11pt grey italic):

> Each box tagged with its corresponding CRISP-DM phase (Business Understanding / Data Understanding / Data Preparation / Modeling / Model Evaluation / Model Deployment). Predict (left two columns) and Learn (right two columns) Canvas split visible in the background panels.

**Speaker notes / takeaway:** This is the slide that demonstrates rubric alignment without dwelling. The PDF version of the canvas is `reports/ml_canvas.pdf`.

---

## Slide 6: BoW Results — A Real Ceiling

**Layout type:** chart-and-panel
**Section label (header bar):** "TRACK 1 — BAG OF WORDS"
**Slide title:** Tuning the standard toolkit hits a ceiling at AUC 0.532
**Subtitle:** Tuning lift was +0.4 pp. Bigrams and trigrams added nothing.

**Visual asset:** `chart_bow_baselines.png` (left side, ~58% slide width)

**Body content (right panel, ~37% slide width):**

- All 9 baseline (vectorizer × classifier) combinations land **0.51–0.53**.
- Tuning lifted AUC from **0.528 → 0.532** (+0.4 pp). Exactly the cai-plan's predicted ceiling.
- **Best vectorizer config: unigrams**, `min_df=5`, `max_df=0.9`. Bigrams and trigrams added nothing — signal is in individual words, not phrases.
- Best classifier: **LinearSVC** with `C=0.01` (heavy regularization). On a 200K-feature problem, the signal is so weak almost any flexibility overfits.
- **Only 2 of 15 Justices** (Kennedy, Alito) had a bootstrap 95% CI lower bound above 0.5 — i.e., statistically distinguishable from chance for just two Justices.
- Top features after tuning are still **thematic legal vocabulary** (officer, jury, sentence, religious, fraud), not stance markers.

**Speaker notes / takeaway:** This is the rubric requirement, executed rigorously, and the result is a defensible null. Don't apologize for it.

---

## Slide 7: Embeddings Results — Lightweight MiniLM Wins

**Layout type:** chart-and-panel
**Section label (header bar):** "TRACK 2 — SENTENCE EMBEDDINGS"
**Slide title:** Lightweight MiniLM beats tuned BoW by 3.7 pp
**Subtitle:** 384-dim, 80 MB, no fine-tuning — and it wins on the same test rows

**Visual asset:** `chart_embeddings_baselines.png` (left side, ~58% slide width)

**Body content (right panel, ~37% slide width):**

- Best classifier × encoder: **LogReg + MiniLM-L6-v2**, `C=100`, `l1_ratio=1.0` (= L1).
- Test ROC AUC **0.5691**. Tuning lift on the embeddings track was small; the encoding itself is most of the work.
- **Lightweight wins:** MiniLM (384-dim, ~80 MB) edged MPNet (768-dim, ~420 MB) by 0.003 — within noise but selected for the 10× encoding-speed advantage.
- **Every embedding combination beats the BoW baseline best.** Even the worst-performing emb combo lands above the dotted BoW reference line in the chart.

**Speaker notes / takeaway:** Don't say "embeddings beat BoW." Say "lightweight pre-trained embeddings, with no fine-tuning, beat a tuned 200K-feature TF-IDF + LinearSVC by 3.7 percentage points." That's the sharper framing.

---

## Slide 8: The Comparative Finding (HEADLINE)

**Layout type:** two-column with KBJackson spotlight
**Section label (header bar):** "THE COMPARATIVE FINDING"
**Slide title:** The embedding lift survives the strict contested-cases test
**Subtitle:** Same fold-0 test set. Same Justices. The gap is real bench-questioning signal, not author-identity recovery.

**Visual asset (top half, full width):** `chart_bow_vs_embeddings_3slice.png` — three grouped bar pairs (Global, Unanimous, Contested), BoW blue + Embeddings gold, lift annotations above each pair.

**Visual asset (bottom-right, ~35% slide width):** `chart_kbjackson_flip.png` — single-Justice spotlight with the big gold `+0.238` lift number.

**Body content (bottom-left navy panel, ~60% slide width, four headline numbers stacked):**

> **+3.7 pp** &nbsp;&nbsp;Test ROC AUC, embeddings over tuned BoW
>
> **+4 pp** &nbsp;&nbsp;Per-Justice mean AUC on contested cases (the strict test)
>
> **13 of 15** &nbsp;&nbsp;Justices above chance with embeddings on contested (BoW: 9/15)
>
> **80 MB** &nbsp;&nbsp;MiniLM encoder size — no fine-tuning required

**Footer prose (small italic grey, full width below the numbers):**

> KBJackson — the most-engaged questioner on the bench (median 1,205 words/case, 96% speaking rate) — is the deck's centerpiece anecdote. With BoW her contested AUC is 0.405, the worst on the bench. With embeddings: 0.643, the third best. Same Justice, same data, opposite verdicts on predictability.
>
> Stevens and Kennedy regress — not every Justice improves with embeddings. We report the mixed pattern honestly.

**Speaker notes / takeaway:** This is THE slide. Spend 90 seconds here. The contested-cases lift is what makes the project not just a numbers contest but a rigorous methodological claim.

---

## Slide 9: What This Means for Legal-Tech

**Layout type:** three-block prose
**Section label (header bar):** "PRODUCT IMPLICATIONS"
**Slide title:** Don't sell a TF-IDF question-classifier
**Subtitle:** Three concrete recommendations for legal-tech product strategy

**Visual asset:** none — three numbered prose blocks

**Body content:**

> **1. Pre-trained semantic embeddings are the necessary baseline.**
> The BoW ceiling is real and replicable. Any legal-tech product in this space that ships with TF-IDF + linear classifiers as its foundation is leaving 3-4 percentage points of AUC on the table. In a domain where 3-4 pp translates to material business decisions, that gap is product-defining.

> **2. The marginal cost over BoW is small. The payoff is access to semantic structure that lexical features cannot reach.**
> Encoding the modeling-table corpus with `all-MiniLM-L6-v2` takes ~12 minutes on CPU. The model file is 80 MB. There's no fine-tuning, no GPU dependency, no architectural complexity. The bar to ship a semantically-aware version is low.

> **3. Tone, sequence, and audio are the next product frontier.**
> The Oyez .mp3 files contain the signal litigators actually pick up on — *how* a question is asked, not just what words appear in the transcript. Sequence-aware models on full transcripts (turn-taking, who interrupts whom) and multimodal audio features are unambiguous next steps. We've established the empirical baseline they need to beat.

**Speaker notes / takeaway:** The audience for this slide is the legal-tech founder, not the academic. Speak to product strategy.

---

## Slide 10: Methodological Recommendations & Honest Caveats

**Layout type:** two-column-prose
**Section label (header bar):** "RECOMMENDATIONS & CAVEATS"
**Slide title:** A lower bound, with concrete next directions

**Visual asset:** none

**Body content:**

LEFT COLUMN (~48% width):

> **What we can honestly claim**
>
> 1. Pre-trained sentence embeddings extract more vote-relevant signal than the standard TF-IDF/linear-classifier toolkit (+3.7 pp ROC AUC overall, +4 pp on contested cases).
> 2. The per-Justice gain is broadly distributed but concentrated in specific personalities (Thomas +0.193, Barrett +0.118, KBJackson +0.229 globally).
> 3. The KBJackson flip — same Justice, same data, opposite predictability conclusions — is the project's sharpest single finding.
> 4. Stevens and Kennedy regressions are real and reported. Not every Justice improves with embeddings.

RIGHT COLUMN (~48% width):

> **What we cannot claim — yet**
>
> Absolute AUC of 0.569 is modest. We're reporting a *lower bound* on bench-reading from text alone. Out-of-scope frontiers that legal-tech teams should pursue:
>
> - **Fine-tuned Legal-BERT** on a SCOTUS oral-argument corpus (+3-5 pp plausible)
> - **Sequence-aware transformers** on full transcripts — capture turn-taking, who interrupts whom (+2-4 pp plausible)
> - **Multimodal audio features** from the Oyez .mp3s — tone, pace, hesitation (potentially substantial)
> - **Structured case features** — issue area, lower-court holding, prior voting record (+1-3 pp plausible)

**Speaker notes / takeaway:** The honest framing is the strong framing. Don't oversell the result.

---

## Slide 11: Outro

**Layout type:** full-bleed navy (mirrors Slide 1)
**Section label (header bar):** none
**Slide title (centered, gold serif, ~36pt):** Citizens United, again

**Visual asset:** none

**Body content (cream serif, centered, ~18pt):**

> Citizens United, January 2010 — five Justices voted with the petitioner, four voted against. Litigators read those nine personalities through the questions they asked at oral argument. They were right.
>
> &nbsp;
>
> *Litigators have read the bench by gut for two hundred years.*
>
> *We tested two computational approaches: the standard one, and the methodologically-appropriate one.*
>
> *The gap between them tells us where the real signal lives.*
>
> *That's the actionable finding.*

**Speaker notes / takeaway:** Bookend with Citizens United. Close clean. Q&A.

---

## Final notes for the extension

- All chart PNGs in this directory are pre-rendered with the locked theme. Do not re-style.
- Do NOT invent numbers. Every numeric claim in this spec is backed by `headline_numbers.md` or a chart asset.
- Do NOT add slides beyond these 11. Do not split slides. The pacing is calibrated.
- Export the assembled deck to **both** `.pptx` and `.pdf`. Save the PDF as `reports/JusticeCast_Pitch.pdf`.
