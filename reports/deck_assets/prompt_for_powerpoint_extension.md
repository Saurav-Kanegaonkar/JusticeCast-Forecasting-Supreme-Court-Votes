# Master Prompt for the PowerPoint Claude Extension

Paste the block below verbatim into the PowerPoint Claude extension. Everything the extension needs is already in this `reports/deck_assets/` directory. **Do not edit the prompt before pasting** — it's been calibrated against the asset bundle.

---

## Prompt (copy from here)

I need you to assemble an 11-slide pitch deck for an academic course project called **JusticeCast**. The full asset bundle for the deck is in this directory (`reports/deck_assets/`). You have everything you need — do not invent numbers, content, or visuals beyond what's specified in the bundle.

**Goal:** produce a single `.pptx` file that I can open in PowerPoint and edit further if needed, plus an exported `.pdf` rendering that's ready for course submission. Save the PDF as `reports/JusticeCast_Pitch.pdf`.

**Source-of-truth files in this directory (read them in this order):**

1. `theme_spec.md` — the locked visual identity. Palette (deep navy `#1A2E47`, warm gold `#C9A961`, cream `#FAF7F2` background), typography (serif for titles + headline numbers, sans-serif for body), layout density rules, standard slide chrome (header bar, gold accent line, footers).

2. `slide_content_spec.md` — per-slide content for all 11 slides. Each section names the layout type, the section label that goes in the header bar, the slide title, the subtitle, the visual asset(s) to use, the body content, and a takeaway line. **Follow these specs strictly.** Do not split slides, do not add slides, do not skip slides.

3. `headline_numbers.md` — every numeric claim that appears in the deck, sourced from the project's result CSVs. **Do not generate any number that isn't in this file.** If a number you need isn't in the file, stop and ask me; do not improvise.

4. The eight chart PNGs in this directory:
   - `chart_bow_vs_embeddings_3slice.png` — Slide 8 centerpiece (top half)
   - `chart_per_justice_lift.png` — supporting chart (use only if Slide 8 needs more visual; otherwise omit)
   - `chart_kbjackson_flip.png` — Slide 8 spotlight (bottom-right)
   - `chart_bow_baselines.png` — Slide 6
   - `chart_embeddings_baselines.png` — Slide 7
   - `chart_data_pipeline_funnel.png` — Slide 3
   - `data_flow_diagram.png` — Slide 4
   - `ml_canvas_summary.png` — Slide 5

   All eight PNGs are pre-rendered with the deck's locked palette and theme. **Drop them in as-is. Do not re-style, re-color, or re-render them.**

**Build steps:**

1. Read `theme_spec.md` first. Extract the palette, typography preferences, layout density rules, and standard slide chrome (header bar, gold accent line, footers, slide numbering).

2. Apply that theme as the master slide layout in PowerPoint. Use the serif font for slide titles and big headline numbers, sans-serif for body, axis labels, and footers. Cream `#FAF7F2` background on every slide.

3. For each of the 11 slides, follow the layout instructions in `slide_content_spec.md`. The spec specifies which chart asset (if any) goes on which slide and roughly how to size it (full width / half width / focused panel).

4. Slides 1 and 11 are full-bleed navy with no header bar — they bookend the deck. Slides 2-10 have the standard chrome (navy header bar + gold accent line + footers).

5. The headline slide is **Slide 8**. Spend extra care on its layout — it's a two-region composition with the comparative chart on the top half, four big headline numbers in a navy panel on the bottom-left, the KBJackson spotlight chart on the bottom-right, and small italic prose at the very bottom. The KBJackson story is the deck's centerpiece anecdote — the visual treatment should reflect that.

6. After the deck is assembled, export to `.pdf` and save as `reports/JusticeCast_Pitch.pdf`. Also keep the `.pptx` source file.

**Hard constraints:**

- **Do not invent content.** Every word and number on a slide must come from `slide_content_spec.md` or `headline_numbers.md`.
- **Do not re-render the charts.** They're in this directory at the right resolution and the right palette already.
- **Do not change the palette or fonts beyond what's in `theme_spec.md`.**
- **No emoji, no exclamation marks, no animations or transitions.** The register is consultancy / business-pitch.
- **11 slides. Not 10. Not 12.** If you find yourself splitting a slide for layout reasons, stop and tell me.

**One judgment call I'm leaving to you:** font picks. `theme_spec.md` recommends serifs (Playfair Display, Lora, EB Garamond — pick one) and sans-serifs (Inter, Source Sans Pro, Helvetica Neue — pick one). Pick the closest available pair on the host system and use them consistently across all slides.

When you're done, give me a 4-line summary: number of slides assembled, file path of the `.pptx`, file path of the `.pdf`, and any decisions you made that diverged from the spec. I'll review and either approve or send specific edits back.

## End of prompt

---

## Notes for the user (NOT for the extension)

- After the extension finishes, open the .pptx in PowerPoint and do a quick eyeball pass for layout issues — the PowerPoint Claude extension is good at applying themes and laying out content, but tight visual composition (especially the four-region Slide 8) sometimes needs a manual nudge.
- If the extension misses any chart asset or invents a number, that's the cue to send a fix-up prompt: cite the chart filename or the row in `headline_numbers.md` that should be used instead.
- The PDF export is the deliverable for Canvas. Save it at `reports/JusticeCast_Pitch.pdf` so the project layout in the README continues to point at it.
- If the extension generates a presenter-notes section, the `Speaker notes / takeaway` line in each section of `slide_content_spec.md` is a good source for those notes.
