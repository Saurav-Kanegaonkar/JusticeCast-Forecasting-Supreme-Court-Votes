# JusticeCast Deck Theme Spec

## Palette

| Role | Hex | Usage |
| --- | --- | --- |
| Primary navy | `#1A2E47` | Slide titles, header bar fill, accent panels, axis labels |
| Accent gold | `#C9A961` | Headline numbers, dividers, callouts, Embeddings track in charts |
| Background cream | `#FAF7F2` | Slide background — warm off-white, NOT pure white |
| Text dark | `#1A1A1A` | Body text |
| Text grey | `#5A5A5A` | Italic subtitles, secondary text |
| Text light | `#9A9A9A` | Slide numbers, footers |
| Data blue | `#2E5C8A` | BoW track in charts |
| Random baseline grey | `#888888` | Dashed reference lines (chance / baseline) |

The palette is consistent across every slide and every chart asset in this bundle. Charts have already been rendered with these colors locked in via `src/build_deck_charts.py`.

## Typography

- **Serif (titles, headline numbers, deck name):** Playfair Display, Lora, EB Garamond, or any contemporary serif with judicial gravitas. The PowerPoint extension may pick whichever serif is available on the host system; pick one and use it consistently across all slides.
- **Sans-serif (body, axis labels, footers, bullets):** Inter, Source Sans Pro, Helvetica Neue, or any modern humanist sans. Conveys readability + analytical rigor.

The serif/sans pairing should evoke "courthouse" + "modern consultancy."

## Layout density

**Balanced.** Each content slide has either:
- one chart + a supporting prose panel, OR
- two compact prose blocks side by side, OR
- one centered statement / case-study card with surrounding white space.

Do not pack slides full of text. Do not use sparse single-line slides either (the deck must read standalone for graders who weren't in the room when it was presented).

## Standard elements per slide

Every content slide (slides 2–10) carries the same chrome so the deck reads as one document:

- **Header bar** at top: thin (~0.30") navy fill, with section label in gold uppercase serif (e.g., "THE COMPARATIVE FINDING")
- **Gold accent line** immediately below the header bar (~1pt thick)
- **Slide title** in navy serif, ~28pt, weight bold
- **Slide subtitle / lede** (optional) in grey serif italic, ~14pt
- **Body** in dark text, sans-serif, ~11pt for prose; bullets allowed
- **Footer left:** "JusticeCast" in light grey italic serif, ~9pt
- **Footer right:** slide number "N / 11" in light grey sans-serif, ~9pt
- **Slide 1 (title)** is the exception — full-bleed navy, no header bar, gold headline serif, gold horizontal divider
- **Slide 11 (outro)** mirrors Slide 1's full-bleed treatment for visual bookend

## Tone

Consultancy / business-pitch register. Confident but not flashy. Avoid:
- emoji (NEVER)
- exclamation marks
- corporate jargon ("synergy", "leverage")
- gratuitous animations or transitions

Use straightforward declarative claims supported by data. The KBJackson centerpiece on Slide 8 is the one moment that benefits from emphasis: render the lift number (`+0.238`) in large gold serif so it carries visually.

## Image rules

All chart PNGs in this bundle are pre-rendered at 16:9-compatible aspect ratios with the locked palette. **Do not re-style them.** Drop them in at full slide width or roughly half-width depending on the layout in `slide_content_spec.md`.

The ML Canvas summary PNG is intentionally dense (12 boxes); use full slide width for that one.

`data_flow_diagram.png` is the original network diagram of SCDB ↔ Oyez ↔ joined parquet ↔ modeling table — drop it in at full width on Slide 4.
