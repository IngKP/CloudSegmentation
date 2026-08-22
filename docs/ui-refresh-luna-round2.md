# CloudSegmentation UI Refresh — Luna Round 2

## Goal

Refine the existing `ui-refresh` implementation. Do **not** redesign it from scratch.

The first pass established the correct three-scene structure and palette, but the page is too vertically spacious, the demo controls feel fragmented, and the visual language is too sharp/boxy.

This pass should make the page:

- more compact on desktop,
- easier to understand as three scenes,
- softer and less edgy,
- more consistent in how the five-color palette is used,
- unchanged in inference behavior.

Work only on the existing `ui-refresh` branch.

Read `AGENTS.md` before editing.

## Hard constraints

Do not modify:

- `app/`
- backend routes
- inference behavior
- model files or manifests
- deployment configuration
- GCP configuration
- rate limiting or caching
- `frontend/app.js` unless a DOM restructuring absolutely requires a tiny compatibility change; prefer preserving all current IDs and avoiding JS changes.

Do not deploy anything and do not merge to `main`.

Keep the five-color site palette as the design system:

- `#0D1B2A` darkest navy
- `#1B263B` secondary navy
- `#415A77` primary blue
- `#778DA9` secondary blue
- `#E0E1DD` light

The scientific visualization colors in the canvases/legends may remain cyan/orange/red because they encode model/error semantics. Do not spread those semantic colors into general UI chrome.

---

# Task 1 — Make Scene 1 fit in one desktop viewport

## Problem

The intro currently requires scrolling to see its full content because the headline, vertical scene padding, and interpretation card together are too tall.

## Required changes

On desktop (roughly >= 900 px wide):

1. Keep the sticky header.
2. Make Scene 1 visually fit inside the remaining viewport below the header at normal laptop/desktop heights such as 864–900 px.
3. Reduce the vertical padding substantially. The current `clamp(76px, 10vw, 144px)` is far too large for viewport-based scenes.
4. Reduce the maximum hero heading size from the current 92 px range. Aim for roughly 68–76 px maximum on a ~1774 px-wide viewport.
5. Keep the hero large, but make it possible to see all of these without scrolling:
   - eyebrow
   - full heading
   - lede
   - CTA
   - technology meta line
   - short interpretation note
6. Do not force this viewport-fitting behavior on small mobile screens. Mobile may scroll naturally.

Suggested desktop direction:

```css
.scene-intro {
  min-height: calc(100svh - 72px);
  display: flex;
  align-items: center;
}

.scene-intro .scene-inner {
  padding-block: clamp(40px, 6vh, 68px);
}

.scene-intro h1 {
  font-size: clamp(46px, 5.3vw, 76px);
}
```

These values are guidance, not requirements; adjust after viewing the page.

---

# Task 2 — Replace the interpretation card with a small note

## Problem

The current interpretation note occupies an entire boxed card and visually competes with the hero.

## Required changes

Remove the large boxed `aside.caveat` treatment.

Use a small secondary note instead, integrated into the intro copy or placed quietly beside/below it.

Use short copy:

> This demo segments surface water; it does not distinguish transient floodwater.

Do not mention probability calibration here. That information already appears where the model score is shown and does not need to compete with the landing message.

The note should:

- use a smaller font than the lede,
- use `#778DA9` or a similarly muted palette role,
- have no large filled box,
- optionally use a subtle left border or small info marker,
- not occupy a separate large grid column.

After this change, reconsider whether `intro-layout` needs two columns at all. A clean single-column hero is acceptable if it produces better balance.

---

# Task 3 — Combine scene selection and inference controls

## Problem

Scene 2 currently has too many layers:

- `Try the model`
- `Curated scenes`
- `Choose a scene`
- `Inference explorer`
- `See the signal`

This makes one workflow feel like several separate sections. The scene cards are also much taller than necessary.

## Required result

Scene 2 should read as one workflow:

**Try the model → choose scene → run inference → adjust threshold → inspect results**

Do not use separate major visual sections for “Choose a scene” and “See the signal”.

### Recommended structure

Keep the main Scene 2 heading:

- eyebrow: `Interactive demo`
- H2: `Try the model`
- one short explanatory sentence

Immediately below it, create one compact control area containing:

1. scene selection
2. run button / status
3. threshold control

Then show metadata and result figures.

### Scene selector

Make the two sample cards compact.

Target desktop height: approximately **64–76 px**, not 112 px.

Remove redundant large `Choose a scene` heading. A small label such as `Scene` is enough.

Keep both samples visible side by side on desktop.

Each card needs only:

- sample name
- small country/split line
- selection state

The arrow is optional; keep it only if it still looks useful at the smaller size.

### Inference heading

Remove the separate large `See the signal` heading and `Inference explorer` eyebrow.

`run-state` should remain visible near the run button or control row.

### Threshold

Keep the threshold control prominent enough to communicate interactivity, but reduce its overall height.

Keep a short explanation, for example:

> Updates locally without rerunning inference.

Do not use the current full sentence if a shorter version is clearer.

### DOM compatibility

Preserve these IDs exactly:

- `sample-count`
- `gallery`
- `workspace`
- `run-state`
- `run-button`
- `threshold`
- `threshold-value`
- `error`
- `scene-label`
- `model-label`
- `latency-label`
- all canvas IDs
- all metric IDs

The HTML hierarchy may change as long as `frontend/app.js` continues to work.

---

# Task 4 — Improve palette roles

## Problem

The palette colors are technically present but their roles are not fully coherent.

Use the palette more intentionally:

### `#0D1B2A`
Use for:

- navigation/header
- primary action button
- deepest canvas/background surfaces

### `#1B263B`
Use for:

- Scene 3 background
- result figure/card surfaces
- selected scene card if it gives clearer hierarchy

### `#415A77`
Use for:

- stronger borders
- interactive emphasis
- secondary controls
- hover/focus states

### `#778DA9`
Use for:

- muted labels on dark backgrounds
- secondary borders
- small metadata

### `#E0E1DD`
Use for:

- Scene 2 light background
- main text on dark surfaces

Do not add orange/red/cyan to general page UI just because those colors exist for the scientific visualizations.

---

# Task 5 — Fix the per-scene diagnostic note

## Problem

`.metric-note` currently uses the semantic orange (`--orange`) and therefore looks disconnected from the site palette.

## Required changes

Restyle `.metric-note` using only the five-color site palette.

For example:

- border or accent: `#415A77`
- subtle background tint derived from `#778DA9` / `#415A77`
- text: `#0D1B2A` / `#415A77`

Keep the existing factual text unchanged unless a very small wording/layout adjustment is needed.

This notice is informational, not an error/warning.

---

# Task 6 — Make the design less sharp

## Problem

The first pass uses `border-radius: 2px` for most UI elements, which creates an overly rigid/edgy appearance.

## Required changes

Introduce consistent radius tokens, for example:

```css
--radius-sm: 6px;
--radius-md: 10px;
--radius-lg: 14px;
```

Use roughly:

- buttons / links: 7–9 px
- sample cards: 9–12 px
- threshold control: 9–12 px
- result figures: 9–12 px
- stats container: 9–12 px
- notices: 8–10 px

Do not make everything pill-shaped. The goal is softer corners, not a bubbly UI.

The canvases themselves may remain square inside their rounded figure cards, or receive only a very small radius if clipping looks cleaner.

Also consider slightly softer borders where many boxes sit next to each other; not every element needs a high-contrast outline.

---

# Task 7 — Make Scene 3 fit in one desktop viewport

## Problem

Model Context currently needs scrolling to see the full section and GitHub link.

## Required changes

At desktop/laptop sizes around 864–900 px high, aim to show the complete Model Context section without internal scrolling:

- eyebrow
- heading
- short paragraph
- `Research vs deployment`
- complete 5-row table
- GitHub CTA

Reduce:

- vertical `scene-inner` padding
- `scene-heading` bottom margin
- table row padding if necessary
- excessive spacing above the GitHub link

Keep the table readable; do not compress it excessively.

Recommended direction for Scene 3 only:

```css
.scene-model {
  min-height: calc(100svh - 72px);
  display: flex;
  align-items: center;
}

.scene-model .scene-inner {
  padding-block: clamp(40px, 5vh, 64px);
}
```

The footer may sit below the scene. The priority is that the **Model Context content itself** fits cleanly in the viewport.

Again, mobile may scroll naturally.

---

# Task 8 — Results are allowed to scroll

Do **not** try to force the six 512×512 result figures into one viewport.

The results area is intentionally the long section of the page.

Keep:

- 3-column result grid on wide desktop where practical
- 2 columns at intermediate widths
- 1 column on mobile
- all six visualizations
- metrics
- diagnostic note

Prioritize readability over viewport fitting here.

---

# Task 9 — Verify visually at specific sizes

Run the app locally using the repository workflow:

```bash
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Verify at least these viewport sizes if browser tooling is available:

1. **1774 × 864** — approximately matches the review screenshots.
2. **1440 × 900** — common desktop/laptop check.
3. **390 × 844** — mobile check.

At both desktop sizes verify:

### Scene 1

- complete hero visible without needing to scroll within the scene
- hero heading is still visually dominant
- short interpretation note does not compete with hero

### Scene 2 controls

- scene selector is compact
- selection + run + threshold read as one workflow
- no duplicate major headings for selection/inference
- results remain readable below

### Scene 3

- complete model context content fits in one viewport if reasonably possible
- table is readable
- GitHub CTA is visible

### General

- corners are visibly softer than the current 2 px treatment
- palette roles look consistent
- orange/red/cyan are restricted to scientific/semantic visualization use
- no horizontal overflow at any tested viewport
- sticky nav links still jump to the correct scenes

---

# Task 10 — Functional verification

Confirm that the UI redesign did not change inference behavior.

Verify:

1. `/samples` populates both scene choices.
2. First sample still auto-selects as before.
3. Clicking the other scene updates selection and SAR/reference loading.
4. `Run inference` works.
5. Model score, binary prediction, overlay, reference, and error canvases render after inference.
6. Moving the threshold rerenders locally without a new `/predict` request.
7. IoU / precision / recall / valid pixels update.
8. GitHub links work.
9. Demo and Model navigation links work.

Run repository checks:

```bash
uv run --locked pytest
uv run --locked ruff check .
```

Do not deploy.

---

# Final report

When finished, report only:

1. files changed,
2. key visual changes,
3. desktop viewport results for 1774×864 and 1440×900,
4. mobile result for 390×844,
5. inference interaction verification,
6. `pytest` result,
7. `ruff` result,
8. anything still imperfect or requiring human visual judgment.

Do not merge to `main` and do not deploy.