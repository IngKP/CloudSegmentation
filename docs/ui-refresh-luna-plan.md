# UI Refresh Plan for Luna

## Goal

Refresh the CloudSegmentation frontend on branch `ui-refresh` while keeping the implementation simple, preserving all existing inference behavior, and avoiding backend/deployment changes.

This is a portfolio/demo UI refactor, not an application architecture rewrite.

## Non-goals

Do **not**:

- add React, Vue, Svelte, Tailwind, Bootstrap, or any frontend framework
- change FastAPI routes or inference logic
- change GCS staging, model loading, rate limiting, caching, deployment configuration, manifests, or artifacts
- change model-result semantics
- remove any of the six current visualization outputs
- remove the interpretation warning that model scores are not calibrated probabilities
- change the scientific visualization colors in `frontend/app.js` unless required for accessibility; the site palette applies to page chrome, not prediction/error semantics
- deploy to Cloud Run
- merge to `main`

## Allowed files

Primary files:

- `frontend/index.html`
- `frontend/style.css`
- `frontend/legends.css`
- `frontend/sample-status.css`

Only modify `frontend/app.js` if a small UI behavior change is genuinely necessary. Preserve all existing inference/data behavior.

Do not modify backend files unless there is a clearly demonstrated frontend-breaking issue. If such an issue appears, stop and report it instead of changing backend code.

## Required palette

Use this palette for page backgrounds, text, borders, buttons, navigation, and decorative UI:

- `#0D1B2A` — darkest navy
- `#1B263B` — secondary dark navy
- `#415A77` — primary blue
- `#778DA9` — secondary blue
- `#E0E1DD` — light neutral

Create CSS custom properties in `:root` and use them consistently.

Do not force every scientific canvas/legend color into this palette. Existing orange/cyan/red/error colors encode model semantics and should remain distinguishable.

## Overall page structure

Reorganize the page into three visually distinct scenes.

### Scene 1 — Introduction

Purpose: explain immediately what the project is.

Requirements:

- dark background using `#0D1B2A`
- prominent title: `Surface-water segmentation from Sentinel-1 SAR`
- concise supporting copy explaining that this is an interactive deployment of a cross-modal Swin-UNet for segmenting surface water from satellite radar imagery
- keep the interpretation warning visible but secondary to the main title
- include a simple call-to-action that scrolls to the interactive demo, for example `Explore the demo ↓`
- include a restrained technology/meta line such as `SENTINEL-1 · SWIN-UNET · ONNX · CLOUD RUN`
- generous whitespace; avoid dense dashboard styling

### Scene 2 — Interactive demo

Purpose: combine sample selection, inference controls, and results into one continuous interaction flow.

Requirements:

- light background using `#E0E1DD`
- heading: `Try the model`
- short explanation: select a curated Sentinel-1 scene, run inference, then explore the threshold interactively
- sample cards appear at the top of this scene
- selecting a sample reveals/uses the existing workspace below; do not visually treat sample selection and inference results as separate numbered scenes
- preserve `Run inference` functionality
- make the threshold slider visually prominent
- add a small explanatory line near the slider that threshold changes rerender locally and do not rerun inference
- preserve all six output panels:
  1. SAR input
  2. Model score
  3. Binary prediction
  4. Prediction overlay
  5. Ground-truth reference
  6. Prediction errors
- keep legends, but simplify visual hierarchy so images dominate
- retain IoU, Precision, Recall, and evaluated-pixel statistics
- retain the note that per-scene metrics are diagnostic examples and not dataset-wide performance

### Scene 3 — Model context

Purpose: explain why this model is being shown and how research/deployment differ.

Requirements:

- dark background, preferably `#1B263B`
- heading: `Model context`
- add a brief intro such as: four candidate architectures were evaluated for accuracy and CPU inference performance; the five-channel cross-modal Swin-UNet was selected for deployment
- preserve the existing research-vs-deployment table values exactly unless repository evidence proves they are stale
- include a prominent GitHub link near the end of this section

## Navigation

Replace the current minimal top bar with a restrained navigation header.

Target structure:

`CloudSegmentation | Demo | Model | GitHub ↗`

Requirements:

- `Demo` scrolls to Scene 2
- `Model` scrolls to Scene 3
- `GitHub` links to `https://github.com/IngKP/CloudSegmentation`
- GitHub opens in a new tab with appropriate `rel="noopener noreferrer"`
- do not add a personal website placeholder yet
- keep navigation simple; no hamburger menu unless truly necessary for small screens

## Footer

Keep a compact footer with:

- project name
- GitHub link
- existing Sen1Floods11 attribution

Do not remove or weaken dataset attribution.

## DOM compatibility constraints

`frontend/app.js` currently depends on specific IDs. Preserve these IDs unless `app.js` is updated consistently:

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
- `prediction-caption`
- `sar-canvas`
- `score-canvas`
- `prediction-canvas`
- `overlay-canvas`
- `reference-canvas`
- `error-canvas`
- `iou`
- `precision`
- `recall`
- `valid-pixels`

Before editing `app.js`, first verify that layout/HTML/CSS changes alone cannot accomplish the requirement.

## Visual style

Use Mobbin only as a general quality reference for:

- strong typography
- clear section hierarchy
- generous spacing
- restrained cards
- clear navigation
- strong contrast between major page sections

Do not copy Mobbin's layout or add unnecessary product-marketing complexity.

Preferred qualities:

- simple
- polished
- portfolio-ready
- technical but understandable
- no gratuitous gradients
- no glassmorphism
- no large animation system
- no heavy shadows
- no excessive rounded cards
- use borders, spacing, typography, and background contrast to create hierarchy

## Responsive behavior

Test at minimum:

- desktop around 1440 px width
- laptop/tablet around 1024 px width
- mobile around 390 px width

Requirements:

- no horizontal overflow in normal page content
- navigation remains usable
- sample cards stack cleanly on small screens
- visualization grid may move from 3 columns to 2 and then 1
- metrics remain readable
- threshold control remains usable on touch-sized screens
- text should not become excessively small

## Implementation sequence

Work in this exact order.

### Task 1 — Inspect and protect behavior

1. Confirm current branch is `ui-refresh`.
2. Read `AGENTS.md`.
3. Read the current frontend files before editing.
4. Identify every DOM ID used by `frontend/app.js`.
5. Do not change application behavior in this task.

Output a short note confirming what must be preserved.

### Task 2 — Restructure `index.html`

1. Build the three-scene semantic structure.
2. Add navigation anchors for Demo and Model.
3. Add GitHub links in navigation, Scene 3, and footer.
4. Preserve all IDs required by JavaScript.
5. Keep all existing warnings, model metrics, and data attribution.
6. Do not style heavily yet beyond classes/structure needed for the next task.

Check the HTML carefully before moving on.

### Task 3 — Rebuild page styling

1. Define the five palette colors as CSS variables.
2. Style Scene 1 dark, Scene 2 light, Scene 3 dark.
3. Improve typography, spacing, navigation, cards, table, controls, and footer.
4. Ensure scientific visualization cards remain legible on the light Scene 2 background.
5. Keep semantic visualization colors distinguishable.
6. Add responsive rules for 1024 px and 390 px-class layouts.

Do not add dependencies.

### Task 4 — Threshold and interaction polish

Without changing inference behavior:

1. make threshold control more prominent
2. visually explain that changing threshold does not rerun inference
3. verify sample selection still highlights the selected sample
4. verify cached inference/result status is still visible
5. verify hidden/revealed workspace behavior still works

Only modify `app.js` if needed for a tiny presentation behavior such as scrolling to revealed results. Do not alter `/predict`, score processing, metric math, or canvas rendering logic.

### Task 5 — Local verification

Run the local app using the repository's existing workflow:

```bash
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8080
```

If required local model/sample artifacts are unavailable, do not invent or download substitutes. Report that limitation and still perform all possible static/frontend checks.

When the app can run locally, verify:

1. `/` loads
2. samples load
3. selecting each sample works
4. `Run inference` works for each available sample
5. threshold movement updates displayed prediction/metrics without another inference request
6. no browser-visible JS errors
7. GitHub links point to the correct repository
8. navigation anchors work
9. desktop layout looks coherent
10. mobile layout looks coherent

If browser/screenshot tooling is available, inspect screenshots at approximately 1440 px, 1024 px, and 390 px widths and fix obvious layout issues.

### Task 6 — Repository checks

Run:

```bash
uv run --locked pytest
uv run --locked ruff check .
```

If a failure is unrelated to the UI changes, report it clearly and do not make unrelated fixes.

Also inspect the git diff and confirm no backend/deployment/artifact files were changed unintentionally.

### Task 7 — Final report

Do not merge or deploy.

Report:

- files changed
- major UI changes
- whether `app.js` changed and why
- local run result
- inference interaction verification result
- responsive verification result
- `pytest` result
- `ruff` result
- any remaining issues or visual decisions that should be reviewed by the user

## Acceptance criteria

The task is complete only if all of these are true:

- page clearly reads as three scenes: Introduction → Interactive Demo → Model Context
- requested palette is visibly used across the site
- existing sample selection and inference behavior remains functional
- all six scientific outputs remain present
- threshold remains interactive and does not rerun inference
- existing scientific/model disclaimers remain intact
- existing model-context values remain intact
- GitHub is accessible from top navigation and lower page
- no personal website placeholder is shown
- no new frontend framework or dependency is introduced
- backend, model, cloud, and deployment logic is untouched
- mobile and desktop layouts are both usable
- branch remains `ui-refresh`; nothing is merged or deployed
