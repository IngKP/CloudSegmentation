# CloudSegmentation

## Purpose

Build a small, production-style inference demo from the existing SAR surface-
water segmentation research. The public-facing story is a flood-event SAR demo,
but the released Sen1Floods11 target contains all water. Do not claim transient-
flood accuracy, probability calibration, external validation, or operational
deployment readiness.

## Sources of Truth

Use sources in this order:

1. Verified artifacts and manifests in this repository.
2. The sibling `SARfloodEstimation` checkout at the commit recorded in
   `manifests/models.toml`.
3. Generated WACV research tables and provenance in that checkout.
4. Narrative summaries only when they agree with generated evidence.

Treat the sibling research repository as read-only. Do not train models, edit
research files, or publish research commits unless explicitly requested.

## Environments

- Use `uv` and the committed `uv.lock` for the CPU serving application.
- Use the isolated Conda environment `cloudseg_export` for PyTorch loading and
  ONNX export. Do not modify the existing `sar_flood` environment.
- Keep CUDA, PyTorch, rasterio, notebooks, and training dependencies out of the
  serving environment and production container.
- Prefix shell commands with `rtk` when it is available.

## Data and Artifacts

- Never commit datasets, checkpoints, ONNX binaries, calibration samples,
  probability arrays, credentials, `.env`, or machine-specific paths.
- Store local large artifacts under `.artifacts/` and stage them only through
  `scripts/stage_research_artifacts.py` so size and SHA-256 are verified.
- Every reported model result must identify the model/job ID, architecture,
  channels, seed, source commit, checkpoint hash, target semantics, hardware,
  and benchmark protocol.
- Do not redistribute Sen1Floods11 samples until redistribution terms are
  verified. Checkpoint release requires separate explicit approval.

## Benchmarking

- Begin with the four-model focused Pareto set in `manifests/models.toml`.
- Use 512 x 512 inputs, batch size 1, explicit warm-up iterations, fixed CPU
  thread counts, sequential inference, and GPU-disabled CPU runs.
- Report mean, p50, p95, peak resident memory, model size, and all-water IoU.
- Use repeated research estimates for accuracy comparisons. A single seed or
  the historical 0.7114 result must not be presented as a repeated estimate.
- Treat explanations as hypotheses until reproduced or traced to evidence.

## Serving and Cloud Safety

- The threshold control is model-score sensitivity, not calibrated risk.
- The initial API accepts curated sample IDs rather than arbitrary tensors or
  uploads. Do not expose raw model tensors as large JSON arrays.
- Do not create cloud projects/resources, change repository visibility, publish
  model binaries, or run cost-incurring load tests without explicit approval.
- Before public release, remove conference-submission wording and perform a
  separate dataset, checkpoint, secret, and identity review.

## Verification

After changes, run the narrowest relevant tests plus `uv run --locked pytest`
and `uv run --locked ruff check .`. For environment or artifact changes, also
run `uv run --locked python scripts/check_environment.py` and artifact checksum
verification. State exactly what was verified and any remaining assumptions.

