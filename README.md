# CloudSegmentation

A compact, production-style ML project that turns an existing Sentinel-1 SAR
research checkpoint into a reproducible CPU inference service and interactive
flood-event demo.

The Sen1Floods11 reference target used by this project is an **all-water mask**,
not a transient-inundation-only target. The project demonstrates model
selection, inference optimization, containerization, API serving, cloud
operations, and honest model limitations; it does not claim operational flood
mapping readiness.

## Current phase

Phases 1–4 are complete. Phase 5 is a live authenticated/private Cloud Run
deployment, and Phase 6A has measured its cold-start, warm, sequential, and
bounded-concurrency performance. The separate max-instances-2 autoscaling
experiment remains deferred. The focused audit compared:

- 2-channel U-Net (VV + VH)
- 5-channel U-Net (VV + VH + DEM + HAND + Water)
- 5-channel ResNet34-U-Net
- 5-channel cross-modal Swin-UNet student

The five-channel cross-modal Swin-UNet student was selected because it combined
the best archived three-seed all-water IoU (`0.7045`) with the fastest focused
CPU estimate. The initial two-channel U-Net hypothesis was rejected by the
measurements. The selected FP32 ONNX artifact passed two-image PyTorch fidelity
checks; INT8 and INT4 results remain experimental.

See [`docs/model-selection.md`](docs/model-selection.md) for the complete
protocol, limitations, and optimization results. The selected model now runs
in a local FastAPI service and CPU-only Docker image with an interactive,
threshold-sensitive visualization dashboard. An authenticated/private Phase 5
Cloud Run deployment is now live; see
[`docs/phase5-cloud-deployment.md`](docs/phase5-cloud-deployment.md) for the
observed revision, image provenance, configuration, and verification results.
See [`docs/phase6-performance.md`](docs/phase6-performance.md) for the Phase
6A deployment/performance evidence and its limitations.

## Local inference API

The service exposes `GET /health`, `GET /samples`, sample visualization routes,
and `POST /predict`. It accepts only curated sample IDs and returns the live
model-score map as a base64-encoded 16-bit PNG. The scores are sensitivity
values, not calibrated probabilities; decode the canonical map as
`uint16 / 65535`.

For browser visualization, `/predict` also returns a lossless PNG containing
scores quantized to `round(score * 255) / 255`. The dashboard uses this preview
to update the thresholded mask, overlay, IoU, precision, and recall locally;
moving the slider does not rerun inference. Invalid reference pixels are
excluded from all metrics.

The two current scenes were selected for export fidelity and interface
verification, not as a representative performance sample. Their per-scene
metrics are diagnostic examples and must not replace the archived three-seed
all-water IoU estimate.

Prepare and verify the two private samples in the isolated export environment:

```bash
rtk /home/kittipat/miniconda3/envs/cloudseg_export/bin/python \
  scripts/stage_research_artifacts.py --sample-mode
rtk /home/kittipat/miniconda3/envs/cloudseg_export/bin/python \
  scripts/stage_research_artifacts.py --sample-mode --verify-only
rtk uv run --locked python scripts/verify_optimized_artifacts.py
```

Run the service directly:

```bash
rtk uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Or build and run the publish-safe image with private artifacts mounted
read-only:

```bash
rtk docker build -t cloudsegmentation:phase3 .
rtk docker run --rm -p 8080:8080 \
  -v "$PWD/.artifacts/onnx/garg_benchmark_seed42/model.fp32.onnx:/artifacts/model/model.onnx:ro" \
  -v "$PWD/.artifacts/samples:/artifacts/samples:ro" \
  cloudsegmentation:phase3
```

Open `http://127.0.0.1:8080/` for the two-scene interactive gallery or
`http://127.0.0.1:8080/docs` for the API schema. You can also request inference
directly with:

```bash
rtk curl -X POST http://127.0.0.1:8080/predict \
  -H 'content-type: application/json' \
  -d '{"sample_id":"Sri-Lanka_713926"}'
```

The image contains no dataset, checkpoint, ONNX artifact, raster stack,
PyTorch, rasterio, CUDA dependency, or credentials. Sen1Floods11 sample and
checkpoint redistribution remain unapproved.

## Setup

See [`docs/setup-pc-pop.md`](docs/setup-pc-pop.md). Large research artifacts are
private, untracked, and verified from [`manifests/models.toml`](manifests/models.toml).

## Private cloud deployment

The production image deliberately excludes the private model and sample
artifacts. Cloud Run uses the attached runtime service account and stages the
checksum-verified GCS objects into ephemeral `/tmp` storage at startup. The
initial service requires authentication, scales from zero to one instance, and
does not expose the artifact bucket or anonymous inference.

## Project direction

The complete implementation plan is in [`plan.md`](plan.md). Phase 6A is
complete; a controlled autoscaling experiment, if approved separately, remains
the next Phase 6 task.

## License

Project source code is available under the MIT License. Dataset and model
artifacts retain their own terms and are not covered by that license.
