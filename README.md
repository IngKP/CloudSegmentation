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

Phases 1–3 are complete. The focused audit compared:

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
in a local FastAPI service and a CPU-only Docker image. The next milestone is
the interactive demo UI.

## Local inference API

The service exposes `GET /health`, `GET /samples`, and `POST /predict`. It
accepts only curated sample IDs and returns the live model-score map as a
base64-encoded 16-bit PNG. The scores are sensitivity values, not calibrated
probabilities; decode them as `uint16 / 65535` before applying a UI threshold.

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

Open `http://127.0.0.1:8080/docs`, or request inference with:

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

## Project direction

The complete implementation plan is in [`plan.md`](plan.md). The next milestone
is Phase 4: a local gallery and threshold-sensitive visualization UI.

## License

Project source code is available under the MIT License. Dataset and model
artifacts retain their own terms and are not covered by that license.
