# CloudSegmentation

A compact, production-style ML project that turns an existing Sentinel-1 SAR
research checkpoint into a reproducible CPU inference service and interactive
flood-event demo.

The Sen1Floods11 reference target used by this project is an **all-water mask**,
not a transient-inundation-only target. The project demonstrates model
selection, inference optimization, containerization, API serving, cloud
operations, and honest model limitations; it does not claim operational flood
mapping readiness.

## Live public demo

[Open the CloudSegmentation portfolio demo](https://cloudsegmentation-622822429157.asia-northeast1.run.app/).
It is served by the existing Cloud Run service and keeps the model and curated
sample objects in a private GCS bucket.

## Current phase

Phases 1–4 are complete. Phase 5 is now a live public Cloud Run deployment,
and Phase 6A has measured its cold-start, warm, sequential, and
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
threshold-sensitive visualization dashboard. The Phase 5 Cloud Run service is
now public while its GCS artifact boundary remains private; see
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
PyTorch, rasterio, CUDA dependency, or credentials. The two derived demo
visualizations are displayed with attribution by an explicit portfolio-release
decision; this does not claim an upstream dataset license or create legal
permission for redistribution.

## Public-release architecture and cost-sensitive routes

The prepared Cloud Run architecture keeps the existing private artifact
boundary: the browser calls the FastAPI service, while the model and prepared
samples remain in a private GCS bucket and are staged only into the instance's
ephemeral filesystem. The intended release configuration remains min instances
`0`, service and revision max instances `1`, concurrency `1`, one CPU, 2 GiB,
request-based billing, and no GPU. Public access is enabled only for the
existing Cloud Run service; the artifact bucket remains private.

| Route | Request-time work | Public-release note |
| --- | --- | --- |
| `/`, `/static/*`, `/docs`, `/redoc`, `/docs/oauth2-redirect`, `/openapi.json`, `/health`, `/samples` | File or metadata response | No model inference |
| `/samples/{id}/sar.png`, `/samples/{id}/reference.png` | Return startup-prepared PNG bytes | No inference; public display is an explicit release decision with attribution |
| `POST /predict` | ONNX Runtime plus score-map encoding | Only request route with meaningful model compute |

`POST /predict` accepts only manifest-selected sample IDs. Within one warm
Cloud Run instance, the first request for each of the two fixed samples runs
ONNX and stores the complete response in a bounded in-process cache. Repeated
requests for that sample return the cached response without another ONNX call;
the two sample IDs have separate entries. The cache is intentionally ephemeral
and is cleared when an instance shuts down or a new revision starts. The
response schema and frontend behavior are unchanged; the cached response keeps
the first inference's `latency_ms` value. This cache reduces duplicate work but
does not replace the server-side limit of 30 valid prediction requests per
60-second process window. Cached and uncached predictions both count; excess
requests receive HTTP 429 with `Retry-After`.

Threshold exploration remains in `frontend/app.js`: the slider rerenders the
decoded score map and local metrics, and does not call `POST /predict`.

## Sen1Floods11 attribution and release gate

The demo visibly attributes the curated scenes to [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), generated by Cloud to Street, and cites Bonafilia et al., *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1* (CVPR Workshops, 2020). The official repository documents access through its Google Cloud Storage distribution and requests that users cite the paper, but it does not state a dataset license or grant public-display/redistribution permission. Its open [missing-license issue](https://github.com/cloudtostreet/Sen1Floods11/issues/18) has no maintainer licensing response. The [official CVPR paper page](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html) establishes the dataset and source repository, not a license for the files.

The project owner has explicitly accepted public display of the two derived
demo visualizations for this portfolio release with the concise attribution
shown in the UI: “Data: Sen1Floods11 — Cloud to Street · Bonafilia et al.,
CVPRW 2020”. This is an accepted release decision, not a claim that the
upstream repository states an explicit dataset license or grants public-display
or redistribution permission; citation itself does not create legal
permission. The repository's MIT license covers this project's source code
only, not the dataset, model, or derived sample artifacts.

The public service keeps the source model and sample objects behind the
private GCS bucket; browsers receive only the application responses.

## Setup

See [`docs/setup-pc-pop.md`](docs/setup-pc-pop.md). Large research artifacts are
private, untracked, and verified from [`manifests/models.toml`](manifests/models.toml).

## Cloud deployment boundary

The production image deliberately excludes the private model and sample
artifacts. Cloud Run uses the attached runtime service account and stages the
checksum-verified GCS objects into ephemeral `/tmp` storage at startup. The
public service scales from zero to one instance with concurrency one; disabling
the Cloud Run Invoker IAM check affects this service only and does not make the
artifact bucket or its objects public.

## Project direction

The complete implementation plan is in [`plan.md`](plan.md). Phase 6A is
complete; a controlled autoscaling experiment, if approved separately, remains
the next Phase 6 task.

## License

Project source code is available under the MIT License. Dataset and model
artifacts retain their own terms and are not covered by that license.
