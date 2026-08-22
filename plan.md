# Cloud-Deployed Flood-Event SAR Segmentation Demo

> **Scientific scope:** Sen1Floods11's released reference is an all-water mask.
> This portfolio project demonstrates production-style engineering on historical
> flood-event scenes; it does not claim transient-flood accuracy, calibrated
> probabilities, external validation, or operational deployment readiness.

## 1. Project Goal

Turn the existing SAR surface-water research pipeline into a small but credible
**production-style ML inference system** for flood-event scenes.

The project should demonstrate that I can take a research model beyond experimentation and:

* select a model based on accuracy/runtime trade-offs
* optimize it for inference
* expose inference through an API
* package it with Docker
* deploy it to a cloud platform
* handle model/data artifacts using cloud services
* monitor latency and failures
* test the service under repeated/concurrent requests
* document how the architecture could scale with more compute budget

The goal is **not** to operate a real flood-monitoring platform or maintain an expensive GPU endpoint 24/7.

The goal is to demonstrate the engineering path from:

> **research checkpoint → optimized model → reproducible container → cloud inference API → interactive demo**

---

# 2. Existing Work to Reuse

Do not train a completely new system unless optimization requires it.

Reuse the existing Sen1Floods11 work:

* Sentinel-1 flood segmentation
* 512 × 512 tiles
* VV and VH SAR channels
* ancillary inputs such as DEM, HAND, and water prior
* archived checkpoints, with one deployment artifact chosen from a focused
  accuracy/runtime Pareto audit
* existing preprocessing and normalization
* existing test split
* existing evaluation code
* IoU / precision / recall evaluation

The current research reports a five-channel cross-modal Swin-UNet mean
**all-water IoU of 0.7045 across three fixed seeds**. Its historical `0.7114`
score is the best individual run, not the repeated estimate. That model can
remain the accuracy-oriented research reference even if it is not deployed.

This distinction is useful:

> **Best research model ≠ best deployment model**

The portfolio project should explicitly demonstrate that engineering trade-off.

---

# 3. Main Architecture

Use **Google Cloud Run** as the deployment target.

Recommended architecture:

```text
                   ┌──────────────────────────┐
                   │       Demo Web UI        │
                   │ sample / threshold / run │
                   └─────────────┬────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │       FastAPI API        │
                   │        Cloud Run         │
                   └─────────────┬────────────┘
                                 │
                         load optimized model
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │    ONNX Runtime / CPU    │
                   │    Flood Segmentation    │
                   └─────────────┬────────────┘
                                 │
                                 ▼
              model score map → threshold → all-water mask
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │      Visualization       │
                   │ mask / overlay / metrics │
                   └──────────────────────────┘

       Cloud Storage                Artifact Registry
     model + demo data              Docker container
             │                             │
             └────────── Cloud Run ────────┘

                    Cloud Logging / Monitoring
```

Cloud Run is suitable because instances can scale to zero when unused, while
still automatically starting when a request arrives. Reverify resource limits,
region availability, and pricing immediately before deployment rather than
hard-coding mutable platform facts into the portfolio claims.

Start with:

* 1 vCPU
* 2–4 GB RAM
* request-based billing
* minimum instances = 0
* maximum instances = 1 initially
* concurrency = 1
* startup CPU boost enabled

---

# 4. Model Selection: Do This Before Building the API

The focused audit is complete. It did not select a model from the historical
`0.7114` best individual run; it compared seed-42 runtime representatives while
using archived three-seed all-water IoU for accuracy.

| Model | Repeated IoU | Parameters | Checkpoint | CPU p50 | GPU p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-channel U-Net | 0.6804 | 31.0M | 372.6 MB | 32,753 ms | 46.9 ms |
| 5-channel U-Net | 0.6820 | 31.0M | 372.6 MB | 17,958 ms | 47.1 ms |
| 5-channel ResNet34-U-Net | 0.6794 | 24.4M | 293.6 MB | 2,529 ms | 13.6 ms |
| 5-channel cross-modal Swin-UNet | **0.7045** | 31.6M | 379.8 MB | **214 ms** | 22.1 ms |

Use seed-42 checkpoints as architecture/runtime representatives, but use the
three-seed research summaries for accuracy comparisons. Select any final
deployable checkpoint using archived validation evidence, never by choosing the
highest test IoU.

Run everything using the same:

* 512 × 512 input
* preprocessing
* hardware, CPU thread configuration, and runtime versions
* batch size = 1
* test samples

Measure:

* mean latency
* p50 latency
* p95 latency
* peak memory
* checkpoint/model size
* repeated all-water IoU

The deployment model was selected from the **repeated-accuracy / latency Pareto
frontier**. Phase 1 rejected the initial two-channel U-Net hypothesis: the
cross-modal Swin-UNet student combined the best repeated IoU with the fastest
focused CPU estimate. See `docs/model-selection.md` for the exact protocol and
its limitations.

A reasonable decision rule is:

> Prefer a simpler or faster model if its repeated IoU is within roughly
> 0.01–0.03 of the best research model and the accuracy/runtime evidence
> supports the trade-off.

Do not force this threshold if the actual results tell a different story.

The resulting evidence supports a different story than originally expected:
the research-oriented Swin-UNet was also the strongest deployment candidate in
the focused audit. The README and model-selection report now state that result.

---

# 5. Optimization Strategy

Optimization was performed in increasing order of complexity. Phase 2 is
complete; the results below use only two fixed images and are feasibility
evidence, not a full accuracy evaluation.

## Stage A — Baseline PyTorch (complete)

The selected cross-modal Swin-UNet has a standalone inference adapter:

```text
input tensor
→ normalization
→ model
→ sigmoid/model score
→ threshold
→ mask
```

No training operation is required. The adapter verifies the research commit,
checkpoint hash, model configuration, channel ordering, and preprocessing.

---

## Stage B — Export to ONNX (complete)

The selected model was exported to fixed-shape FP32 ONNX and run with **ONNX
Runtime CPU** in the serving environment.

Compare:

```text
PyTorch FP32
vs
ONNX Runtime FP32
```

Across two fixed samples, FP32 ONNX and PyTorch had maximum absolute score error
`2.98e-6` and 100% thresholded-mask agreement. The 134.95 MB FP32 ONNX artifact
is the serving candidate, so the production container does not need PyTorch or
the research environment.

---

## Stage C — Quantization (complete, experimental)

Static INT8 and weight-only INT4 feasibility artifacts were produced from two
fixed samples.

| Version | Size | ORT CPU p50 | Mask agreement vs FP32 | Decision |
| --- | ---: | ---: | ---: | --- |
| ONNX FP32 | 134.95 MB | 767 ms | 100% | Deployment candidate |
| ONNX INT8 | 122.63 MB | 641 ms | 99.916% | Experimental |
| ONNX INT4 | 39.80 MB | 764 ms | 99.911% | Experimental |

Neither quantized model is promoted because two images are insufficient for a
defensible accuracy decision.

---

## Stage D — Pruning only if necessary

Pruning is optional.

Do **not start with unstructured pruning** simply because it is easy to apply. A model containing many zero weights does not automatically become faster on ordinary CPU inference.

If model size/runtime is still problematic, investigate:

* structured channel pruning
* reducing U-Net base channel width
* deploying an already smaller U-Net architecture
* retraining/distilling a smaller student

This is deferred. The verified FP32 ONNX model is sufficient to begin the local
API, and pruning would require a separate research and validation effort.

---

# 6. Simplify the Serving Input

Do NOT attempt to reproduce the complete satellite-data acquisition pipeline inside Cloud Run.

That would create unnecessary work involving:

* Sentinel downloads
* geospatial reprojection
* DEM acquisition
* HAND generation
* WorldCover processing
* GDAL/raster dependencies
* geographic alignment

Those are important research components but are not necessary to prove cloud ML deployment.

For the portfolio service, use **already-preprocessed 512 × 512 model inputs**.

Store approximately 10–20 carefully selected examples.

Each sample can contain:

```text
VV
VH
DEM
HAND
water prior
ground-truth mask
metadata
```

The API receives a prepared model input and performs the **actual model inference live**.

This keeps the deployment focused on model serving rather than geospatial ETL.

---

# 7. Make the Demo Visually Strong

Do not make the user upload an obscure five-channel NumPy tensor as the main demo.

Instead provide a gallery of prepared flood scenes.

The page should show:

### Input

A useful visualization of the Sentinel-1 tile.

For example:

* VV
* VH
* VV/VH visualization

### Prediction

Show:

* predicted flood mask
* model-score map
* overlay on the SAR image

### Ground Truth

For curated test samples, also show:

* reference flood mask
* true positive areas
* false positives
* false negatives

This makes the project immediately understandable to someone who knows nothing about remote sensing.

---

# 8. Add a Threshold Slider

This should be one of the main features.

The model API should return a compact encoded **model-score map**, not only the
binary mask.

Then let the user adjust a **model-score threshold**:

```text
Flood threshold: 0.10 ─────●───── 0.90
```

When the threshold changes, update the binary all-water prediction.

For samples where ground truth is available, also update:

* IoU
* precision
* recall

This demonstrates how threshold choice changes false positives and false
negatives on the curated benchmark examples. It is sensitivity analysis, not a
calibrated operational decision rule; model calibration was not evaluated in
the source research.

It also makes the project more interesting than:

> upload image → model gives mask.

It demonstrates that decision behavior depends on threshold choice while
remaining explicit that an operational threshold requires calibrated evidence
and task-specific error costs.

---

# 9. Show Research Model vs Deployment Model

Include a small section such as:

## Model Trade-off

|              | Research Model              | Deployment Model       |
| ------------ | --------------------------- | ---------------------- |
| Architecture | Cross-modal Swin-UNet       | Cross-modal Swin-UNet  |
| All-water IoU| 0.7045 three-seed mean      | 0.7045 archived mean   |
| Artifact size| 379.85 MB checkpoint        | 134.95 MB FP32 ONNX    |
| CPU latency  | 214 ms focused PyTorch estimate | 767 ms ORT p50, one thread |
| Purpose      | Accuracy-oriented reference | Reproducible CPU serving |

The research model result can be precomputed and must be labeled as a repeated
benchmark estimate. The best individual `0.7114` run may be shown only with
that qualification.

Only the deployment model needs to run live in Cloud Run.

The audit showed that the best research model was also the strongest deployment
candidate under the focused protocol. That evidence replaces the original
assumption that deployment would necessarily require a simpler U-Net.

---

# 10. API Design

Use FastAPI.

Minimum endpoints:

```text
GET /health
```

Returns:

```json
{
  "status": "ok",
  "model": "unet-5ch-onnx"
}
```

---

```text
GET /samples
```

Returns available demonstration scenes.

---

```text
POST /predict
```

Input:

```text
sample / prepared tensor
threshold
```

Output:

```json
{
  "prediction": "...",
  "latency_ms": 742,
  "threshold": 0.5,
  "model_version": "unet-5ch-int8"
}
```

Return a compact score-map artifact or encoded image rather than a
512-by-512 JSON float array. The client can threshold that response locally so
slider changes do not rerun inference.

FastAPI should automatically expose `/docs`, which gives a useful API demo for technical recruiters.

---

# 11. Cloud Components to Use

Use enough GCP services to demonstrate cloud knowledge without turning the project into cloud-engineering theatre.

## Cloud Run

Runs the inference API.

## Artifact Registry

Stores the Docker image.

## Cloud Storage

Stores:

* optimized model artifact
* curated demo samples
* optional benchmark outputs

Cloud Run's local filesystem is ephemeral, so persistent artifacts should live outside the container. The final design must explicitly choose and measure either a checksum-verified startup download or a model baked into the image; do not leave this implicit.

## IAM / Service Account

The Cloud Run service account should have only the permissions required to read the relevant bucket.

This gives genuine IAM experience.

## Cloud Logging

Record:

* request timestamp
* model version
* inference latency
* errors
* input sample ID

Do not log large image tensors.

## Cloud Monitoring

Track:

* request count
* latency
* CPU
* memory
* error rate
* container-instance count

---

# 12. Docker Container

The container should contain only inference dependencies.

Ideally:

```text
Python
FastAPI
Uvicorn
ONNX Runtime
NumPy
Pillow / lightweight image utilities
```

Avoid shipping:

* training code
* CUDA
* full research environment
* Jupyter
* unnecessary geospatial packages

if they are not required for serving.

Keep preprocessing code shared with the research pipeline so that training and inference normalization remain consistent.

---

# 13. Cold Start Strategy

Because minimum instances will be zero, cold starts are expected.

Measure them rather than hiding them.

Report:

```text
Cold-start latency
Warm inference latency
```

Evaluate startup CPU boost during cloud benchmarking rather than assuming its
benefit.

Load the model **once when the container starts**, not separately for every request.

Conceptually:

```python
model = load_model()

@app.post("/predict")
def predict(...):
    return model(...)
```

not:

```python
@app.post("/predict")
def predict(...):
    model = load_model()
    return model(...)
```

---

# 14. Demonstrate Continuous Inference Without Paying for 24/7 Compute

This is important.

We do not need to keep an instance permanently warm.

Instead, prove that the application behaves correctly under sustained traffic.

Use a simple load-testing tool such as:

* Locust
* k6

Test something like:

### Test A — Sequential

50 inference requests.

Measure:

* success rate
* mean latency
* p50
* p95

### Test B — Concurrent

For example:

```text
1 concurrent user
2 concurrent users
5 concurrent users
10 concurrent users
```

Measure throughput and errors.

### Test C — Autoscaling

Temporarily allow:

```text
max instances = 2 or 3
```

Send enough traffic to trigger additional instances.

Capture:

* Cloud Run instance count
* requests/sec
* p95 latency

Then reduce the maximum back to 1 afterward.

Cloud Run can add instances under sufficient load, but scaling behavior must be
measured on the deployed revision rather than inferred from local tests.

The README can therefore honestly state:

> The portfolio deployment scales to zero when idle to minimize cost. Load testing demonstrates that the containerized inference service can process sustained requests and horizontally scale to additional Cloud Run instances when traffic increases.

This is much stronger than merely leaving an expensive VM running.

---

# 15. Cost Controls

Configure the project defensively.

Use:

```text
minimum instances = 0
maximum instances = 1
```

for normal public deployment.

Only temporarily increase maximum instances during load testing.

Also configure:

* Google Cloud budget alert
* low budget threshold
* no GPU
* request-based billing

Verify the current Cloud Run pricing page, eligible region, billing mode, and
free-tier rules immediately before deployment. Do not promise a zero-cost
service in durable documentation.

---

# 16. GitHub Repository

Suggested structure:

```text
flood-segmentation-cloud/
│
├── app/
│   ├── main.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── schemas.py
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── scripts/
│   ├── export_onnx.py
│   ├── quantize.py
│   ├── benchmark_models.py
│   └── load_test.py
│
├── tests/
│   ├── test_inference.py
│   └── test_api.py
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── deploy.sh
├── README.md
└── LICENSE
```

Do not dump the entire research repository into this project.

The deployment repository should be small and understandable.

---

# 17. README Structure

The README is almost as important as the live service.

Use:

## Problem

Flood segmentation from Sentinel-1 SAR imagery.

## Research Background

Brief description of the existing benchmark and models.

## Productionization Goal

Explain why the highest-accuracy research model was not automatically selected.

## Architecture

Include a simple cloud architecture diagram.

## Model Optimization

Show:

```text
PyTorch
→ ONNX
→ quantization
```

and the benchmark table.

## Demo

GIF/video/screenshots.

## Cloud Deployment

Explain:

* Docker
* Artifact Registry
* Cloud Run
* Cloud Storage
* IAM
* Logging

## Performance

Show:

* all-water IoU and its target semantics
* model size
* warm latency
* cold-start latency
* RAM
* throughput
* load-test results

## Scaling

Explain the current scale-to-zero configuration and how higher-budget production deployment would differ.

---

# 18. Nice Demo Video

Record a 30–60 second demonstration.

Show:

1. open public Cloud Run URL
2. select a flood sample
3. click **Run inference**
4. show prediction
5. change threshold
6. show IoU/precision/recall change
7. briefly show FastAPI `/docs`
8. briefly show Cloud Run metrics/logs

Put the GIF/video near the top of the README.

Even if the live service is later disabled, the portfolio evidence remains.

---

# 19. Development Phases

## Phase 1 — Model Audit

**Status: complete.** See `docs/model-selection.md`.

**Estimated difficulty: 2/5**

* collect candidate checkpoints
* make standalone inference scripts
* benchmark CPU performance
* record parameters/model size/RAM/latency/IoU
* choose deployment model

**Deliverable:** model-selection table.

---

## Phase 2 — Optimize Inference

**Status: complete.** FP32 ONNX is the deployment candidate; INT8 and INT4 are
experimental two-image feasibility artifacts. See `docs/model-selection.md`.

**Estimated difficulty: 3/5**

* export selected model to ONNX
* verify predictions
* benchmark ONNX
* try static INT8 if appropriate
* select final inference artifact

**Deliverable:** optimized model + benchmark table.

---

## Phase 3 — Local API

**Status: complete.** The selected FP32 model is loaded once by a checksum-
verified FastAPI service. Private prepared samples and the model are mounted
read-only into a CPU-only Docker container.

**Estimated difficulty: 2/5**

* FastAPI
* `/health`
* `/samples`
* `/predict`
* model loaded once
* Docker container

**Deliverable:** local Docker inference API.

---

## Phase 4 — Demo UI

**Status: complete.** The FastAPI service now serves a same-origin interactive
gallery for the two verified private samples. Live inference returns a canonical
PNG16 score map and a browser preview; threshold changes update visualization
and valid-pixel metrics locally without rerunning the model.

**Estimated difficulty: 2–3/5**

* curated sample gallery
* SAR visualization
* prediction overlay
* model-score map
* threshold slider
* IoU/precision/recall

**Deliverable:** visually convincing local demo.

---

## Phase 5 — Cloud Deployment

**Estimated difficulty: 3/5**

* GCP project
* Artifact Registry
* Cloud Storage
* IAM service account
* Cloud Run
* logging
* budget protection

**Deliverable:** public cloud URL.

---

## Phase 6 — Performance / Scaling Test

**Estimated difficulty: 2–3/5**

### Phase 6A — Existing private deployment performance

**Status: complete.** See [`docs/phase6-performance.md`](docs/phase6-performance.md).
The existing authenticated revision was measured without changing its
configuration:

* verified the deployed revision and Cloud Run configuration
* measured a post-scale-to-zero cold request and separated startup/model time
* measured 20 warm sequential requests
* measured 1, 2, and 5 concurrent clients with five requests per client
* inspected Cloud Run logs and one-minute Monitoring CPU, memory, request,
  latency, and instance observations

### Phase 6B — Controlled autoscaling experiment

**Status: deferred.** Temporarily increasing max instances to 2 or more is a
separate controlled task and was intentionally not performed in Phase 6A.

**Deliverable:** production-style performance section.

---

## Phase 7 — Portfolio Polish

**Estimated difficulty: 2/5**

* README
* architecture diagram
* demo video
* benchmark tables
* clean Git history
* resume bullet

**Deliverable:** application-ready GitHub project.

---

# 20. Expected Time

If most existing checkpoints and inference code are usable:

| Stage                  | Expected effort |
| ---------------------- | --------------: |
| Model benchmarking     |       0.5–1 day |
| ONNX / optimization    |        1–2 days |
| FastAPI + Docker       |           1 day |
| UI                     |        1–2 days |
| Cloud deployment       |           1 day |
| Load test + monitoring |       0.5–1 day |
| README / demo          |       0.5–1 day |

Realistic total:

> **5–8 focused working days**

The biggest uncertainty is ONNX/export/model optimization, not the cloud deployment itself.

---

# 21. Stop Rules

Do not let this become another research project.

### If Swin-UNet export is difficult:

Deploy U-Net.

### If INT8 hurts accuracy:

Use FP32 ONNX.

### If ONNX gives no meaningful benefit:

Deploy PyTorch.

### If structured pruning becomes complicated:

Stop pruning.

### If raw GeoTIFF handling creates dependency problems:

Use preprocessed tensors.

### If the frontend takes too long:

Use simple HTML/JavaScript rather than a full React application.

### If Cloud Run CPU inference is slow:

Accept a few seconds of latency for the portfolio demo and report it honestly.

The objective is **deployment evidence**, not another month of model research.

---

# 22. Definition of Done

The project is finished when all of these are true:

* [ ] existing flood model performs live inference
* [ ] deployed model has documented all-water IoU and target semantics
* [ ] model runs on CPU
* [ ] Docker image builds reproducibly
* [ ] FastAPI endpoint works
* [ ] service runs on Cloud Run
* [ ] service scales to zero when idle
* [ ] public demo can execute inference
* [ ] prediction visualization is clear
* [ ] threshold slider works
* [ ] cold and warm latency are measured
* [ ] repeated-request/load test is documented
* [ ] Cloud Logging/Monitoring are demonstrated
* [ ] architecture diagram exists
* [ ] GitHub README explains optimization decisions
* [ ] short demo video/GIF exists

---

# 23. Resume Value

Once finished, the project should allow a resume bullet approximately like:

> **Productionized a Sentinel-1 surface-water segmentation model for flood-event
> scenes as a containerized cloud inference service using FastAPI, Docker, ONNX
> Runtime, and Google Cloud Run; optimized CPU inference, implemented
> scale-to-zero deployment and cloud monitoring, and benchmarked
> accuracy/latency trade-offs under concurrent requests.**

Then Skills can legitimately include:

> **Cloud / Deployment:** Google Cloud Platform, Cloud Run, Cloud Storage, Artifact Registry, IAM, Cloud Logging, Docker, FastAPI, ONNX Runtime

The important point is that every one of those technologies would be backed by an actual working project rather than simply appearing in the Skills section.
 
