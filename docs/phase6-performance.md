# Phase 6A — Private Cloud Run performance evidence

**Status: complete.** This is deployment/performance evidence for the existing
private service, collected on 2026-08-22 UTC. It is not a statistically strong
benchmark and does not change the model, data artifacts, scientific claims, or
Cloud Run scaling configuration.

## Scope and safety

The service remained authenticated-only throughout the test. No `allUsers` or
`allAuthenticatedUsers` binding was added, and anonymous `GET /health` returned
`403` before the performance run. The max-instances `2` autoscaling experiment
was not performed.

The test generated a small number of curated `POST /predict` requests against
`Sri-Lanka_713926`. The harness did not send raw tensors or uploads and did not
store access tokens or credentials.

## Deployment identity and starting configuration

| Item | Observed value |
| --- | --- |
| Project / region | `cloudsegmentation-506305` / `asia-northeast1` |
| Service / tested URL | `cloudsegmentation` / `https://cloudsegmentation-3cqmz4dvmq-an.a.run.app` |
| Tested revision | `cloudsegmentation-00003-k7c` |
| Application source/image tag | `679ba1419819e7eeee8c1157876ad69f0fc98e67` |
| Revision-resolved image | `sha256:843d7133cf5978dce0c3aac4f9d04621cae2c9022ac4ea9c9972749dfecbfaf2` |
| Research source commit | `9d9430fd1d53d7fbb21e63a4be2bd4b47177b7a6` |
| Model | `garg_benchmark_seed42`, cross-modal Swin-UNet student, seed 42 |
| Input | 5 channels: VV, VH, DEM, HAND, Water; `1 x 5 x 512 x 512` |
| Target semantics | Sen1Floods11 released all-water mask |
| Serving artifact | FP32 ONNX, `134,949,341` bytes, one ORT thread |
| Compute | 1 vCPU, 2 GiB memory, CPU only, no GPU |
| Billing / CPU allocation | Request-based; CPU throttled outside request processing |
| Startup CPU boost | Enabled |
| Minimum instances | 0 |
| Revision-level maximum | 1 (`autoscaling.knative.dev/maxScale`) |
| Container concurrency | 1 |
| Timeout | 300 seconds |
| Authentication | IAM required; one explicit user invoker binding |

Cloud Run exposes service-level and revision-level maximums separately. The
starting `gcloud run services describe` output showed service-level scaling
`Max: 20` and revision-level `Max instances: 1`; the effective limit for this
100%-traffic revision was therefore 1, and neither setting was changed during
this work. The service-level/revision-level distinction is described in the
[Cloud Run maximum-instances documentation](https://cloud.google.com/run/docs/configuring/max-instances).

## Methodology

The reproducible harness is [`scripts/cloud_run_performance.py`](../scripts/cloud_run_performance.py).
It obtains a user identity token with `gcloud auth print-identity-token` (or
uses `CLOUDSEG_ID_TOKEN` if supplied), sends `POST /predict`, and records both:

- **End-to-end latency:** client-side time around the HTTP request, including
  connection/network time, Cloud Run queueing, application response encoding,
  and the base64 score-map response.
- **Model inference latency:** the API's existing `latency_ms`, measured around
  the ONNX Runtime session only.

The harness uses an interpolated p50/p95 calculation and records failures
without hiding them. Token acquisition occurs before the timed request and
tokens are never written to JSON output.

Commands used:

```bash
rtk uv run --locked python scripts/cloud_run_performance.py \
  --url https://cloudsegmentation-3cqmz4dvmq-an.a.run.app \
  --sample-id Sri-Lanka_713926 --mode cold --timeout 120 \
  --output /tmp/cloudseg-phase6-cold-final.json

rtk uv run --locked python scripts/cloud_run_performance.py \
  --url https://cloudsegmentation-3cqmz4dvmq-an.a.run.app \
  --sample-id Sri-Lanka_713926 --mode sequential --count 20 --timeout 120 \
  --output /tmp/cloudseg-phase6-sequential.json

rtk uv run --locked python scripts/cloud_run_performance.py \
  --url https://cloudsegmentation-3cqmz4dvmq-an.a.run.app \
  --sample-id Sri-Lanka_713926 --mode concurrency \
  --clients 1,2,5 --requests-per-client 5 --timeout 120 \
  --output /tmp/cloudseg-phase6-concurrency.json
```

For the cold test, the preceding instance was allowed to become idle and
Cloud Run logged `Shutting down` at `2026-08-22T07:23:10.902805Z`. The next
request was then the cold request. The first exploratory request earlier in
the session is not used as cold evidence because a preceding token-validation
health request had warmed the service.

## Cold start

| Measure | Result |
| --- | ---: |
| Requests | 1 |
| Success / failure | 1 / 0 |
| End-to-end mean / p50 / p95 | 8,104.1 / 8,104.1 / 8,104.1 ms |
| Model inference mean / p50 / p95 | 1,387.1 / 1,387.1 / 1,387.1 ms |
| End-to-end minus model inference | 6,716.9 ms |
| Cloud Run request-log server latency | 7.200 s |

Cold lifecycle logs for the tested revision showed:

| Event | UTC timestamp |
| --- | --- |
| `Starting new instance` | 07:24:21.749 |
| `gcs_artifacts_staged` | 07:24:26.515 |
| `service_ready` | 07:24:27.498 |
| Startup TCP probe succeeded | 07:24:27.500 |
| `prediction_complete` | 07:24:29.104 |

The instance reached `service_ready` about 5.75 seconds after the start event;
GCS staging was logged about 4.77 seconds after instance start. This is startup
and artifact-loading time, not model inference. The client-side cold result
also includes request/response transport and score-map encoding/transfer.

## Warm sequential inference

This run reused one HTTP client and sent 20 sequential predictions after the
service was warm.

| Requests | Success / failure | End-to-end mean | End-to-end p50 | End-to-end p95 | Model mean | Model p50 | Model p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 20 / 0 | 1,853.7 ms | 1,762.7 ms | 2,232.3 ms | 1,373.5 ms | 1,353.7 ms | 1,453.2 ms |

The first request in this small warm sample was slower than most later
requests. It was retained in the aggregate; no warm-up request was silently
discarded.

## Concurrent clients with max instances 1 and concurrency 1

Each client issued five sequential requests. The service configuration was not
changed between levels, so this tests queueing behavior on one constrained
revision rather than autoscaling.

| Clients | Requests | Success / failure | End-to-end mean / p50 / p95 | Model mean / p50 / p95 | E2E−model p50 | Wall time | Throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 5 / 0 | 2,086.4 / 2,114.6 / 2,223.4 ms | 1,370.2 / 1,361.1 / 1,401.7 ms | 721.0 ms | 10.48 s | 0.477 req/s |
| 2 | 10 | 10 / 0 | 3,090.8 / 3,081.5 / 3,652.2 ms | 1,379.2 / 1,380.5 / 1,401.2 ms | 1,701.9 ms | 16.28 s | 0.614 req/s |
| 5 | 25 | 25 / 0 | 7,184.3 / 7,698.3 / 8,026.1 ms | 1,376.0 / 1,376.6 / 1,390.8 ms | 6,318.6 ms | 39.16 s | 0.638 req/s |

Model inference stayed near 1.37 seconds while end-to-end latency increased
with client count. The growing end-to-end-minus-model component is consistent
with requests waiting behind the single concurrency-1 instance, plus network
and response-transfer time. No second instance was created and no failures
occurred.

## Cloud Run monitoring and logging observations

Cloud Monitoring data was queried for the tested revision over the warm and
concurrent run window. These are one-minute distribution/gauge observations,
not resource limits or long-run capacity estimates.

| Metric | Observed values |
| --- | --- |
| CPU utilization | Mean 12.7% (07:05), 5.6% (07:06), and 46.4% (07:07) |
| Memory utilization | Mean 37.4% (07:05), 47.8% (07:06), and 48.2% (07:07) |
| Active/idle instances | Active 1, idle 0 during the request window; active 0, idle 1 while waiting for termination; shutdown was then logged |
| Request count | Monitoring exported 2xx counts of 20 in 07:06 and 33 in 07:07; the harness is the authoritative count for this small run |
| Server request latency | 2xx mean 1,542.6 ms for 20 requests in 07:06 and 1,541.5 ms for 33 requests in 07:07 |
| Errors | No application error or `prediction_failed` entries; all 62 harness inference requests returned 200 |

The monitoring request series also contains the separate authentication
probes: anonymous access returned 403 and an intentionally invalid OAuth
token probe returned 401. Those are not inference failures.

## Interpretation and limitations

- The measured cold path is about 8.1 seconds end-to-end, of which the API
  reported about 1.39 seconds of model inference. The remaining time is
  dominated by instance startup, private GCS staging, readiness, transport,
  and response handling.
- The warm sequential path is about 1.76 seconds at p50 end-to-end and about
  1.35 seconds at p50 inside ONNX Runtime.
- With max instances 1 and concurrency 1, extra clients queue. At five
  clients, p95 end-to-end latency was about 8.03 seconds while model p95 stayed
  about 1.39 seconds.
- The sample is intentionally modest: one cold request, 20 sequential
  requests, and 5 requests at each of three client levels. It is suitable for
  portfolio deployment evidence, not statistically strong benchmarking,
  capacity planning, or an SLA.
- Only one curated sample and one revision were used. Results may vary with
  platform load, network path, instance placement, and sample response size.
- Cloud Monitoring values are delayed one-minute aggregates and include
  platform/authentication probes in their request series. The harness JSON
  files provide the exact performance-test request counts and timings.
- This phase did not change `max instances`, test autoscaling, add
  infrastructure, or make the service public. It makes no claim about
  transient-flood accuracy, calibration, external validation, or operational
  readiness.

## Phase 6A completion record

- Tested revision: `cloudsegmentation-00003-k7c`.
- Inference requests generated by the Phase 6 harness: approximately **62**
  (`1` exploratory request, `20` sequential, `40` concurrency, and `1` true
  cold request). All returned HTTP 200.
- The service remained private/authenticated; no public IAM binding was added.
- Region, CPU, memory, min instances, revision max instances, concurrency,
  request-based billing, and GPU settings were not changed.
- The max-instances-2 experiment remains deferred to a separate controlled
  task.
