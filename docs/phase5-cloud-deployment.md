# Phase 5 — Guarded Google Cloud Deployment

## Goal

Deploy the Phase 4 CloudSegmentation application to Google Cloud Run while keeping private artifacts private and putting cost-abuse controls in place before any anonymous public access is enabled.

The deployment should demonstrate Cloud Run, Artifact Registry, Cloud Storage, IAM, logging, monitoring, and cost controls without turning the portfolio project into a large infrastructure project.

## Actual initial/private deployment

The initial private deployment was completed and verified on 2026-08-22. It is
authenticated-only; anonymous/public inference was deliberately not enabled.

### Application implementation

- `app/staging.py` detects the complete `CLOUDSEG_GCS_*` configuration and
  downloads the manifest-selected FP32 ONNX model plus every sample in the
  tracked sample manifest to a unique writable `/tmp/cloudseg/...` directory.
- Downloads use `google.cloud.storage.Client()` with Application Default
  Credentials. No service-account key file is read, created, or included in the
  image.
- Every staged object is verified with the existing size/SHA-256 integrity
  mechanism before the existing sample loader and one ONNX Runtime session are
  initialized.
- If the GCS variables are absent, the existing local filesystem paths remain
  active. The production Docker image contains application code and manifests
  only; model and NPZ files are not copied into it.

### Observed deployment record

| Item | Observed value |
| --- | --- |
| Project / region | `cloudsegmentation-506305` / `asia-northeast1` |
| Service | `cloudsegmentation` |
| Private URL | `https://cloudsegmentation-3cqmz4dvmq-an.a.run.app` |
| Revision | `cloudsegmentation-00003-k7c` |
| Source commit / image tag | `679ba1419819e7eeee8c1157876ad69f0fc98e67` |
| Artifact Registry image | `asia-northeast1-docker.pkg.dev/cloudsegmentation-506305/cloudsegmentation/cloudsegmentation:679ba1419819e7eeee8c1157876ad69f0fc98e67` |
| Registry manifest digest | `sha256:a60f3e26417758a12ceae13a97528b962223d99dc2f22aba1016459725402361` |
| Revision-resolved image digest | `sha256:843d7133cf5978dce0c3aac4f9d04621cae2c9022ac4ea9c9972749dfecbfaf2` |
| Runtime service account | `cloudseg-runtime@cloudsegmentation-506305.iam.gserviceaccount.com` |
| CPU / memory | `1` vCPU / `2Gi` |
| Scaling / concurrency | min `0`, max `1`, concurrency `1` |
| Billing / GPU | request-based CPU throttling / none configured |

The private artifact bucket and runtime service account were reused. The
`cloudsegmentation` Artifact Registry Docker repository was created because it
did not exist. Cloud Run service IAM contains one explicit owner-user
`roles/run.invoker` binding for authenticated verification; it contains no
`allUsers` or `allAuthenticatedUsers` binding. The bucket has uniform
bucket-level access and public access prevention enforced, with the runtime
service account granted only bucket-scoped `roles/storage.objectViewer`.

No billing accounts, budgets, spend caps, payment settings, organization
settings, or unrelated infrastructure were changed.

### Verification results

- Anonymous `GET /health`: `403`.
- Authenticated `GET /health`: `200`, model `garg_benchmark_seed42`, FP32,
  two samples, and the expected Sen1Floods11 all-water target semantics.
- Authenticated `GET /samples`: `200`, listing `Sri-Lanka_713926` and
  `India_80221`.
- Authenticated dashboard `/`, `/static/app.js`, SAR visualization, and
  reference visualization: `200`.
- Authenticated live inference succeeded once for each sample. Observed model
  latency was approximately `1721 ms` for Sri Lanka and `1699 ms` for India;
  these are two verification requests, not a benchmark estimate.
- Cloud Run logs contain `gcs_artifacts_staged`, `service_ready`, and
  `prediction_complete` messages with bucket/object/sample/model/latency
  metadata only. No error-level entries or large tensors/credentials were
  observed in the final verification window.

## Security and cost principles

1. **Use a dedicated GCP project for this demo.** Do not share the project with unrelated workloads.
2. **Deploy privately first.** The first Cloud Run revision must require authentication.
3. **Keep model and sample artifacts private.** Cloud Storage must use public access prevention and uniform bucket-level access.
4. **Use least-privilege IAM.** The Cloud Run runtime service account receives read-only object access to the artifact bucket and no project-wide editor/owner role.
5. **Cap compute before public exposure.** Use request-based billing, minimum instances `0`, service-level maximum instances `1`, concurrency `1`, CPU only, and no GPU.
6. **Configure billing controls before anonymous traffic.** Create ordinary budget alerts and, where available, a Cloud Run spend-cap budget. Treat neither as instantaneous protection.
7. **Never depend on client-side controls for cost protection.** If anonymous inference is later enabled, add server-side rate/quota controls for `POST /predict`.
8. **Do not expose private dataset/model objects directly.** The browser talks only to the Cloud Run application.
9. **Do not store service-account keys in the repository or container.** Use Cloud Run service identity / Application Default Credentials.
10. **Do not log tensors, model files, credentials, or full request bodies.** Log only operational metadata such as sample ID, model version, latency, response status, and request timestamps.

## Artifact strategy

Use a private Cloud Storage bucket for:

- `models/garg_benchmark_seed42/model.fp32.onnx`
- `samples/Sri-Lanka_713926.npz`
- `samples/India_80221.npz`

The tracked manifests remain the source of truth for expected filename, byte size, and SHA-256.

On container startup:

1. Download the selected model and demo sample objects to the container's ephemeral filesystem.
2. Verify byte size and SHA-256 against the tracked manifests.
3. Refuse startup on any mismatch.
4. Load samples and create the ONNX Runtime session only after verification succeeds.

Do not make the bucket public and do not serve signed object URLs to the browser.

## GCP resources

Use one region consistently where practical. `asia-northeast1` is preferred for this project because development is in Japan; verify service and pricing availability immediately before creation.

Create only these resources initially:

- one dedicated GCP project
- one private Cloud Storage bucket
- one Artifact Registry Docker repository
- one Cloud Run runtime service account
- one Cloud Run service
- Cloud Logging / Monitoring generated by Cloud Run
- one normal billing budget
- one Cloud Run spend-cap budget if available to the billing account

Avoid VPC connectors, GPUs, external load balancers, Cloud Armor, databases, and other paid infrastructure until there is a demonstrated need.

## Runtime IAM

Create a dedicated service account, for example:

`cloudseg-runtime@PROJECT_ID.iam.gserviceaccount.com`

Grant it only:

- `roles/storage.objectViewer` on the CloudSegmentation artifact bucket

Do not grant:

- Owner
- Editor
- Storage Admin
- Artifact Registry Admin
- Cloud Run Admin
- Service Account Token Creator

The deployer identity may need broader deployment permissions, but the running container must not inherit them.

## Cloud Storage guardrails

The artifact bucket must have:

- uniform bucket-level access enabled
- public access prevention enforced
- no `allUsers` or `allAuthenticatedUsers` bindings
- no public object ACLs

Before deployment, verify the effective IAM policy and confirm the runtime service account is the only application identity with object-read access.

## Cloud Run baseline configuration

Initial authenticated deployment:

```text
CPU                    1 vCPU
Memory                 2 GiB initially; increase only if measured necessary
Billing                request-based
Minimum instances      0
Maximum instances      1 (service level)
Concurrency            1
Workers                1
GPU                    none
Authentication         required
Ingress                all initially; IAM still blocks anonymous requests
Runtime service account dedicated cloudseg-runtime identity
```

Do not enable anonymous/public invocation in the same step as the first deployment.

The maximum-instance setting is a cost guardrail, not an exact billing cap. Cloud Run can temporarily exceed configured maximums in some circumstances, and billing controls have reporting/enforcement delay.

## Billing guardrails

Before any public access:

1. Create a standard monthly budget scoped to the dedicated project with low warning thresholds.
2. Configure Cloud Run Budget spend cap for the project if the billing account exposes the Preview feature.
3. Set the spend cap below the absolute amount you are willing to pay because enforcement is not instantaneous.
4. Keep `max instances = 1` even when a spend cap exists.
5. Check the Cloud Run Billing panel during Phase 5 and Phase 6 tests.

A spend cap should be treated as the final financial backstop, not the first line of defense.

## Public-release gate

Anonymous access is **not part of the initial deployment**.

Before making the Cloud Run service public, all of the following must be true:

- [ ] Cloud Run works correctly while authenticated.
- [ ] Model and sample artifacts download from GCS and pass checksum verification.
- [ ] Runtime service account has only bucket-scoped read access.
- [ ] Artifact bucket has public access prevention enabled.
- [ ] Artifact bucket has uniform bucket-level access enabled.
- [ ] Cloud Run minimum instances is `0`.
- [ ] Cloud Run service-level maximum instances is `1`.
- [ ] Cloud Run concurrency is `1`.
- [ ] Request-based billing is enabled.
- [ ] Normal budget alerts are configured.
- [ ] Cloud Run spend cap is configured if available.
- [ ] Logging has been checked for accidental sensitive-data output.
- [ ] Sen1Floods11/model redistribution and public-display constraints are resolved for the assets shown by the UI.
- [ ] A server-side inference abuse policy has been selected.

## Anonymous inference abuse controls

If the demo is later made public, protect the expensive route separately from the static/read-only routes.

`GET /`, `/static/*`, `/health`, `/samples`, and cached visualization routes are cheap. `POST /predict` performs model inference and is the cost-sensitive operation.

Minimum application-side protection before public release:

- reject unknown sample IDs (already implemented)
- keep one inference worker / concurrency `1`
- add a server-side rate limit for `POST /predict`
- optionally add a project-wide daily inference quota with HTTP `429` when exhausted
- log rate-limit events without storing sensitive request data

Do not rely on a disabled button, JavaScript cooldown, or any other browser-only mechanism; clients can call the API directly.

For the first public version, prefer a deliberately small quota. The objective is a recruiter/demo endpoint, not an unrestricted inference API.

Cloud Armor and an external Application Load Balancer are deferred. Add them only if the public service attracts enough traffic to justify the extra complexity and recurring cost.

## Phase 5 implementation order

### 5A — Account/project safety setup

- install/authenticate `gcloud`
- create or select the dedicated project
- attach the intended billing account
- enable the required APIs
- create normal budget alerts
- create a Cloud Run spend cap if available
- record project ID, region, and resource names in local environment configuration, not secrets in Git

### 5B — Private artifact storage

- create Cloud Storage bucket
- enforce public access prevention
- enable uniform bucket-level access
- upload FP32 ONNX model and two prepared samples
- verify uploaded object sizes/hashes

### 5C — Runtime identity

- create `cloudseg-runtime` service account
- grant bucket-scoped `roles/storage.objectViewer`
- verify no unnecessary project-wide roles are attached

### 5D — Cloud artifact loader

- add GCS startup-download support
- download only manifest-declared artifacts
- checksum/size verify before model load
- fail closed on missing or modified artifacts
- retain the current local-path mode for local Docker development
- add tests with a fake/mock storage client where practical

### 5E — Container publication

- create Artifact Registry Docker repository
- configure Docker authentication
- build the Phase 5 image
- tag with an immutable version/commit identifier
- push to Artifact Registry

### 5F — Private Cloud Run deployment

Deploy with authentication required and conservative resources:

- 1 vCPU
- 2 GiB RAM initially
- request-based billing
- min instances 0
- service max instances 1
- concurrency 1
- dedicated runtime service account

Verify:

- startup succeeds from empty ephemeral storage
- GCS artifact hashes pass
- `/health` works for an authenticated caller
- UI works through authenticated access
- live inference works
- application logs contain no private arrays or credentials
- scale-to-zero works

### 5G — Public-release decision

Only after the public-release gate is complete:

- resolve public-display/redistribution permission for the demo assets
- add server-side inference rate/quota protection
- make the service anonymous only if desired
- immediately verify anonymous access cannot reach private GCS objects directly
- confirm `max instances = 1` and budget/spend-cap settings after the change

## Phase 5 deliverable

A **guarded Cloud Run deployment** with:

- private GCS artifacts with checksum-verified startup download
- Artifact Registry container image
- least-privilege runtime IAM
- scale-to-zero behavior
- maximum one service instance during normal operation
- billing alerts and spend-cap protection where available
- Cloud Logging/Monitoring visibility
- authenticated working demo first
- optional public demo only after explicit security, licensing, and abuse-control checks

The Phase 5 deliverable is not simply "a public URL." A secure authenticated deployment with documented public-release controls counts as Phase 5 complete; anonymous publication is a separate release decision.
