# Phase 5 Agent Handoff

This document is a safe, non-secret handoff for an implementation agent working on CloudSegmentation Phase 5.

## Current repository state

- Repository: `IngKP/CloudSegmentation`
- Default branch: `main`
- Phase 4 is complete, including the per-sample inference race fix.
- Local repo path used by the owner: `/home/kittipat/Work/IngKP/CloudSegmentation`
- Deployment model: fixed-shape FP32 ONNX cross-modal Swin-UNet
- Model manifest: `manifests/optimized_models.toml`
- Model size: `134949341` bytes
- Model SHA-256: `36628caa6289da9fa76d93369525b1d5a0a35d4cd2ecfbcabcf775d6dfb85b88`
- Local model path: `.artifacts/onnx/garg_benchmark_seed42/model.fp32.onnx`
- Local samples:
  - `.artifacts/samples/India_80221.npz`
  - `.artifacts/samples/Sri-Lanka_713926.npz`

## Scientific guardrails

Sen1Floods11's released target is an all-water mask. Do not describe the system as validated operational flood mapping, calibrated flood probability, transient-inundation accuracy, or external validation.

The two demo samples are verification examples and are not representative benchmark evidence. Keep the existing scientific caveats visible.

## GCP context

All identifiers below are non-secret and mirrored in `deploy/gcp.env.example`.

- Project ID: `cloudsegmentation-506305`
- Region: `asia-northeast1` (Tokyo)
- Private artifact bucket: `cloudsegmentation-506305-artifacts`
- Runtime service account: `cloudseg-runtime@cloudsegmentation-506305.iam.gserviceaccount.com`
- Desired Artifact Registry repository: `cloudsegmentation`
- Desired Cloud Run service: `cloudsegmentation`

The owner has already:

1. Created the project and linked billing.
2. Created an alert-only project budget.
3. Created a Cloud Run spend-cap budget.
4. Installed and authenticated Google Cloud CLI.
5. Set the default project to `cloudsegmentation-506305`.
6. Set Cloud Run default region to `asia-northeast1`.
7. Enabled:
   - `run.googleapis.com`
   - `artifactregistry.googleapis.com`
   - `storage.googleapis.com`
8. Created private bucket `gs://cloudsegmentation-506305-artifacts` with:
   - uniform bucket-level access enabled
   - public access prevention enforced
9. Created runtime service account `cloudseg-runtime@cloudsegmentation-506305.iam.gserviceaccount.com`.
10. Granted that service account `roles/storage.objectViewer` on this bucket only.
11. Uploaded:
   - `gs://cloudsegmentation-506305-artifacts/models/garg_benchmark_seed42/model.fp32.onnx`
   - `gs://cloudsegmentation-506305-artifacts/samples/India_80221.npz`
   - `gs://cloudsegmentation-506305-artifacts/samples/Sri-Lanka_713926.npz`
12. Verified the uploaded model object is `134949341` bytes; the local SHA-256 matches the tracked manifest.

Do not assume the Artifact Registry repository or Cloud Run service already exists. Check before creating them.

## Hard safety constraints

These constraints override convenience:

- Do not modify billing accounts, budgets, spend caps, payment information, or organization settings.
- Do not create service-account keys or download JSON credentials.
- Do not put credentials, OAuth tokens, access tokens, API keys, billing-account IDs, or secrets into Git or env examples.
- Use Application Default Credentials / attached Cloud Run service-account identity rather than key files.
- Do not grant `Owner`, `Editor`, `Storage Admin`, or other broad project-level roles to the runtime service account.
- Keep the artifact bucket private.
- Do not add `allUsers` or `allAuthenticatedUsers` to the bucket.
- Do not make Cloud Run unauthenticated during the initial deployment.
- Do not use `--allow-unauthenticated`.
- Cloud Run normal configuration must remain:
  - CPU: 1 vCPU
  - memory: 2 GiB initially
  - min instances: 0
  - max instances: 1
  - concurrency: 1
  - no GPU
  - request-based billing
- Do not create a load balancer, Cloud Armor, VPC connector, database, Firestore, VM, GKE cluster, GPU resource, or other infrastructure unless explicitly requested later.
- Before every cloud mutation, inspect whether the resource already exists; make steps idempotent where practical.
- Do not delete existing cloud resources.

## Phase 5 implementation objective

The initial objective is an authenticated/private Cloud Run deployment that runs the existing Phase 4 application using private GCS artifacts.

Expected architecture:

```text
authenticated user
      |
      v
Cloud Run (FastAPI + frontend)
1 vCPU / 2 GiB / min 0 / max 1 / concurrency 1
      |
      | attached runtime service account
      v
private Cloud Storage
model + two prepared samples

Artifact Registry
private Docker image
```

## Required code change: GCS startup artifact staging

The current local service loads artifacts from local filesystem paths. Preserve that mode for local development.

Add a cloud mode that, during application/container startup:

1. Reads the bucket/object configuration from environment variables.
2. Downloads the FP32 ONNX model and the two prepared sample files from GCS to writable ephemeral local storage such as `/tmp/cloudseg/...`.
3. Uses Google Application Default Credentials automatically; do not use a service-account key file.
4. Verifies the downloaded model size and SHA-256 using the existing manifest/integrity mechanisms before loading it.
5. Then initializes the existing sample/model loading and ONNX Runtime session exactly once.
6. Fails startup clearly if download or integrity validation fails.

Keep local filesystem mode working exactly as before.

Prefer a small, testable staging module instead of scattering GCS logic through `app/main.py`.

## Environment design

Use `deploy/gcp.env.example` as non-secret deployment context, but do not treat it as a secrets file.

Application runtime variables may include names such as:

- `CLOUDSEG_GCS_BUCKET`
- `CLOUDSEG_GCS_MODEL_OBJECT`
- `CLOUDSEG_GCS_SAMPLE_PREFIX`
- existing `CLOUDSEG_MODEL_PATH`
- existing `CLOUDSEG_SAMPLE_ROOT`
- existing `CLOUDSEG_ORT_THREADS`

Choose a clean design that automatically stages cloud artifacts when the GCS variables are present and otherwise retains current local paths.

## Artifact Registry / image

If `cloudsegmentation` Artifact Registry repository does not exist in `asia-northeast1`, create a Docker repository with that name.

Build the existing production Docker image and push it using an immutable or traceable tag, preferably the current Git commit SHA rather than relying only on `latest`.

Do not bake the private model or sample NPZ files into the Docker image.

## Initial Cloud Run deployment

Deploy only after code/tests pass and the image is pushed.

Required initial properties:

- project: `cloudsegmentation-506305`
- region: `asia-northeast1`
- service name: `cloudsegmentation`
- runtime service account: `cloudseg-runtime@cloudsegmentation-506305.iam.gserviceaccount.com`
- require authentication
- 1 vCPU
- 2 GiB memory initially
- min instances 0
- service-level max instances 1
- concurrency 1
- no GPU
- request-based billing
- set only the non-secret runtime environment variables required for GCS staging

Do not expose public access to the service.

## Verification

After private deployment:

1. Inspect the deployed service configuration and confirm authentication is required.
2. Confirm max instances = 1, min = 0, concurrency = 1, runtime service account is correct, and no GPU is configured.
3. Invoke the service as the authenticated owner.
4. Verify `/health`.
5. Verify `/samples`.
6. Load the UI while authenticated.
7. Run one live inference for each sample.
8. Confirm logs contain startup/download/inference information but no large tensors or credentials.
9. Report the Cloud Run URL, deployed image digest/tag, revision, and any measured cold/warm latency observed during verification.

## Public-access gate

The original initial-deployment gate below was intentionally private-only. For
the current portfolio release, the project owner has explicitly accepted
public display of the two derived visualizations with factual attribution.
That decision does not claim an explicit upstream dataset license or create
legal permission through citation.

Reasons:

- The upstream Sen1Floods11 repository does not state an explicit dataset
  license; retain that caveat in release documentation.
- Public inference requires the implemented process-wide prediction rate limit
  (`CLOUDSEG_PREDICTION_RATE_LIMIT`, default 30 valid requests per 60 seconds)
  in addition to the per-sample cache.

Anonymous access remains a separate final step: complete private verification
first, then disable the Cloud Run Invoker IAM check only for this service.

## Definition of completion for this handoff

This task is complete when:

- GCS startup staging is implemented and tested.
- Local development mode still works.
- Artifact Registry repository exists.
- Docker image is built and pushed without private artifacts embedded.
- Cloud Run is deployed privately with the required resource limits and runtime identity.
- The authenticated UI and inference work against the cloud deployment.
- Public access, if enabled, is added only after the private verification gate
  and does not make the GCS bucket public.
- Relevant deployment documentation is updated with what was actually done.
