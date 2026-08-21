# `pc-pop` setup

This repository uses two isolated Python environments:

- `.venv`, managed by `uv`, contains only CPU serving and development tools.
- `cloudseg_export`, cloned from `sar_flood`, contains research-model and ONNX
  export dependencies.

The sibling `SARfloodEstimation` checkout is read-only to this project.

## 1. Verify the checkout

```bash
cd ~/Work/IngKP/CloudSegmentation
rtk git status --short --branch
rtk git -C ../SARfloodEstimation rev-parse HEAD
```

The research commit must match `source_commit` in `manifests/models.toml`.

## 2. Create the serving environment

```bash
rtk uv sync --locked
rtk uv run --locked pytest
rtk uv run --locked ruff check .
rtk uv run --locked python scripts/check_environment.py
```

Do not install application packages manually into `.venv`; update
`pyproject.toml` and regenerate `uv.lock` instead.

## 3. Create the isolated export environment

```bash
rtk /home/kittipat/miniconda3/bin/conda create --name cloudseg_export --clone sar_flood
rtk /home/kittipat/miniconda3/envs/cloudseg_export/bin/python -m pip install \
  -r environments/export-additions.lock.txt
```

The original `sar_flood` environment must remain unchanged.

## 4. Stage private model representatives

```bash
rtk uv run --locked python scripts/stage_research_artifacts.py
rtk uv run --locked python scripts/stage_research_artifacts.py --verify-only
```

Artifacts are written under `.artifacts/checkpoints/`, excluded from Git, and
accepted only when byte size and SHA-256 match the tracked manifest.

## 5. Docker prerequisite

Docker and Google Cloud tooling are deliberately not Python dependencies.
Docker Engine is installed from Docker's Ubuntu 24.04 (`noble`) package
repository. Because Pop!_OS is an Ubuntu derivative, the repository suite is
set explicitly to `noble`.

The required privileged commands are documented in the implementation handoff;
review them before running. Adding the user to the `docker` group permits
Docker-daemon access without `sudo` and is effectively root-equivalent access.

Google Cloud CLI installation, authentication, project creation, billing, APIs,
IAM, and deployments are deferred until local ONNX inference and a local Docker
service pass.

## 6. Start the next Codex session

```bash
cd ~/Work/IngKP/CloudSegmentation
codex
```

The next milestone is to reproduce PyTorch inference for the four manifest
entries and produce a controlled CPU benchmark before selecting an export
target.

