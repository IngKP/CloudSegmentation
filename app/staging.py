from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.cloud import storage

from app.artifacts import verify_file

logger = logging.getLogger("cloudsegmentation.staging")

_GCS_VARIABLES = (
    "CLOUDSEG_GCS_BUCKET",
    "CLOUDSEG_GCS_MODEL_OBJECT",
    "CLOUDSEG_GCS_SAMPLE_PREFIX",
)


class ArtifactStagingError(RuntimeError):
    """Raised when cloud artifacts cannot be downloaded and verified."""


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved local paths used by the existing model and sample loaders."""

    model_path: Path
    sample_root: Path
    staging_root: Path | None = None


def resolve_artifact_paths(
    local_model_path: Path,
    local_sample_root: Path,
    model_manifest: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> ArtifactPaths:
    """Use local paths unless the complete GCS staging configuration is present."""
    environment = os.environ if environment is None else environment
    values = {name: environment.get(name, "").strip() for name in _GCS_VARIABLES}
    if not any(values.values()):
        return ArtifactPaths(local_model_path, local_sample_root)

    missing = [name for name, value in values.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise ArtifactStagingError(f"GCS staging configuration is incomplete; missing: {joined}")

    return _stage_from_gcs(
        bucket_name=values["CLOUDSEG_GCS_BUCKET"],
        model_object=values["CLOUDSEG_GCS_MODEL_OBJECT"],
        sample_prefix=values["CLOUDSEG_GCS_SAMPLE_PREFIX"],
        staging_parent=Path(environment.get("CLOUDSEG_GCS_STAGING_ROOT", "/tmp/cloudseg")),
        model_manifest=model_manifest,
        sample_manifest=sample_manifest,
        client_factory=client_factory or storage.Client,
    )


def _stage_from_gcs(
    bucket_name: str,
    model_object: str,
    sample_prefix: str,
    staging_parent: Path,
    model_manifest: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    client_factory: Callable[[], Any],
) -> ArtifactPaths:
    staging_root: Path | None = None
    try:
        artifact = _deployment_artifact(model_manifest)
        model_filename = _safe_relative_path(str(artifact["filename"]))
        records = list(sample_manifest.get("samples", []))
        if not records:
            raise ArtifactStagingError("sample manifest contains no samples")

        staging_parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix="artifacts-", dir=staging_parent))
        model_path = staging_root / "model" / model_filename
        sample_root = staging_root / "samples"
        bucket = client_factory().bucket(bucket_name)

        _download(bucket, model_object, model_path)
        _verify_staged_file(
            model_path,
            int(artifact["bytes"]),
            str(artifact["sha256"]),
            model_object,
        )

        normalized_prefix = sample_prefix.rstrip("/")
        for record in records:
            filename = _safe_relative_path(str(record["filename"]))
            object_name = (
                f"{normalized_prefix}/{filename.as_posix()}"
                if normalized_prefix
                else filename.as_posix()
            )
            destination = sample_root / filename
            _download(bucket, object_name, destination)
            _verify_staged_file(
                destination,
                int(record["bytes"]),
                str(record["sha256"]),
                object_name,
            )

        logger.info(
            "gcs_artifacts_staged bucket=%s model_object=%s sample_count=%d model_bytes=%d",
            bucket_name,
            model_object,
            len(records),
            model_path.stat().st_size,
        )
        return ArtifactPaths(model_path, sample_root, staging_root)
    except ArtifactStagingError:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        raise
    except Exception as exc:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        raise ArtifactStagingError(f"GCS startup artifact staging failed: {exc}") from exc


def _deployment_artifact(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        variant = manifest["deployment_variant"]
        return next(item for item in manifest["artifacts"] if item["variant"] == variant)
    except (KeyError, StopIteration) as exc:
        raise ArtifactStagingError("optimized-model manifest has no deployment artifact") from exc


def _safe_relative_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactStagingError(f"artifact filename must be a safe relative path: {filename}")
    return path


def _download(bucket: Any, object_name: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(object_name).download_to_filename(str(temporary))
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactStagingError(
            f"failed to download gs://{bucket.name}/{object_name}: {exc}"
        ) from exc


def _verify_staged_file(
    path: Path,
    expected_size: int,
    expected_hash: str,
    object_name: str,
) -> None:
    try:
        verify_file(path, expected_size, expected_hash)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ArtifactStagingError(
            f"integrity verification failed for staged object {object_name}: {exc}"
        ) from exc
