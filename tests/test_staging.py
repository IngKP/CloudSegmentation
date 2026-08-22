from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from app.staging import ArtifactStagingError, resolve_artifact_paths


class FakeBlob:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.payload)


class FakeBucket:
    name = "test-bucket"

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.requested: list[str] = []

    def blob(self, object_name: str) -> FakeBlob:
        self.requested.append(object_name)
        return FakeBlob(self.objects[object_name])


class FakeClient:
    def __init__(self, bucket: FakeBucket) -> None:
        self._bucket = bucket

    def bucket(self, bucket_name: str) -> FakeBucket:
        assert bucket_name == "test-bucket"
        return self._bucket


def _record(filename: str, payload: bytes) -> dict[str, Any]:
    return {
        "filename": filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _manifests(model: bytes, samples: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_manifest = {
        "deployment_variant": "fp32",
        "artifacts": [{"variant": "fp32", **_record("model.onnx", model)}],
    }
    sample_manifest = {"samples": [_record(name, payload) for name, payload in samples.items()]}
    return model_manifest, sample_manifest


def test_gcs_staging_downloads_and_verifies_all_manifest_artifacts(tmp_path: Path) -> None:
    model = b"model-payload"
    samples = {"India.npz": b"india", "Sri-Lanka.npz": b"sri-lanka"}
    model_manifest, sample_manifest = _manifests(model, samples)
    bucket = FakeBucket(
        {
            "models/model.onnx": model,
            "samples/India.npz": samples["India.npz"],
            "samples/Sri-Lanka.npz": samples["Sri-Lanka.npz"],
        }
    )

    paths = resolve_artifact_paths(
        tmp_path / "local.onnx",
        tmp_path / "local-samples",
        model_manifest,
        sample_manifest,
        environment={
            "CLOUDSEG_GCS_BUCKET": "test-bucket",
            "CLOUDSEG_GCS_MODEL_OBJECT": "models/model.onnx",
            "CLOUDSEG_GCS_SAMPLE_PREFIX": "samples/",
            "CLOUDSEG_GCS_STAGING_ROOT": str(tmp_path / "staging"),
        },
        client_factory=lambda: FakeClient(bucket),
    )

    assert paths.staging_root is not None
    assert paths.model_path.read_bytes() == model
    assert (paths.sample_root / "India.npz").read_bytes() == samples["India.npz"]
    assert (paths.sample_root / "Sri-Lanka.npz").read_bytes() == samples["Sri-Lanka.npz"]
    assert bucket.requested == [
        "models/model.onnx",
        "samples/India.npz",
        "samples/Sri-Lanka.npz",
    ]


def test_local_paths_are_preserved_without_gcs_configuration(tmp_path: Path) -> None:
    model_manifest, sample_manifest = _manifests(b"model", {"sample.npz": b"sample"})

    def unexpected_client() -> FakeClient:
        raise AssertionError("GCS client should not be created in local mode")

    paths = resolve_artifact_paths(
        tmp_path / "local.onnx",
        tmp_path / "local-samples",
        model_manifest,
        sample_manifest,
        environment={},
        client_factory=unexpected_client,
    )

    assert paths == resolve_artifact_paths(
        tmp_path / "local.onnx",
        tmp_path / "local-samples",
        model_manifest,
        sample_manifest,
        environment={},
    )
    assert paths.staging_root is None


def test_gcs_staging_rejects_partial_configuration(tmp_path: Path) -> None:
    model_manifest, sample_manifest = _manifests(b"model", {"sample.npz": b"sample"})

    with pytest.raises(ArtifactStagingError, match="CLOUDSEG_GCS_SAMPLE_PREFIX"):
        resolve_artifact_paths(
            tmp_path / "local.onnx",
            tmp_path / "local-samples",
            model_manifest,
            sample_manifest,
            environment={
                "CLOUDSEG_GCS_BUCKET": "test-bucket",
                "CLOUDSEG_GCS_MODEL_OBJECT": "models/model.onnx",
            },
        )


def test_gcs_staging_fails_closed_on_model_integrity_mismatch(tmp_path: Path) -> None:
    model_manifest, sample_manifest = _manifests(b"expected-model", {"sample.npz": b"sample"})
    bucket = FakeBucket(
        {"models/model.onnx": b"modified-model", "samples/sample.npz": b"sample"}
    )

    with pytest.raises(ArtifactStagingError, match="integrity verification failed"):
        resolve_artifact_paths(
            tmp_path / "local.onnx",
            tmp_path / "local-samples",
            model_manifest,
            sample_manifest,
            environment={
                "CLOUDSEG_GCS_BUCKET": "test-bucket",
                "CLOUDSEG_GCS_MODEL_OBJECT": "models/model.onnx",
                "CLOUDSEG_GCS_SAMPLE_PREFIX": "samples",
                "CLOUDSEG_GCS_STAGING_ROOT": str(tmp_path / "staging"),
            },
            client_factory=lambda: FakeClient(bucket),
        )
