from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stage_research_artifacts import _bundle_bytes, load_sample_manifest, verify_sample


def _record(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "id": "fixture",
        "filename": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": [5, 512, 512],
        "label_shape": [512, 512],
    }


def test_bundle_is_deterministic_and_valid(tmp_path: Path) -> None:
    image = np.zeros((5, 512, 512), dtype=np.float32)
    label = np.full((512, 512), -1, dtype=np.int8)
    metadata = {"sample_id": "fixture"}
    first = _bundle_bytes(image, label, metadata)
    second = _bundle_bytes(image, label, metadata)
    assert first == second
    path = tmp_path / "fixture.npz"
    path.write_bytes(first)
    verify_sample(path, _record(path, first))


def test_bundle_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    image = np.zeros((5, 512, 512), dtype=np.float32)
    label = np.zeros((512, 512), dtype=np.int8)
    payload = _bundle_bytes(image, label, {"sample_id": "fixture"})
    path = tmp_path / "fixture.npz"
    path.write_bytes(payload)
    record = _record(path, payload)
    record["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_sample(path, record)


def test_sample_manifest_validates_shapes() -> None:
    manifest = Path(__file__).parents[1] / "manifests" / "demo_samples.toml"
    assert len(load_sample_manifest(manifest)["samples"]) == 2
