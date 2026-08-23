import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from model_pipeline import load_manifest, resolve_checkpoint, select_channels


def test_manifest_metadata():
    manifest = load_manifest(Path("manifests/models.toml"))
    assert len(manifest["models"]) == 4

    first_model = manifest["models"][0]
    assert first_model["filename"] == "best_model.pth"
    assert first_model["sha256"]
    assert first_model.get("bytes", 0) > 0


def test_resolve_checkpoint_with_verified_temporary_artifact(tmp_path):
    payload = b"cloudsegmentation-test-checkpoint"
    model = {
        "id": "test_model",
        "filename": "best_model.pth",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    checkpoint_dir = tmp_path / model["id"]
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / model["filename"]
    checkpoint.write_bytes(payload)

    assert resolve_checkpoint(model, tmp_path) == checkpoint


def test_select_channels_requires_canonical_names():
    class FakeImage:
        def __getitem__(self, index):
            return index

    assert select_channels(FakeImage(), ["VV", "VH"]) == [0, 1]
    with pytest.raises(ValueError):
        select_channels(FakeImage(), ["unknown"])
