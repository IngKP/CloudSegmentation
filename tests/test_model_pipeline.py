import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from model_pipeline import load_manifest, resolve_checkpoint, select_channels


def test_manifest_and_checkpoint_metadata():
    manifest = load_manifest(Path("manifests/models.toml"))
    assert len(manifest["models"]) == 4
    checkpoint = resolve_checkpoint(manifest["models"][0], manifest["artifact_root"])
    assert checkpoint.name == "best_model.pth"


def test_select_channels_requires_canonical_names():
    class FakeImage:
        def __getitem__(self, index):
            return index

    assert select_channels(FakeImage(), ["VV", "VH"]) == [0, 1]
    with pytest.raises(ValueError):
        select_channels(FakeImage(), ["unknown"])
