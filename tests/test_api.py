from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.artifacts import encode_score_map
from app.config import Settings
from app.main import create_app


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[1, 5, 512, 512])]

    def get_outputs(self):
        return [SimpleNamespace(name="score_map", shape=[1, 1, 512, 512])]

    def run(self, output_names, inputs):
        self.calls += 1
        assert output_names == ["score_map"]
        assert inputs["input"].shape == (1, 5, 512, 512)
        return [np.full((1, 1, 512, 512), 0.5, dtype=np.float32)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_settings(tmp_path: Path) -> tuple[Settings, FakeSession]:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model")
    sample_root = tmp_path / "samples"
    sample_root.mkdir()
    sample = sample_root / "demo.npz"
    np.savez(
        sample,
        image=np.zeros((5, 512, 512), dtype=np.float32),
        label=np.zeros((512, 512), dtype=np.int8),
        metadata=np.asarray(
            json.dumps(
                {
                    "sample_id": "demo",
                    "channels": ["VV", "VH", "DEM", "HAND", "Water"],
                }
            )
        ),
    )
    model_manifest = tmp_path / "models.toml"
    model_manifest.write_text(
        f'''schema_version = 1
source_commit = "commit"
selected_model = "model-id"
deployment_variant = "fp32"
input_name = "input"
output_name = "score_map"
input_shape = [1, 5, 512, 512]
target_semantics = "all-water mask"

[[artifacts]]
variant = "fp32"
filename = "model.onnx"
bytes = {model.stat().st_size}
sha256 = "{_sha256(model)}"
''',
        encoding="utf-8",
    )
    sample_manifest = tmp_path / "samples.toml"
    sample_manifest.write_text(
        f'''schema_version = 1

[[samples]]
id = "demo"
country = "Demo"
split = "test"
filename = "demo.npz"
bytes = {sample.stat().st_size}
sha256 = "{_sha256(sample)}"
''',
        encoding="utf-8",
    )
    settings = Settings(
        model_path=model,
        sample_root=sample_root,
        model_manifest_path=model_manifest,
        sample_manifest_path=sample_manifest,
        ort_threads=1,
    )
    return settings, FakeSession()


def test_score_map_png_round_trip_error_is_bounded() -> None:
    scores = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    decoded = np.asarray(Image.open(io.BytesIO(encode_score_map(scores))), dtype=np.uint16)
    reconstructed = decoded.astype(np.float32) / 65535.0
    assert np.max(np.abs(reconstructed - scores)) <= 1.0 / 65535.0


@pytest.mark.parametrize("value", [np.nan, -0.01, 1.01])
def test_score_map_rejects_invalid_values(value: float) -> None:
    scores = np.zeros((512, 512), dtype=np.float32)
    scores[0, 0] = value
    with pytest.raises(ValueError):
        encode_score_map(scores)


def test_api_contract_and_single_session(tmp_path: Path) -> None:
    settings, session = make_settings(tmp_path)
    factory_calls = 0

    def factory(path: Path, threads: int) -> FakeSession:
        nonlocal factory_calls
        factory_calls += 1
        assert path == settings.model_path
        assert threads == 1
        return session

    with TestClient(create_app(settings, factory)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_id": "model-id",
            "variant": "fp32",
            "source_commit": "commit",
            "target_semantics": "all-water mask",
            "sample_count": 1,
        }
        assert client.get("/samples").json() == [
            {"id": "demo", "country": "Demo", "split": "test", "has_reference": True}
        ]
        first = client.post("/predict", json={"sample_id": "demo"})
        second = client.post("/predict", json={"sample_id": "demo"})
        assert first.status_code == second.status_code == 200
        body = first.json()
        decoded = Image.open(io.BytesIO(base64.b64decode(body["score_map"])))
        assert decoded.size == (512, 512)
        assert body["score_map_encoding"] == "png16-base64; score=uint16/65535"
        assert body["score_map_shape"] == [512, 512]
        assert client.post("/predict", json={"sample_id": "missing"}).status_code == 404
        assert client.post("/predict", json={"sample_id": "demo", "extra": 1}).status_code == 422

    assert factory_calls == 1
    assert session.calls == 2


def test_startup_rejects_modified_model(tmp_path: Path) -> None:
    settings, session = make_settings(tmp_path)
    settings.model_path.write_bytes(b"changed")
    with (
        pytest.raises(RuntimeError, match="size mismatch"),
        TestClient(create_app(settings, lambda _path, _threads: session)),
    ):
        pass
