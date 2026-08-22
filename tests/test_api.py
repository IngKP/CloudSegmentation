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

from app.artifacts import (
    encode_reference,
    encode_sar_visualization,
    encode_score_map,
    encode_score_preview,
)
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


def make_settings(
    tmp_path: Path, sample_ids: tuple[str, ...] = ("demo",)
) -> tuple[Settings, FakeSession]:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model")
    sample_root = tmp_path / "samples"
    sample_root.mkdir()
    sample_records = []
    for sample_id in sample_ids:
        sample = sample_root / f"{sample_id}.npz"
        np.savez(
            sample,
            image=np.zeros((5, 512, 512), dtype=np.float32),
            label=np.zeros((512, 512), dtype=np.int8),
            metadata=np.asarray(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "channels": ["VV", "VH", "DEM", "HAND", "Water"],
                    }
                )
            ),
        )
        sample_records.append(
            f'''[[samples]]
id = "{sample_id}"
country = "Demo"
split = "test"
filename = "{sample.name}"
bytes = {sample.stat().st_size}
sha256 = "{_sha256(sample)}"'''
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
        "schema_version = 1\n\n" + "\n\n".join(sample_records) + "\n",
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


def test_score_preview_quantization_is_bounded() -> None:
    scores = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    decoded = np.asarray(Image.open(io.BytesIO(encode_score_preview(scores))), dtype=np.uint8)
    reconstructed = decoded.astype(np.float32) / 255.0
    assert np.max(np.abs(reconstructed - scores)) <= 1.0 / 255.0


def test_visual_encodings_preserve_documented_values() -> None:
    tensor = np.zeros((5, 512, 512), dtype=np.float32)
    tensor[0, 0, 0] = 1.0
    tensor[1, 0, 1] = 1.0
    sar = np.asarray(Image.open(io.BytesIO(encode_sar_visualization(tensor))), dtype=np.uint8)
    assert tuple(sar[0, 0]) == (128, 128, 128)
    assert tuple(sar[0, 1]) == (128, 128, 128)
    labels = np.array([[-1, 0], [1, 0]], dtype=np.int8)
    # The production shape is fixed at 512 x 512.
    padded = np.zeros((512, 512), dtype=np.int8)
    padded[:2, :2] = labels
    reference = np.asarray(Image.open(io.BytesIO(encode_reference(padded))), dtype=np.uint8)
    assert reference[:2, :2].tolist() == [[255, 0], [1, 0]]


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
        dashboard = client.get("/")
        assert dashboard.status_code == 200
        assert "CloudSegmentation" in dashboard.text
        assert client.get("/static/app.js").status_code == 200
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
        assert first.status_code == 200
        assert session.calls == 1
        second = client.post("/predict", json={"sample_id": "demo"})
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        body = first.json()
        decoded = Image.open(io.BytesIO(base64.b64decode(body["score_map"])))
        assert decoded.size == (512, 512)
        assert body["score_map_encoding"] == "png16-base64; score=uint16/65535"
        preview = Image.open(io.BytesIO(base64.b64decode(body["score_map_preview"])))
        assert preview.mode == "L"
        assert body["score_map_preview_encoding"] == "png8-base64; score=rint(score*255)/255"
        assert body["score_map_shape"] == [512, 512]
        for path in ("/samples/demo/sar.png", "/samples/demo/reference.png"):
            visual = client.get(path)
            assert visual.status_code == 200
            assert visual.headers["content-type"] == "image/png"
            assert visual.headers["cache-control"] == "private, max-age=3600"
            assert Image.open(io.BytesIO(visual.content)).size == (512, 512)
        assert client.get("/samples/missing/sar.png").status_code == 404
        assert client.post("/predict", json={"sample_id": "missing"}).status_code == 404
        assert client.post("/predict", json={"sample_id": "demo", "extra": 1}).status_code == 422
        assert (
            client.post("/predict", json={"sample_id": "demo", "threshold": 0.25}).status_code
            == 422
        )

    assert factory_calls == 1
    assert session.calls == 1


def test_prediction_cache_is_separate_per_sample(tmp_path: Path) -> None:
    settings, session = make_settings(tmp_path, ("demo", "alternate"))

    with TestClient(create_app(settings, lambda _path, _threads: session)) as client:
        first_demo = client.post("/predict", json={"sample_id": "demo"})
        first_alternate = client.post("/predict", json={"sample_id": "alternate"})
        second_demo = client.post("/predict", json={"sample_id": "demo"})
        second_alternate = client.post("/predict", json={"sample_id": "alternate"})

    assert first_demo.status_code == first_alternate.status_code == 200
    assert second_demo.status_code == second_alternate.status_code == 200
    assert first_demo.json() == second_demo.json()
    assert first_alternate.json() == second_alternate.json()
    assert first_demo.json()["sample_id"] == "demo"
    assert first_alternate.json()["sample_id"] == "alternate"
    assert session.calls == 2


def test_threshold_exploration_is_client_side() -> None:
    frontend = (Path(__file__).resolve().parents[1] / "frontend/app.js").read_text(
        encoding="utf-8"
    )

    assert "$('threshold').addEventListener('input', render);" in frontend
    assert frontend.count("fetch('/predict'") == 1
    assert "body: JSON.stringify({ sample_id: sampleId })" in frontend


def test_startup_rejects_modified_model(tmp_path: Path) -> None:
    settings, session = make_settings(tmp_path)
    settings.model_path.write_bytes(b"changed")
    with (
        pytest.raises(RuntimeError, match="size mismatch"),
        TestClient(create_app(settings, lambda _path, _threads: session)),
    ):
        pass
