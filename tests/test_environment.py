import hashlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from verify_optimized_artifacts import verify_manifest


def test_runtime_dependencies_are_usable() -> None:
    app = FastAPI()
    array = np.zeros((2, 2), dtype=np.uint8)
    image = Image.fromarray(array)

    assert app is not None
    assert image.size == (2, 2)
    assert "CPUExecutionProvider" in ort.get_available_providers()


def test_optimized_artifact_manifest_verification(tmp_path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"onnx")
    digest = hashlib.sha256(b"onnx").hexdigest()
    manifest = tmp_path / "optimized.toml"
    manifest.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'artifact_root = "{tmp_path}"',
                "[[artifacts]]",
                'filename = "model.onnx"',
                "bytes = 4",
                f'sha256 = "{digest}"',
            )
        )
    )
    assert verify_manifest(manifest) == [artifact]
