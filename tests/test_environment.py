import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image


def test_runtime_dependencies_are_usable() -> None:
    app = FastAPI()
    array = np.zeros((2, 2), dtype=np.uint8)
    image = Image.fromarray(array)

    assert app is not None
    assert image.size == (2, 2)
    assert "CPUExecutionProvider" in ort.get_available_providers()

