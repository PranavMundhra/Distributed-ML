"""Local inference helpers for the two trusted ONNX demonstration artifacts.

These helpers do not register models with the running gateway. Install the
optional requirements-models.txt dependencies before using them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "demo"
METADATA_KEY = "distributed_ml.demo"
HOUSE_FEATURES = (
    {"name": "area_sqft", "label": "Living area", "unit": "square feet", "source": "area"},
    {"name": "bedrooms", "label": "Bedrooms above ground", "unit": "count", "source": "Bedroom.AbvGr"},
    {"name": "full_bathrooms", "label": "Full bathrooms above ground", "unit": "count", "source": "Full.Bath"},
    {"name": "garage_spaces", "label": "Garage capacity", "unit": "cars", "source": "Garage.Cars"},
    {"name": "year_built", "label": "Year built", "unit": "year", "source": "Year.Built"},
    {"name": "overall_quality", "label": "Material and finish quality", "unit": "rating from 1 to 10", "source": "Overall.Qual"},
)
FACE_LABELS = ("female", "male")
FACE_SIZE = 32
FACE_MEANS = np.array([122.781998, 117.000999, 104.297997], dtype=np.float32)


def create_session(path: Path) -> ort.InferenceSession:
    """Use one CPU thread per session, including for the pretrained network."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def read_metadata(path: Path) -> dict:
    """Read the demo contract embedded in a locally prepared model."""
    model = onnx.load(path, load_external_data=False)
    values = {entry.key: entry.value for entry in model.metadata_props}
    if METADATA_KEY not in values:
        raise ValueError("This file has no Distributed-ML demo metadata")
    metadata = json.loads(values[METADATA_KEY])
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported demo metadata version")
    return metadata


def house_tensor(features: list[float] | np.ndarray) -> np.ndarray:
    """Validate one house or a batch; reject nonfinite user inputs."""
    original = np.asarray(features)
    if original.dtype.kind not in "iuf":
        raise ValueError("House features must be numbers")
    values = original.astype(np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(HOUSE_FEATURES):
        raise ValueError("Supply exactly six features per house, in the documented order")
    if not 1 <= len(values) <= 256:
        raise ValueError("Supply between 1 and 256 houses per batch")
    if not np.isfinite(values).all():
        raise ValueError("House features must be finite numbers")
    return np.ascontiguousarray(values)


def face_tensor(path: Path) -> np.ndarray:
    """Resize one already-cropped, upright face to raw RGB float32 NCHW.

    The ONNX graph contains the channel normalization. There is deliberately
    no detector: a photograph of a scene is not a supported input.
    """
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Use a still image containing one cropped face")
        width, height = source.size
        if min(width, height) < FACE_SIZE or width * height > 20_000_000:
            raise ValueError("Face image must be at least 32 x 32 and at most 20 megapixels")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize((FACE_SIZE, FACE_SIZE), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def predict_house(model_path: Path, features: list[float] | np.ndarray) -> list[float]:
    values = house_tensor(features)
    session = create_session(model_path)
    result = session.run(["price_usd"], {"features": values})[0]
    if result.shape != (len(values), 1) or not np.isfinite(result).all():
        raise ValueError("Model returned an invalid house price")
    return [float(value) for value in result[:, 0]]


def predict_face(model_path: Path, image_path: Path) -> dict:
    session = create_session(model_path)
    probabilities = session.run(["probabilities"], {"images": face_tensor(image_path)})[0]
    if (
        probabilities.shape != (1, 2)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Model returned invalid class probabilities")
    winner = int(probabilities[0].argmax())
    return {
        "label": FACE_LABELS[winner],
        "score": float(probabilities[0, winner]),
        "probabilities": dict(zip(FACE_LABELS, map(float, probabilities[0]))),
    }
