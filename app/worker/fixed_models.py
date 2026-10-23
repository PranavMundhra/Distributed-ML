"""Load trusted local ONNX replicas once per worker; never load user models."""

from io import BytesIO
import json
import logging
from pathlib import Path
import warnings

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from app.schemas.models import FACE_MODEL, HOUSE_MODEL

LOGGER = logging.getLogger("inference")
ARTIFACTS = {
    HOUSE_MODEL: ("house_price/house_price.onnx", "features", [None, 6], "price_usd", [None, 1]),
    FACE_MODEL: ("male_female/male_female.onnx", "images", [None, 3, 32, 32], "probabilities", [None, 2]),
}


def face_tensor(data: bytes) -> np.ndarray:
    """Decode a bounded PNG/JPEG; the ONNX graph handles channel normalization."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG"} or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Use a still PNG or JPEG image")
                width, height = source.size
                if min(width, height) < 32 or width * height > 20_000_000:
                    raise ValueError("Image must be at least 32 × 32 and at most 20 megapixels")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = image.resize((32, 32), Image.Resampling.BILINEAR)
                values = np.asarray(image, dtype=np.float32)
    except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Cannot read this image; use a valid PNG or JPEG") from exc
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def load_models(directory: Path) -> dict[str, ort.InferenceSession]:
    sessions = {}
    for model_id, (filename, input_name, input_shape, output_name, output_shape) in ARTIFACTS.items():
        path = directory / filename
        if not path.is_file():
            LOGGER.warning("model_missing model=%s; run scripts/prepare_demo_models.py", model_id)
            continue
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            metadata = json.loads(session.get_modelmeta().custom_metadata_map.get("distributed_ml.demo", "{}"))
            if metadata.get("model_id") != model_id or metadata.get("schema_version") != 1:
                raise ValueError("Unexpected demo model identity")
            for nodes, name, shape in ((session.get_inputs(), input_name, input_shape),
                                       (session.get_outputs(), output_name, output_shape)):
                if len(nodes) != 1 or nodes[0].name != name or nodes[0].type != "tensor(float)":
                    raise ValueError("Unexpected model tensor contract")
                actual = nodes[0].shape
                if len(actual) != len(shape) or actual[1:] != shape[1:]:
                    raise ValueError("Unexpected model tensor dimensions")
            sessions[model_id] = session
        except Exception:
            # One absent/broken optional artifact must not disable the Iris demo.
            LOGGER.exception("model_load_failed model=%s", model_id)
    return sessions
