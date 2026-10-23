"""Prepare reproducible house-price and male/female ONNX demo bundles.

Only the house model is trained here. The image model reuses the CC0 Cydral
weights converted to ONNX by mowshon/age-and-gender. Downloads are pinned by
SHA-256; --offline rebuilds from the already verified local source cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

if __package__:
    from .demo_models import (
        DEFAULT_MODEL_DIR, FACE_LABELS, FACE_MEANS, FACE_SIZE, HOUSE_FEATURES,
        METADATA_KEY, create_session, face_tensor, predict_face,
    )
else:
    from demo_models import (
        DEFAULT_MODEL_DIR, FACE_LABELS, FACE_MEANS, FACE_SIZE, HOUSE_FEATURES,
        METADATA_KEY, create_session, face_tensor, predict_face,
    )

REVISION = "6ddb6ce2d598705ccbe2067967b2f8be01fd3914"
UPSTREAM = f"https://raw.githubusercontent.com/mowshon/age-and-gender/{REVISION}/"
SOURCES = {
    "ames.csv": (
        "https://www.openintro.org/data/csv/ames.csv",
        "cce822e6d71812f6edcf9d34bf42a2d65b927702c99ac4e6bcf20e5759d7be11",
    ),
    "gender-v1.onnx": (
        UPSTREAM + "src/age_and_gender/models/gender-v1.onnx",
        "d4c4116cf720cde18de28a09071d8e1826bcff7935ef3470052163f3f6b5d734",
    ),
    "barack-obama.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "744dd848fbb0584229169e01c4944664957c62495fb9e8af514a088ebca43e19",
    ),
    "michelle-obama.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/0/05/Michelle_Obama_official_portrait.jpg",
        "3a3d498af7266c2c5d8d3885b0ee0a142e93ca4dbc20e2454f0a324573abfbec",
    ),
    "upstream-license.txt": (
        UPSTREAM + "LICENSE",
        "f780c103f67e3f84468b19b62cdf29f53089c2c62fac1e74f9fa7228f875c624",
    ),
    "model-notice.md": (
        UPSTREAM + "src/age_and_gender/models/notices/Cydral-age-gender-models.md",
        "bc8ef03ce2fcb9349977a9f357a44f1194ef13b67fee1b7562ffa49e12d35bf2",
    ),
}
PORTRAITS = (
    {
        "source": "barack-obama.jpg", "filename": "sample_male.png", "label": "male",
        "crop": (750, 0, 1900, 1250), "person": "Barack Obama",
        "photographer": "Pete Souza / White House",
        "page": "https://commons.wikimedia.org/wiki/File:President_Barack_Obama.jpg",
    },
    {
        "source": "michelle-obama.jpg", "filename": "sample_female.png", "label": "female",
        "crop": (660, 680, 1340, 1400), "person": "Michelle Obama",
        "photographer": "Joyce N. Boghosian / White House",
        "page": "https://commons.wikimedia.org/wiki/File:Michelle_Obama_official_portrait.jpg",
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def get_source(cache: Path, name: str, *, offline: bool) -> Path:
    url, expected = SOURCES[name]
    path = cache / name
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Missing offline source: {path}. Run once without --offline.")
        print(f"Downloading {name}...", flush=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "Distributed-ML-student-project/1.0 (educational model testing)"}
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(25_000_001)
        if len(payload) > 25_000_000:
            raise ValueError(f"Source exceeds the 25 MB download limit: {name}")
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}; no model was built from it")
        path.write_bytes(payload)
    if expected and sha256(path) != expected:
        raise ValueError(f"Source checksum mismatch: {path}; restore the original source")
    return path


def save_model(model: onnx.ModelProto, path: Path, metadata: dict) -> None:
    # skl2onnx collects operator domains in a set. Their serialized order can
    # depend on Python's hash seed even when all weights and predictions match.
    imports = [helper.make_opsetid(item.domain, item.version) for item in sorted(
        model.opset_import, key=lambda item: (item.domain, item.version)
    )]
    del model.opset_import[:]
    model.opset_import.extend(imports)
    helper.set_model_props(model, {METADATA_KEY: json.dumps(metadata, sort_keys=True)})
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, path)
    manifest = {**metadata, "artifact": {"filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}}
    write_json(path.with_name("manifest.json"), manifest)


def versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in (
        "numpy", "scikit-learn", "onnx", "onnxruntime", "skl2onnx", "Pillow"
    )}


def metric_values(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae_usd": float(mean_absolute_error(targets, predictions)),
        "rmse_usd": float(root_mean_squared_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
    }


def prepare_house(output: Path, cache: Path, *, offline: bool) -> dict:
    destination = output / "house_price"
    destination.mkdir(parents=True, exist_ok=True)
    source = get_source(cache, "ames.csv", offline=offline)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2930:
        raise ValueError("Expected the original 2,930-row Ames dataset")
    features = np.array([
        [float(row[item["source"]]) if row[item["source"]] not in ("", "NA") else np.nan for item in HOUSE_FEATURES]
        for row in rows
    ], dtype=np.float32)
    targets = np.array([float(row["price"]) for row in rows], dtype=np.float64)
    train, test = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42)
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    pipeline.fit(features[train], targets[train])
    native_predictions = pipeline.predict(features[test])
    model = convert_sklearn(
        pipeline, name="Ames house price linear regression",
        initial_types=[("features", FloatTensorType([None, len(HOUSE_FEATURES)]))],
        final_types=[("price_usd", FloatTensorType([None, 1]))],
        target_opset={"": 18, "ai.onnx.ml": 1},
    )
    metadata = {
        "schema_version": 1, "model_id": "house-price-linear-v1",
        "title": "House price prediction", "task": "regression",
        "training": "trained locally; 80/20 split; random_state=42; not refitted on the test data",
        "input": {
            "kind": "tabular", "name": "features", "dtype": "float32", "shape": ["N", 6],
            "features": [
                {**item, "training_min": float(np.nanmin(features[train, index])), "training_max": float(np.nanmax(features[train, index]))}
                for index, item in enumerate(HOUSE_FEATURES)
            ],
        },
        "output": {"name": "price_usd", "dtype": "float32", "shape": ["N", 1], "unit": "USD"},
        "preprocessing": {"in_graph": True, "steps": ["training-median imputation", "training standardization", "linear regression"]},
        "source": {
            "dataset": "Ames housing, 2,930 sales in Ames, Iowa, 2006-2010",
            "author": "Dean De Cock (2011); distributed by OpenIntro",
            "page": "https://www.openintro.org/data/index.php?data=ames",
            "download": SOURCES["ames.csv"][0], "sha256": sha256(source),
            "license_note": "Retain source attribution; see OpenIntro's source and license terms. No new license is asserted for the dataset.",
        },
        "limitations": [
            "Historical Ames, Iowa prices in USD; not current Indian house prices.",
            "A simple six-feature baseline; location and other important property details are omitted.",
            "Extrapolation outside training ranges can give implausible or negative prices.",
        ],
    }
    model_path = destination / "house_price.onnx"
    save_model(model, model_path, metadata)
    session = create_session(model_path)
    predictions = session.run(["price_usd"], {"features": features[test]})[0].ravel()
    np.testing.assert_allclose(predictions, native_predictions, rtol=2e-5, atol=0.15)
    missing = features[test[:2]].copy()
    missing[0, 0] = np.nan
    np.testing.assert_allclose(
        session.run(["price_usd"], {"features": missing})[0].ravel(),
        pipeline.predict(missing), rtol=2e-5, atol=0.15,
    )
    metrics = {
        "train_rows": len(train), "test_rows": len(test), "random_state": 42,
        "onnx_test_metrics": metric_values(targets[test], predictions),
        "sklearn_test_metrics": metric_values(targets[test], native_predictions),
        "mean_price_baseline": metric_values(targets[test], np.full(len(test), targets[train].mean())),
        "max_onnx_sklearn_difference_usd": float(np.abs(predictions - native_predictions).max()),
        "versions": versions(),
    }
    write_json(destination / "metrics.json", metrics)
    # Preserve the complete holdout so tests can independently recompute the report.
    write_json(destination / "evaluation.json", {
        "features": features[test].tolist(), "actual_prices_usd": targets[test].tolist(),
        "source_row_numbers": [int(index) + 2 for index in test],
    })
    example_features = np.array([[1800, 3, 2, 2, 2000, 7]], dtype=np.float32)
    example_price = float(session.run(["price_usd"], {"features": example_features})[0][0, 0])
    examples = {
        "feature_order": [item["name"] for item in HOUSE_FEATURES],
        "custom_example": {"features": example_features[0].tolist(), "predicted_price_usd": example_price},
        "first_five_holdout_examples": [
            {"features": features[index].tolist(), "actual_price_usd": float(targets[index]), "predicted_price_usd": float(prediction)}
            for index, prediction in zip(test[:5], predictions[:5])
        ],
    }
    write_json(destination / "examples.json", examples)
    return {"artifact": str(model_path.resolve()), "example_price_usd": example_price, **metrics}


def embed_face_normalization(model: onnx.ModelProto) -> onnx.ModelProto:
    """Accept raw RGB pixels; retain the original CNN weights and output order."""
    if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
        raise ValueError("Unexpected upstream gender model input")
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "images":
                node.input[index] = "demo_normalized_images"
    model.graph.initializer.extend([
        numpy_helper.from_array(FACE_MEANS.reshape(1, 3, 1, 1), "demo_channel_means"),
        numpy_helper.from_array(np.array(256, dtype=np.float32), "demo_scale"),
    ])
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([
        helper.make_node("Sub", ["images", "demo_channel_means"], ["demo_centered_images"]),
        helper.make_node("Div", ["demo_centered_images", "demo_scale"], ["demo_normalized_images"]),
        *original_nodes,
    ])
    return model


def prepare_faces(output: Path, cache: Path, *, offline: bool) -> dict:
    destination = output / "male_female"
    samples = destination / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    source = get_source(cache, "gender-v1.onnx", offline=offline)
    model = embed_face_normalization(onnx.load(source))
    metadata = {
        "schema_version": 1, "model_id": "male-female-cydral-v1",
        "title": "Male/female face image classification", "task": "classification",
        "training": "pretrained Cydral Technology CNN; reused, not trained by this project",
        "input": {"kind": "image", "name": "images", "dtype": "float32", "shape": ["N", 3, FACE_SIZE, FACE_SIZE], "color": "RGB", "pixel_range": [0, 255]},
        "preprocessing": {
            "outside_graph": "One upright face crop; apply EXIF orientation, convert to RGB, resize to 32 x 32 with Pillow BILINEAR, transpose to NCHW.",
            "in_graph": "(raw RGB float32 - channel means) / 256.0",
            "channel_means": FACE_MEANS.tolist(), "face_detection_included": False,
        },
        "output": {"name": "probabilities", "dtype": "float32", "shape": ["N", 2], "labels": list(FACE_LABELS), "softmax_in_graph": True},
        "source": {
            "weights_author": "Cydral Technology", "weights_license": "CC0-1.0",
            "original": "https://github.com/davisking/dlib-models#dnn_gender_classifier_v1datbz2",
            "onnx_conversion": "https://github.com/mowshon/age-and-gender",
            "revision": REVISION, "download": SOURCES["gender-v1.onnx"][0], "sha256": sha256(source),
        },
        "limitations": [
            "Predicts the training dataset's binary appearance labels; does not establish gender identity or biological sex.",
            "Requires one upright cropped face; no automatic face detection or no-face rejection.",
            "Manual resizing does not reproduce the upstream dlib landmark alignment; accuracy can vary with crop, pose, lighting and population.",
            "Two example portraits are smoke checks, not an independent accuracy benchmark. Scores are not calibrated certainty.",
        ],
    }
    model_path = destination / "male_female.onnx"
    save_model(model, model_path, metadata)
    original_session = create_session(source)
    session = create_session(model_path)
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, size=(8, 3, FACE_SIZE, FACE_SIZE), dtype=np.uint8).astype(np.float32)
    normalized = (raw - FACE_MEANS.reshape(1, 3, 1, 1)) / np.float32(256)
    original_predictions = original_session.run(["probabilities"], {"images": normalized})[0]
    wrapped_predictions = session.run(["probabilities"], {"images": raw})[0]
    np.testing.assert_allclose(wrapped_predictions, original_predictions, rtol=1e-6, atol=1e-6)
    examples = []
    for item in PORTRAITS:
        portrait = get_source(cache, item["source"], offline=offline)
        with Image.open(portrait) as image:
            # Fixed, visually checked crops. They are not learned or searched for a class result.
            image.convert("RGB").crop(item["crop"]).resize((256, 256), Image.Resampling.LANCZOS).save(samples / item["filename"])
        prediction = predict_face(model_path, samples / item["filename"])
        if prediction["label"] != item["label"]:
            raise ValueError(f"Portrait smoke check failed: {item['filename']}: {prediction}")
        raw_face = face_tensor(samples / item["filename"])
        expected = original_session.run(["probabilities"], {"images": (raw_face - FACE_MEANS.reshape(1, 3, 1, 1)) / np.float32(256)})[0]
        np.testing.assert_allclose(session.run(["probabilities"], {"images": raw_face})[0], expected, atol=1e-6)
        examples.append({
            "image": "samples/" + item["filename"], "expected_demo_label": item["label"],
            **prediction, "source": {**item, "sha256": sha256(portrait), "rights": "Public domain in the United States; official federal government photograph"},
        })
    write_json(destination / "examples.json", examples)
    report = {
        "prepared_as": "pretrained reuse with normalization embedded in ONNX",
        "independent_accuracy_measured": False,
        "normalization_parity_cases": len(raw),
        "max_probability_difference": float(np.abs(original_predictions - wrapped_predictions).max()),
        "portrait_smoke_checks": [{"image": item["image"], "expected": item["expected_demo_label"], "prediction": item["label"], "score": item["score"]} for item in examples],
        "versions": versions(),
    }
    write_json(destination / "validation.json", report)
    for name in ("upstream-license.txt", "model-notice.md"):
        shutil.copyfile(get_source(cache, name, offline=offline), destination / name)
    (destination / "ATTRIBUTION.md").write_text(
        "# Source credits\n\nCNN weights: Cydral Technology, CC0-1.0.\n"
        "Original: https://github.com/davisking/dlib-models\n\n"
        f"ONNX conversion: https://github.com/mowshon/age-and-gender/tree/{REVISION}\n"
        "See upstream-license.txt and model-notice.md. Only gender weights are included.\n\n"
        "Example photographs: Pete Souza (Barack Obama) and Joyce N. Boghosian (Michelle Obama), White House.\n"
        "The official photographs are public domain in the United States. Fixed face crops and resizing were applied.\n"
        "Source pages, download hashes and crop coordinates are recorded in examples.json. No endorsement is implied.\n",
        encoding="utf-8",
    )
    return {"artifact": str(model_path.resolve()), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--offline", action="store_true", help="Use verified cached sources without network access")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    cache = output / ".sources"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        house = prepare_house(output, cache, offline=args.offline)
        print(f"House model ready: {house['artifact']}", flush=True)
        faces = prepare_faces(output, cache, offline=args.offline)
        print(f"Image model ready: {faces['artifact']}", flush=True)
        write_json(output / "validation_report.json", {"house_price": house, "male_female": faces})
    except (ValueError, OSError, AssertionError) as error:
        parser.exit(1, f"Model preparation failed: {error}\n")
    print("PASS: both ONNX models load and run on CPU; regression and normalization parity checks passed.")


if __name__ == "__main__":
    main()
