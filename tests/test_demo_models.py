"""Optional CPU tests for the prepared artifacts; never download data in tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="Install requirements-models.txt for model checks")
pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")
pytest.importorskip("PIL")

from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from scripts.demo_models import (
    DEFAULT_MODEL_DIR, FACE_MEANS, ROOT, create_session, face_tensor,
    house_tensor, predict_face, predict_house, read_metadata,
)
from scripts.prepare_demo_models import SOURCES, get_source, save_model


@pytest.fixture(scope="module")
def model_dir():
    if not (DEFAULT_MODEL_DIR / "house_price" / "house_price.onnx").exists():
        pytest.skip("Run scripts/prepare_demo_models.py once to prepare local artifacts")
    if not (DEFAULT_MODEL_DIR / "male_female" / "male_female.onnx").exists():
        pytest.skip("Image demo artifact has not been prepared")
    return DEFAULT_MODEL_DIR


@pytest.mark.parametrize("folder,filename", [
    ("house_price", "house_price.onnx"), ("male_female", "male_female.onnx"),
])
def test_models_are_self_contained_and_match_manifests(model_dir, folder, filename):
    path = model_dir / folder / filename
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    assert all(tensor.data_location != onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer)
    manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["artifact"]["bytes"] == path.stat().st_size
    assert read_metadata(path) == {key: value for key, value in manifest.items() if key != "artifact"}
    assert create_session(path).get_providers() == ["CPUExecutionProvider"]


def test_house_model_is_useful_on_untouched_holdout(model_dir):
    folder = model_dir / "house_price"
    evaluation = json.loads((folder / "evaluation.json").read_text())
    report = json.loads((folder / "metrics.json").read_text())
    features = np.array(evaluation["features"], dtype=np.float32)
    actual = np.array(evaluation["actual_prices_usd"])
    # The inference CLI deliberately caps user batches at 256; evaluate in chunks.
    predictions = np.concatenate([
        predict_house(folder / "house_price.onnx", chunk)
        for chunk in np.array_split(features, 3)
    ])
    assert report["train_rows"] == 2344
    assert report["test_rows"] == len(actual) == 586
    assert len(set(evaluation["source_row_numbers"])) == 586
    measured = {
        "mae_usd": mean_absolute_error(actual, predictions),
        "rmse_usd": root_mean_squared_error(actual, predictions),
        "r2": r2_score(actual, predictions),
    }
    assert measured == pytest.approx(report["onnx_test_metrics"])
    assert measured["mae_usd"] < 35_000
    assert measured["r2"] > 0.7
    assert measured["mae_usd"] < report["mean_price_baseline"]["mae_usd"]
    assert report["max_onnx_sklearn_difference_usd"] < 1


def test_house_batch_matches_single_predictions_and_embedded_imputation(model_dir):
    path = model_dir / "house_price" / "house_price.onnx"
    values = np.array([[1800, 3, 2, 2, 2000, 7], [1100, 2, 1, 1, 1970, 5]], dtype=np.float32)
    batch = predict_house(path, values)
    assert batch == pytest.approx([predict_house(path, row)[0] for row in values], abs=0.1)
    assert batch[0] == pytest.approx(232247.4, abs=1)
    values[0, 0] = np.nan
    # The training pipeline supports missing values even though CLI inputs must be finite.
    missing_result = create_session(path).run(["price_usd"], {"features": values})[0]
    assert np.isfinite(missing_result).all()


@pytest.mark.parametrize("invalid", [
    [], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7], [[1, 2], [3, 4]],
    [True] * 6, ["1"] * 6, [0, 0, 0, 0, 0, np.nan], [0, 0, 0, 0, 0, np.inf],
    np.ones((0, 6)), np.ones((257, 6)), np.ones((1, 1, 6)),
])
def test_house_rejects_invalid_user_inputs(invalid):
    with pytest.raises(ValueError):
        house_tensor(invalid)


@pytest.mark.parametrize("filename,expected", [("sample_male.png", "male"), ("sample_female.png", "female")])
def test_face_portrait_smoke_checks(model_dir, filename, expected):
    folder = model_dir / "male_female"
    result = predict_face(folder / "male_female.onnx", folder / "samples" / filename)
    assert result["label"] == expected
    assert set(result["probabilities"]) == {"female", "male"}
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert result["score"] == max(result["probabilities"].values())


def test_image_wrapper_preserves_upstream_probabilities_and_batch_order(model_dir):
    original_path = model_dir / ".sources" / "gender-v1.onnx"
    if not original_path.exists():
        pytest.skip("Upstream source cache absent")
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == SOURCES["gender-v1.onnx"][1]
    raw = np.random.default_rng(123).integers(0, 256, (8, 3, 32, 32), dtype=np.uint8).astype(np.float32)
    original = create_session(original_path)
    prepared = create_session(model_dir / "male_female" / "male_female.onnx")
    expected = original.run(None, {"images": (raw - FACE_MEANS.reshape(1, 3, 1, 1)) / np.float32(256)})[0]
    actual = prepared.run(None, {"images": raw})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    singles = np.concatenate([prepared.run(None, {"images": row[None]})[0] for row in raw])
    np.testing.assert_allclose(actual, singles, atol=1e-6, rtol=1e-6)


def test_face_tensor_keeps_raw_rgb_channel_order_and_converts_grayscale(tmp_path):
    path = tmp_path / "face.png"
    Image.new("RGB", (64, 80), (255, 125, 7)).save(path)
    values = face_tensor(path)
    assert values.shape == (1, 3, 32, 32)
    assert values.dtype == np.float32 and values.flags.c_contiguous
    np.testing.assert_array_equal(values[0, :, 0, 0], [255, 125, 7])
    Image.new("L", (64, 64), 42).save(path)
    assert (face_tensor(path) == 42).all()


def test_face_tensor_applies_camera_orientation(tmp_path):
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:, :40, 0] = 255
    exif = Image.Exif()
    exif[274] = 6  # Camera requests a 90-degree clockwise rotation.
    path = tmp_path / "rotated.jpg"
    Image.fromarray(pixels).save(path, exif=exif, quality=100)
    values = face_tensor(path)
    assert values[0, 0, :8].mean() > 240
    assert values[0, 0, -8:].mean() < 10


@pytest.mark.parametrize("kind", ["small", "animated", "corrupt"])
def test_face_tensor_rejects_unsupported_images(tmp_path, kind):
    path = tmp_path / "input.png"
    if kind == "small":
        Image.new("RGB", (31, 64)).save(path)
    elif kind == "animated":
        Image.new("RGB", (64, 64), "red").save(
            path, save_all=True, append_images=[Image.new("RGB", (64, 64), "blue")], duration=100,
        )
    else:
        path.write_text("not an image")
    with pytest.raises((ValueError, OSError)):
        face_tensor(path)


def test_source_cache_checksum_and_offline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing offline source"):
        get_source(tmp_path, "ames.csv", offline=True)
    (tmp_path / "ames.csv").write_text("corrupted download")
    with pytest.raises(ValueError, match="checksum mismatch"):
        get_source(tmp_path, "ames.csv", offline=True)


def test_export_hash_does_not_depend_on_operator_domain_order(tmp_path):
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "stable-demo",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None, 1])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None, 1])],
    )
    imports = [helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 1)]
    first = helper.make_model(graph, opset_imports=imports)
    second = helper.make_model(graph, opset_imports=list(reversed(imports)))
    save_model(first, tmp_path / "first.onnx", {"schema_version": 1})
    save_model(second, tmp_path / "second.onnx", {"schema_version": 1})
    assert (tmp_path / "first.onnx").read_bytes() == (tmp_path / "second.onnx").read_bytes()


@pytest.mark.parametrize("task", ["house", "face"])
def test_prediction_command_works_from_another_folder(model_dir, tmp_path, task):
    command = [sys.executable, str(ROOT / "scripts" / "predict_demo_model.py"), task]
    if task == "house":
        command += ["--features", "1800", "3", "2", "2", "2000", "7"]
    else:
        command += ["--image", str(model_dir / "male_female" / "samples" / "sample_female.png")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    if task == "house":
        assert output["predicted_price_usd"] == pytest.approx(232247.4, abs=1)
    else:
        assert output["label"] == "female"
