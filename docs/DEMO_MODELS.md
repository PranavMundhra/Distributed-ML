# Two prepared inference models

The models are ready locally and connected to the three-model interface and
both workers. Choose House or Image in the model selector; each worker loads its
own local ONNX replica. PNG/JPEG uploading supplies an inference image, not a new
model. Public model uploads, authentication and automatic model distribution
remain deferred. See [USER_GUIDE.md](USER_GUIDE.md) for UI and API instructions.

| Model | File under the project folder | Input | Output |
| --- | --- | --- | --- |
| House price | `models/demo/house_price/house_price.onnx` | Six property measurements | Predicted historical sale price in USD |
| Male/female face classifier | `models/demo/male_female/male_female.onnx` | One upright, cropped face image | `female` / `male` probabilities |

Both are self-contained ONNX files. Model IDs, feature order, image preparation,
labels, source credits, and limitations are embedded in their metadata. Companion
`manifest.json` files contain the same information plus artifact hashes and sizes.
These describe the demo models; support for arbitrary ONNX models is not implemented.

## 1. House price: trained locally

This is a simple **multiple linear regression**: six inputs are combined with
learned weights to estimate a price. Training-median imputation and feature
standardization are included inside the exported model.

The data contains 2,930 residential sales in **Ames, Iowa, from 2006 to 2010**.
The split uses 2,344 rows for training and 586 separate rows for testing, with
`random_state=42`. The saved model is not refitted on the test rows.

Supply the inputs in this exact order:

| Position | Feature | Example | Meaning |
| --- | --- | --- | --- |
| 1 | `area_sqft` | 1800 | Above-ground living area in square feet |
| 2 | `bedrooms` | 3 | Bedrooms above ground |
| 3 | `full_bathrooms` | 2 | Full bathrooms above ground |
| 4 | `garage_spaces` | 2 | Garage capacity in cars |
| 5 | `year_built` | 2000 | Original construction year |
| 6 | `overall_quality` | 7 | Material and finish quality, rated 1 to 10 |

For `[1800, 3, 2, 2, 2000, 7]`, the prediction is **US$232,247.41**.

Measured on the 586 held-out sales:

| Metric | Result |
| --- | --- |
| Mean absolute error | US$27,385.01 |
| Root mean squared error | US$42,175.11 |
| R² | 0.7781 |
| Mean-price baseline's mean absolute error | US$64,611.41 |
| Largest ONNX vs. scikit-learn prediction difference | US$0.03125 |

Mean absolute error means the average absolute gap between the predicted and
actual price. R² describes improvement over a constant-price predictor; it is
**not a percentage accuracy**. This baseline omits neighborhood and many other
property details. Predictions outside the observed training ranges can be
implausible, including negative values. These are historical US prices, not
current Indian prices or rupees.

The bundle includes `examples.json`, `metrics.json`, and the complete held-out
inputs and targets in `evaluation.json`, so the reported metrics can be checked.

Source: [OpenIntro Ames dataset](https://www.openintro.org/data/index.php?data=ames),
originally Dean De Cock (2011), *Ames, Iowa: Alternative to the Boston housing
data as an end of semester regression project*. Retain this attribution and
consult [OpenIntro's license terms](https://www.openintro.org/license/) before
redistributing source data; this project does not assert a new dataset license.

## 2. Male/female: pretrained model prepared for reuse

This is a compact convolutional neural network originally trained by **Cydral
Technology**. We reused its pretrained weights; we did not train this image
classifier from scratch. The weights are CC0-1.0, and the ONNX conversion comes
from [mowshon/age-and-gender](https://github.com/mowshon/age-and-gender), pinned
to revision `6ddb6ce2d598705ccbe2067967b2f8be01fd3914`.

We added normalization inside the ONNX graph and embedded the input contract
and class labels. The CNN weights are unchanged. No age or identity model is included.

Image preparation is:

1. Provide one upright face crop with little surrounding background.
2. Apply the image's EXIF orientation and convert to RGB.
3. Resize to 32 × 32 using Pillow's bilinear interpolation.
4. Supply float32 raw pixels, range 0–255, shaped `[N, 3, 32, 32]`.

The graph performs `(RGB - [122.781998, 117.000999, 104.297997]) / 256`.
Do not normalize a second time. Output columns are **`female`, then `male`**;
softmax is already in the graph. Choose the larger score.

Included examples:

| Image under `models/demo/male_female/samples/` | Predicted label | Model score |
| --- | --- | --- |
| `sample_male.png` | male | 0.9566 |
| `sample_female.png` | female | 0.9999 |

These are two smoke checks, **not a measured general accuracy**. Eight additional
random input tensors produced the same probabilities as the original normalized
network, with a maximum difference of zero in this environment. Tests also check
that batch and single-image results agree.

There is no face detector or rejection of images without faces. The upstream
pipeline uses landmark alignment; the demo's manually cropped and resized input
does not reproduce that alignment. Pose, crop, lighting, and population can affect
results. The model estimates its training dataset's binary appearance labels; it
does not establish a person's gender identity or biological sex. Scores are not
calibrated certainty.

The sample crops come from official public-domain US government photographs of
Barack Obama by Pete Souza and Michelle Obama by Joyce N. Boghosian. Source
pages, hashes, and crop coordinates are in `examples.json`; license notices and
credits are in the bundle. Original model provenance:
[davisking/dlib-models](https://github.com/davisking/dlib-models#dnn_gender_classifier_v1datbz2).

## Try them on this PC

The serving and preparation dependencies are installed in the working verification
environment. The original Python 3.11 runtime is currently blocked by Windows
Application Control; use the separate verified Python 3.12 environment below on
this PC. Use **Command Prompt**, starting with the complete folder change:

```bat
cd /d "D:\WORK\CODING\PYTHON\DC\Distributed-ML"
..\.verify-venv\Scripts\python.exe scripts\predict_demo_model.py house --features 1800 3 2 2 2000 7
..\.verify-venv\Scripts\python.exe scripts\predict_demo_model.py face --image models\demo\male_female\samples\sample_male.png
..\.verify-venv\Scripts\python.exe scripts\predict_demo_model.py face --image models\demo\male_female\samples\sample_female.png
```

To try your own image, replace the example path with a quoted path to an already
cropped face image. The predictor converts JPEG/PNG images to the required tensor.
For PowerShell, change folders using `Set-Location` instead of `cd /d`.

## Rebuild and verify

`requirements-models.txt` lists additional preparation tools. ONNX Runtime and
Pillow are serving dependencies in `requirements-prototype.txt`. A fresh
environment with pip can install them using
`python -m pip install -r requirements-models.txt`, alongside the prototype's
requirements. No GPU or PyTorch is needed.

```bat
.\.venv\Scripts\python.exe scripts\prepare_demo_models.py
.\.venv\Scripts\python.exe scripts\prepare_demo_models.py --offline
.\.venv\Scripts\python.exe -m pytest tests\test_demo_models.py -q
.\.venv\Scripts\python.exe -m pytest -q --tb=short
```

The first preparation downloads missing sources to `models/demo/.sources/` and
checks pinned SHA-256 hashes. It reuses valid cached sources. Offline preparation
uses only that cache. A checksum mismatch fails rather than silently replacing
the expected source. Rebuilding writes the two generated bundles again.

The checks cover ONNX validity, self-contained weights, metadata and artifact
hashes, independently recomputed holdout metrics, export equivalence, image
normalization, batch consistency, invalid inputs, EXIF orientation, and commands
run from a different current folder. With the optional dependencies absent, model
tests are skipped; with artifacts absent, only artifact-dependent tests are skipped.
Tests never download sources. Three offline rebuilds from the parent directory,
with different Python hash seeds, reproduced identical ONNX files in this environment.

Verified on 2026-10-22 with Python 3.11.16: **27 model tests and 214 total tests
passed**, with one existing Starlette TestClient deprecation warning. Full library
versions and measured results are saved in `models/demo/validation_report.json`.
Generated data and ONNX artifacts remain ignored by Git. Nothing was committed.
