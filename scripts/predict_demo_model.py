"""Try the prepared models locally, without starting the gateway or workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .demo_models import DEFAULT_MODEL_DIR, predict_face, predict_house
else:
    from demo_models import DEFAULT_MODEL_DIR, predict_face, predict_house


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    commands = parser.add_subparsers(dest="task", required=True)
    house = commands.add_parser("house", help="Predict an Ames house sale price in historical USD")
    house.add_argument(
        "--features", nargs=6, type=float, required=True,
        metavar=("AREA_SQFT", "BEDROOMS", "BATHROOMS", "GARAGE_SPACES", "YEAR_BUILT", "QUALITY"),
    )
    face = commands.add_parser("face", help="Classify one already-cropped face")
    face.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.task == "house":
            prediction = predict_house(args.model_dir / "house_price" / "house_price.onnx", args.features)[0]
            result = {"predicted_price_usd": round(prediction, 2), "currency": "USD", "market": "Ames, Iowa, 2006-2010"}
        else:
            result = predict_face(args.model_dir / "male_female" / "male_female.onnx", args.image)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Unable to predict: {error}\n")


if __name__ == "__main__":
    main()
