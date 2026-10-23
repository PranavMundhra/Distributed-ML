"""Three fixed demo models, each with an explicit input/output contract."""

import base64
import binascii
import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.inference import MODEL_NAME, Identifier, Probability

HOUSE_MODEL = "house-price-linear-v1"
FACE_MODEL = "male-female-cydral-v1"
ModelID = Literal["iris-logistic-regression-v1", "house-price-linear-v1", "male-female-cydral-v1"]
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BASE64 = 4 * ((MAX_IMAGE_BYTES + 2) // 3)


class HouseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    area_sqft: Annotated[float, Field(ge=334, le=5642, allow_inf_nan=False)]
    bedrooms: Annotated[int, Field(ge=0, le=8)]
    full_bathrooms: Annotated[int, Field(ge=0, le=4)]
    garage_spaces: Annotated[int, Field(ge=0, le=4)]
    year_built: Annotated[int, Field(ge=1872, le=2010)]
    overall_quality: Annotated[int, Field(ge=1, le=10)]

    def features(self) -> list[float]:
        return [self.area_sqft, self.bedrooms, self.full_bathrooms,
                self.garage_spaces, self.year_built, self.overall_quality]


class HouseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    predicted_price_usd: Annotated[float, Field(allow_inf_nan=False)]
    currency: Literal["USD"] = "USD"
    market: Literal["Ames, Iowa, 2006-2010"] = "Ames, Iowa, 2006-2010"
    model: Literal["house-price-linear-v1"] = HOUSE_MODEL
    worker_id: Identifier
    request_id: Identifier


class FaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_base64: str = Field(min_length=1, max_length=MAX_IMAGE_BASE64,
                             description="Raw base64 of a PNG/JPEG still image, at most 5 MiB; no data-URL prefix.")

    @field_validator("image_base64")
    @classmethod
    def require_base64(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Supply valid base64 image data") from exc
        if not decoded or len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError("Image must be nonempty and at most 5 MiB")
        return value

    def image_bytes(self) -> bytes:
        return base64.b64decode(self.image_base64, validate=True)


class FaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prediction: Literal["female", "male"]
    probabilities: dict[Literal["female", "male"], Probability]
    model: Literal["male-female-cydral-v1"] = FACE_MODEL
    worker_id: Identifier
    request_id: Identifier

    @model_validator(mode="after")
    def require_consistent_prediction(self) -> Self:
        if set(self.probabilities) != {"female", "male"}:
            raise ValueError("Both face labels are required")
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        if self.probabilities[self.prediction] != max(self.probabilities.values()):
            raise ValueError("Prediction must have the largest probability")
        return self


class ModelAvailability(BaseModel):
    model: ModelID
    title: str
    available_workers: int


FIXED_MODELS = (
    (MODEL_NAME, "Iris classification"),
    (HOUSE_MODEL, "House price prediction"),
    (FACE_MODEL, "Male/female image classification"),
)
