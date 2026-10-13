"""The same explicit contract is enforced at both HTTP boundaries."""

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPECIES = ("setosa", "versicolor", "virginica")
MODEL_NAME = "iris-logistic-regression-v1"
Species = Literal["setosa", "versicolor", "virginica"]
Measurement = Annotated[float, Field(strict=True, gt=0, le=100, allow_inf_nan=False)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: list[Measurement] = Field(
        min_length=4, max_length=4,
        description="Sepal length, sepal width, petal length, petal width in centimeters; each >0 and <=100.",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prediction: Species
    class_id: Annotated[int, Field(ge=0, le=2)]
    probabilities: dict[Species, Probability]
    model: Literal["iris-logistic-regression-v1"]
    worker_id: Identifier
    request_id: Identifier

    @model_validator(mode="after")
    def require_consistent_prediction(self) -> Self:
        if set(self.probabilities) != set(SPECIES):
            raise ValueError("probabilities must include all three species")
        if SPECIES[self.class_id] != self.prediction:
            raise ValueError("class_id must match prediction")
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        if self.probabilities[self.prediction] != max(self.probabilities.values()):
            raise ValueError("prediction must have the largest probability")
        return self


class WorkerHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ok"]
    service: Literal["worker"]
    ready: bool
    worker_id: Identifier
    model: Literal["iris-logistic-regression-v1"]


class GatewayHealth(BaseModel):
    status: Literal["ok", "degraded"]
    service: Literal["gateway"] = "gateway"
    ready: bool
    worker_available: bool
    worker_url: str
    worker_id: str | None = None
