"""Registration contract and service origins shared by both processes."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.schemas.inference import MODEL_NAME, Identifier
from app.schemas.models import ModelID


def require_origin(value: HttpUrl) -> HttpUrl:
    if value.username or value.password or value.query or value.fragment:
        raise ValueError("Service URL must not include credentials, query, or fragment")
    if value.path not in (None, "", "/"):
        raise ValueError("Service URL must be an HTTP origin without a path")
    return value


ServiceURL = Annotated[HttpUrl, AfterValidator(require_origin)]


class WorkerRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    worker_id: Identifier
    worker_url: ServiceURL
    model: Literal["iris-logistic-regression-v1"] = MODEL_NAME
    models: list[ModelID] = Field(default_factory=lambda: [MODEL_NAME], min_length=1, max_length=3)
    instance_id: Identifier | None = None
    demo_control_available: bool = False

    @field_validator("models")
    @classmethod
    def require_unique_models(cls, value):
        if len(value) != len(set(value)) or MODEL_NAME not in value:
            raise ValueError("Advertise unique fixed models including Iris")
        return value


class RegisteredWorker(WorkerRegistration):
    # The coordinator owns this mutable status object on its event loop.
    model_config = ConfigDict(extra="forbid", frozen=False)
    # None means not yet observed, rather than claiming registration proves health.
    available: bool | None = None
    heartbeat_timeout_seconds: float = Field(default=8.0, gt=0, allow_inf_nan=False)
    last_heartbeat_at: datetime | None = None
    heartbeat_sequence: int = Field(default=0, ge=0)
    heartbeat_expired: bool = False
    demo_paused: bool = False
    completed_predictions: int = Field(default=0, ge=0)
    last_prediction_at: datetime | None = None


class WorkerHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    worker_id: Identifier
    instance_id: Identifier
    sequence: int = Field(ge=1)


class DemoAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["pause", "resume"]
    instance_id: Identifier


class DemoState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    worker_id: Identifier
    instance_id: Identifier
    paused: bool


class NodeStatus(BaseModel):
    worker: RegisteredWorker
    lease_remaining_seconds: float


class CompletedPrediction(BaseModel):
    worker_id: Identifier
    instance_id: Identifier | None
    model: ModelID
    request_id: Identifier
    trace_id: str
    completed_at: datetime


class ClusterStatus(BaseModel):
    nodes: list[NodeStatus]
    recent_predictions: list[CompletedPrediction]
    demo_controls_enabled: bool
