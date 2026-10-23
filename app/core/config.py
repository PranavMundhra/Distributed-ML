"""Portable environment configuration, validated before service startup."""

from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.workers import ServiceURL

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=120, allow_inf_nan=False)
    # The local launcher generates a fresh shared secret; no public demo control by default.
    demo_control_token: SecretStr | None = Field(default=None, min_length=32, max_length=256, exclude=True)


class WorkerSettings(CommonSettings):
    demo_model_dir: Path = PROJECT_ROOT / "models" / "demo"
    worker_id: str = Field(default="cpu-worker-1", min_length=1, max_length=128,
                           pattern=r"^[A-Za-z0-9_.-]+$")
    worker_url: ServiceURL = HttpUrl("http://127.0.0.1:8001")
    gateway_url: ServiceURL | None = None
    heartbeat_interval_seconds: float = Field(default=2.0, gt=0, le=60, allow_inf_nan=False)

    @field_validator("gateway_url", mode="before")
    @classmethod
    def allow_standalone_worker(cls, value):
        return None if value == "" else value


class GatewaySettings(CommonSettings):
    """Workers are discovered through registration, rather than a fixed URL."""
    heartbeat_timeout_seconds: float = Field(default=8.0, gt=0, le=300, allow_inf_nan=False)
    heartbeat_check_interval_seconds: float = Field(default=0.5, gt=0, le=30, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_timely_expiry_checks(self):
        if self.heartbeat_check_interval_seconds > self.heartbeat_timeout_seconds:
            raise ValueError("Heartbeat check interval must not exceed the heartbeat timeout")
        return self
