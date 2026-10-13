"""Portable environment configuration, validated before service startup."""

from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class WorkerSettings(CommonSettings):
    worker_id: str = Field(default="cpu-worker-1", min_length=1, max_length=128,
                           pattern=r"^[A-Za-z0-9_.-]+$")


class GatewaySettings(CommonSettings):
    worker_url: HttpUrl = HttpUrl("http://127.0.0.1:8001")
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=120, allow_inf_nan=False)

    @field_validator("worker_url")
    @classmethod
    def require_base_url(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("worker_url must not include credentials, query, or fragment")
        if value.path not in (None, "", "/"):
            raise ValueError("worker_url must be an HTTP service origin without a path")
        return value
