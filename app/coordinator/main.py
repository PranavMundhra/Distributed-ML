"""Single-worker HTTP gateway; run with uvicorn app.coordinator.main:app."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.core.config import GatewaySettings
from app.core.logging import LOGGER, install_request_logging
from app.schemas.inference import (
    GatewayHealth, PredictionRequest, PredictionResponse, WorkerHealth,
)

STATIC_DIRECTORY = Path(__file__).resolve().parents[1] / "static"


def create_app(
    settings: GatewaySettings | None = None, client: httpx.AsyncClient | None = None
) -> FastAPI:
    """An injected HTTP client belongs to the caller and is never closed here."""
    config = settings if settings is not None else GatewaySettings()
    worker_url = str(config.worker_url).rstrip("/")
    timeout = httpx.Timeout(config.request_timeout_seconds)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(message)s")
        application.state.client = client if client is not None else httpx.AsyncClient(
            timeout=timeout, trust_env=False, follow_redirects=False,
        )
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            if client is None:
                await application.state.client.aclose()
            application.state.client = None

    application = FastAPI(title="Inference gateway", version="0.1.0", lifespan=lifespan)
    application.state.settings = config
    application.state.ready = False
    application.state.client = None
    install_request_logging(application)

    def require_client() -> httpx.AsyncClient:
        if not application.state.ready or application.state.client is None:
            raise HTTPException(503, "Gateway is not ready")
        return application.state.client

    @application.get("/health", response_model=GatewayHealth)
    async def health(request: Request) -> GatewayHealth:
        """200 means the gateway is ready; worker_available describes dependency readiness."""
        upstream = require_client()
        worker = None
        try:
            # HTTPX limits each I/O operation; also bound the complete exchange.
            async with asyncio.timeout(config.request_timeout_seconds):
                response = await upstream.get(
                    f"{worker_url}/health", timeout=timeout,
                    headers={"X-Request-ID": request.state.request_id},
                    follow_redirects=False,
                )
            if response.status_code == 200:
                candidate = WorkerHealth.model_validate(response.json())
                if candidate.ready:
                    worker = candidate
        except (TimeoutError, httpx.RequestError, ValidationError, ValueError):
            LOGGER.warning("worker_health_unavailable request_id=%s", request.state.request_id)
        if worker is not None:
            request.state.worker_id = worker.worker_id
        return GatewayHealth(
            status="ok" if worker is not None else "degraded", ready=True,
            worker_available=worker is not None, worker_url=worker_url,
            worker_id=worker.worker_id if worker is not None else None,
        )

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        upstream = require_client()
        request_id = request.state.request_id
        try:
            async with asyncio.timeout(config.request_timeout_seconds):
                response = await upstream.post(
                    f"{worker_url}/predict", json=payload.model_dump(), timeout=timeout,
                    headers={"X-Request-ID": request_id}, follow_redirects=False,
                )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise HTTPException(504, "Worker request timed out") from exc
        except httpx.ConnectError as exc:
            raise HTTPException(503, "Worker is unavailable") from exc
        except httpx.RequestError as exc:
            raise HTTPException(502, "Worker communication failed") from exc

        if response.status_code == 503:
            raise HTTPException(503, "Worker is unavailable")
        if response.status_code != 200:
            raise HTTPException(502, "Worker returned an unexpected response")
        try:
            prediction = PredictionResponse.model_validate(response.json())
            upstream_id = response.headers.get("X-Request-ID")
            if prediction.request_id != request_id or upstream_id not in (None, request_id):
                raise ValueError("Worker returned a mismatched request ID")
        except (ValidationError, ValueError) as exc:
            raise HTTPException(502, "Worker returned an invalid prediction") from exc
        request.state.worker_id = prediction.worker_id
        return prediction

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_DIRECTORY, check_dir=False), name="static")
    return application


app = create_app()
