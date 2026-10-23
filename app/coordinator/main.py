"""Registered-worker gateway; run one uvicorn app.coordinator.main:app process."""

import asyncio
import logging
from contextlib import asynccontextmanager
from contextlib import suppress
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.coordinator.registry import HeartbeatConflict, RegistrationConflict, UnknownWorker, WorkerRegistry
from app.core.config import GatewaySettings
from app.core.config import PROJECT_ROOT
from app.core.logging import LOGGER, install_request_logging
from app.core.lamport import InferenceTrace, LamportClock, TraceStore, clocked_request
from app.schemas.inference import (
    GatewayHealth, PredictionRequest, PredictionResponse, WorkerHealth,
)
from app.schemas.workers import (
    ClusterStatus, DemoAction, DemoState, RegisteredWorker, WorkerHeartbeat, WorkerRegistration,
)
from app.schemas.models import (
    FACE_MODEL, HOUSE_MODEL, FIXED_MODELS, FaceRequest, FaceResponse,
    HouseRequest, HouseResponse, ModelAvailability,
)
from app.schemas.inference import MODEL_NAME
from app.worker.fixed_models import face_tensor

STATIC_DIRECTORY = Path(__file__).resolve().parents[1] / "static"


def create_app(
    settings: GatewaySettings | None = None, client: httpx.AsyncClient | None = None
) -> FastAPI:
    """An injected HTTP client belongs to the caller and is never closed here."""
    config = settings if settings is not None else GatewaySettings()
    timeout = httpx.Timeout(config.request_timeout_seconds)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(message)s")
        application.state.lamport = LamportClock("gateway")
        application.state.traces = TraceStore()
        application.state.client = client if client is not None else httpx.AsyncClient(
            timeout=timeout, trust_env=False, follow_redirects=False,
        )
        application.state.registry = WorkerRegistry(config.heartbeat_timeout_seconds)

        async def expire_stale_workers():
            while True:
                await asyncio.sleep(config.heartbeat_check_interval_seconds)
                application.state.registry.expire_workers()

        expiry_task = asyncio.create_task(expire_stale_workers(), name="worker-heartbeat-expiry")
        application.state.expiry_task = expiry_task
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            expiry_task.cancel()
            with suppress(asyncio.CancelledError):
                await expiry_task
            application.state.expiry_task = None
            if client is None:
                await application.state.client.aclose()
            application.state.client = None
            application.state.registry = WorkerRegistry(config.heartbeat_timeout_seconds)

    application = FastAPI(title="Inference gateway", version="0.6.0", lifespan=lifespan)
    application.state.settings = config
    application.state.ready = False
    application.state.client = None
    application.state.registry = WorkerRegistry(config.heartbeat_timeout_seconds)
    application.state.expiry_task = None
    application.state.lamport = LamportClock("gateway")
    application.state.traces = TraceStore()
    install_request_logging(application)

    def require_client() -> httpx.AsyncClient:
        if not application.state.ready or application.state.client is None:
            raise HTTPException(503, "Gateway is not ready")
        return application.state.client

    @application.post("/workers/register", response_model=RegisteredWorker, status_code=201)
    async def register(payload: WorkerRegistration, response: Response) -> RegisteredWorker:
        require_client()
        try:
            worker, created = application.state.registry.register(payload)
        except RegistrationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        response.status_code = 201 if created else 200
        LOGGER.info("worker_registered worker_id=%s worker_url=%s", worker.worker_id, worker.worker_url)
        return worker

    @application.get("/workers", response_model=list[RegisteredWorker])
    async def workers(request: Request, response: Response):
        require_client()
        response.headers["Vary"] = "Accept"
        if "text/html" in request.headers.get("Accept", "") and request.query_params.get("format") != "json":
            return FileResponse(STATIC_DIRECTORY / "workers.html", headers={"Vary": "Accept", "Cache-Control": "no-store"})
        return application.state.registry.list_workers()

    @application.get("/workers/view", include_in_schema=False)
    def worker_page():
        return FileResponse(STATIC_DIRECTORY / "workers.html", headers={"Cache-Control": "no-store"})

    @application.get("/workers/status", response_model=ClusterStatus)
    async def worker_status() -> ClusterStatus:
        require_client()
        registry = application.state.registry
        # Read-only observation: no health probes, scheduling turns or lease renewal.
        return ClusterStatus(nodes=registry.node_statuses(),
                             recent_predictions=list(reversed(registry.recent_predictions)),
                             demo_controls_enabled=config.demo_control_token is not None)

    @application.post("/workers/{worker_id}/demo", response_model=DemoState, include_in_schema=False)
    async def control_worker(worker_id: str, payload: DemoAction, request: Request) -> DemoState:
        upstream = require_client()
        secret = config.demo_control_token
        if secret is None:
            raise HTTPException(404, "Demo controls are disabled; use the local launcher")
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "Demo controls are only available from this computer")
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != f"{request.url.scheme}://{request.url.netloc}":
            raise HTTPException(403, "Use the worker page on this gateway")
        registry = application.state.registry
        worker = registry.get_worker(worker_id)
        if worker is None:
            raise HTTPException(404, "Worker is not registered")
        if worker.instance_id != payload.instance_id:
            raise HTTPException(409, "Worker process changed; refresh the worker page")
        if not worker.demo_control_available:
            raise HTTPException(409, "This worker was started without demo controls")
        request.state.worker_id = worker.worker_id
        try:
            async with asyncio.timeout(config.request_timeout_seconds):
                response, _ = await clocked_request(
                    upstream, application.state.lamport, "POST",
                    f"{str(worker.worker_url).rstrip('/')}/demo/control",
                    headers={"X-Demo-Token": secret.get_secret_value()},
                    json=payload.model_dump(), timeout=timeout, request_id=request.state.request_id,
                    peer_id=worker.worker_id,
                )
            if response.status_code == 409:
                raise HTTPException(409, "Worker process changed; refresh the worker page")
            if response.status_code != 200:
                raise ValueError("Worker rejected demo control")
            state = DemoState.model_validate(response.json())
            if (state.worker_id != worker.worker_id or state.instance_id != worker.instance_id
                    or state.paused != (payload.action == "pause")):
                raise ValueError("Invalid worker demo acknowledgement")
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise HTTPException(504, "Worker control timed out") from exc
        except httpx.ConnectError as exc:
            raise HTTPException(503, "Worker process is unreachable; restart it from its terminal") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise HTTPException(502, "Could not confirm worker control") from exc
        if registry.get_worker(worker_id) is not worker:
            raise HTTPException(409, "Worker registration changed; refresh the worker page")
        # Record the acknowledged demo state without changing eligibility or heartbeat deadlines.
        worker.demo_paused = state.paused
        return state

    @application.get("/models", response_model=list[ModelAvailability])
    async def models() -> list[ModelAvailability]:
        require_client()
        registered = application.state.registry.list_workers()
        return [ModelAvailability(model=model_id, title=title, available_workers=sum(
            model_id in worker.models and worker.available is not False for worker in registered
        )) for model_id, title in FIXED_MODELS]

    @application.get("/traces/{trace_id}", response_model=InferenceTrace)
    async def trace(trace_id: str) -> InferenceTrace:
        require_client()
        result = application.state.traces.get(trace_id)
        if result is None:
            raise HTTPException(404, "Trace was not found; only the last 200 requests are retained until restart")
        return result

    @application.post("/workers/heartbeat", response_model=RegisteredWorker)
    async def heartbeat(payload: WorkerHeartbeat) -> RegisteredWorker:
        require_client()
        try:
            return application.state.registry.heartbeat(payload)
        except UnknownWorker as exc:
            raise HTTPException(404, str(exc)) from exc
        except HeartbeatConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get("/health", response_model=GatewayHealth)
    async def health(request: Request) -> GatewayHealth:
        """200 means the gateway is ready; worker_available describes dependency readiness."""
        upstream = require_client()
        registry = application.state.registry
        registered = registry.list_workers()

        async def probe(worker: RegisteredWorker) -> bool:
            available = False
            try:
                async with asyncio.timeout(config.request_timeout_seconds):
                    response, _ = await clocked_request(
                        upstream, application.state.lamport, "GET", f"{str(worker.worker_url).rstrip('/')}/health", timeout=timeout,
                        request_id=request.state.request_id, peer_id=worker.worker_id,
                    )
                if response.status_code == 200:
                    candidate = WorkerHealth.model_validate(response.json())
                    available = candidate.ready and candidate.worker_id == worker.worker_id
            except (TimeoutError, httpx.RequestError, ValidationError, ValueError):
                LOGGER.warning("worker_health_unavailable worker_id=%s request_id=%s",
                               worker.worker_id, request.state.request_id)
            registry.set_available(worker, available)
            return worker.available is True

        # Probe in parallel so a health request is bounded by one timeout, not N.
        results = await asyncio.gather(*(probe(worker) for worker in registered))
        available = [worker for worker, ready in zip(registered, results) if ready]
        first = available[0] if available else None
        if first is not None:
            request.state.worker_id = first.worker_id
        return GatewayHealth(
            status="ok" if available else "degraded", ready=True,
            worker_available=bool(available),
            worker_url=str(first.worker_url).rstrip("/") if first else None,
            worker_id=first.worker_id if first else None,
            registered_workers=len(registered), available_workers=len(available),
        )

    async def forward_prediction(payload, request: Request, model_id: str, path: str, response_schema):
        upstream = require_client()
        registry = application.state.registry
        worker = registry.select(model_id)
        if worker is None:
            detail = "No registered workers are available" if model_id == MODEL_NAME else (
                "No available worker has this model loaded; prepare the demo models on each worker and restart"
            )
            raise HTTPException(503, detail)
        worker_url = str(worker.worker_url).rstrip("/")
        request.state.worker_id = worker.worker_id
        request_id = request.state.request_id
        try:
            async with asyncio.timeout(config.request_timeout_seconds):
                response, _ = await clocked_request(
                    upstream, application.state.lamport, "POST", f"{worker_url}{path}",
                    json=payload.model_dump(), timeout=timeout, request_id=request_id, peer_id=worker.worker_id,
                    trace_id=request.state.trace_id, events=request.state.clock_events,
                )
        except (TimeoutError, httpx.TimeoutException) as exc:
            registry.set_available(worker, False)
            raise HTTPException(504, "Worker request timed out") from exc
        except httpx.ConnectError as exc:
            registry.set_available(worker, False)
            raise HTTPException(503, "Worker is unavailable") from exc
        except httpx.RequestError as exc:
            registry.set_available(worker, False)
            raise HTTPException(502, "Worker communication failed") from exc
        except ValueError as exc:
            registry.set_available(worker, False)
            raise HTTPException(502, "Worker returned invalid logical clock metadata") from exc

        if response.status_code == 503:
            registry.set_available(worker, False)
            raise HTTPException(503, "Worker is unavailable")
        if response.status_code != 200:
            registry.set_available(worker, False)
            raise HTTPException(502, "Worker returned an unexpected response")
        try:
            prediction = response_schema.model_validate(response.json())
            if prediction.worker_id != worker.worker_id:
                raise ValueError("Worker returned a mismatched worker ID")
            upstream_id = response.headers.get("X-Request-ID")
            if prediction.request_id != request_id or upstream_id not in (None, request_id):
                raise ValueError("Worker returned a mismatched request ID")
        except (ValidationError, ValueError) as exc:
            registry.set_available(worker, False)
            raise HTTPException(502, "Worker returned an invalid prediction") from exc
        request.state.worker_id = prediction.worker_id
        registry.set_available(worker, True)
        registry.record_prediction(worker, model_id, request_id, request.state.trace_id)
        return prediction

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        return await forward_prediction(payload, request, MODEL_NAME, "/predict", PredictionResponse)

    @application.post("/predict/house", response_model=HouseResponse)
    async def predict_house(payload: HouseRequest, request: Request) -> HouseResponse:
        return await forward_prediction(payload, request, HOUSE_MODEL, "/predict/house", HouseResponse)

    @application.post("/predict/face", response_model=FaceResponse)
    async def predict_face(payload: FaceRequest, request: Request) -> FaceResponse:
        require_client()
        try:
            await run_in_threadpool(face_tensor, payload.image_bytes())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return await forward_prediction(payload, request, FACE_MODEL, "/predict/face", FaceResponse)

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @application.get("/samples/face/{label}", include_in_schema=False)
    def face_sample(label: Literal["female", "male"]) -> FileResponse:
        path = PROJECT_ROOT / "models" / "demo" / "male_female" / "samples" / f"sample_{label}.png"
        if not path.is_file():
            raise HTTPException(404, "Prepare the demo models to use example images")
        return FileResponse(path, media_type="image/png")

    application.mount("/static", StaticFiles(directory=STATIC_DIRECTORY, check_dir=False), name="static")
    return application


app = create_app()
