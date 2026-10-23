"""Run with python -m uvicorn app.worker.main:app --port 8001."""

import asyncio
import logging
from secrets import compare_digest
from threading import Event
from contextlib import AsyncExitStack, asynccontextmanager, suppress

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.core.config import WorkerSettings
from app.core.logging import install_request_logging
from app.core.lamport import LamportClock
from app.schemas.inference import (
    MODEL_NAME, SPECIES, PredictionRequest, PredictionResponse, WorkerHealth,
)
from app.schemas.workers import DemoAction, DemoState, WorkerRegistration
from app.schemas.models import (
    FACE_MODEL, HOUSE_MODEL, FaceRequest, FaceResponse, HouseRequest, HouseResponse,
)
from app.worker.fixed_models import face_tensor, load_models
from app.worker.heartbeat import register_with_gateway, send_heartbeats
from app.worker.model import train_model


def create_app(
    settings: WorkerSettings | None = None, client: httpx.AsyncClient | None = None
) -> FastAPI:
    config = settings if settings is not None else WorkerSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(message)s")
        application.state.lamport = LamportClock(config.worker_id)
        application.state.demo_paused = Event()
        application.state.model = await run_in_threadpool(train_model)
        application.state.demo_models = await run_in_threadpool(load_models, config.demo_model_dir)
        async with AsyncExitStack() as stack:
            task = None
            try:
                if config.gateway_url is not None:
                    upstream = client if client is not None else await stack.enter_async_context(
                        httpx.AsyncClient(trust_env=False, follow_redirects=False)
                    )
                    registration = WorkerRegistration(
                        worker_id=config.worker_id, worker_url=config.worker_url,
                        instance_id=application.state.lamport.instance_id,
                        models=[MODEL_NAME, *application.state.demo_models],
                        demo_control_available=config.demo_control_token is not None,
                    )
                    registered = await register_with_gateway(config, upstream, registration, application.state.lamport)
                    task = asyncio.create_task(
                        send_heartbeats(config, upstream, registration, registered, application.state.lamport,
                                        paused=application.state.demo_paused), name="worker-heartbeats",
                    )
                application.state.heartbeat_task = task
                application.state.ready = True
                yield
            finally:
                application.state.ready = False
                if task is not None:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                application.state.heartbeat_task = None
                application.state.model = None
                application.state.demo_models = {}

    application = FastAPI(title="CPU inference worker", version="0.6.0", lifespan=lifespan)
    application.state.settings = config
    application.state.ready = False
    application.state.model = None
    application.state.demo_models = {}
    application.state.demo_paused = Event()
    application.state.heartbeat_task = None
    application.state.lamport = LamportClock(config.worker_id)
    install_request_logging(application)

    def require_running():
        if application.state.demo_paused.is_set():
            raise HTTPException(503, "Worker is paused for the failure demonstration")

    @application.post("/demo/control", response_model=DemoState, include_in_schema=False)
    async def demo_control(payload: DemoAction, request: Request) -> DemoState:
        secret = config.demo_control_token
        if secret is None:
            raise HTTPException(404, "Demo controls are disabled")
        supplied = request.headers.get("X-Demo-Token", "")
        if not compare_digest(supplied.encode(), secret.get_secret_value().encode()):
            raise HTTPException(403, "Invalid demo control token")
        if not application.state.ready:
            raise HTTPException(503, "Worker is not ready")
        if payload.instance_id != application.state.lamport.instance_id:
            raise HTTPException(409, "Worker process changed; refresh the worker page")
        if payload.action == "pause":
            application.state.demo_paused.set()
        else:
            application.state.demo_paused.clear()
        return DemoState(worker_id=config.worker_id, instance_id=payload.instance_id,
                         paused=application.state.demo_paused.is_set())

    @application.get("/health", response_model=WorkerHealth)
    def health() -> WorkerHealth:
        require_running()
        if not application.state.ready:
            raise HTTPException(503, "Worker model is not ready")
        return WorkerHealth(status="ok", service="worker", ready=True,
                            worker_id=config.worker_id, model=MODEL_NAME)

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        require_running()
        model = application.state.model
        if model is None or not application.state.ready:
            raise HTTPException(503, "Worker model is not ready")
        probabilities = model.predict_proba([payload.features])[0]
        class_id = int(model.classes_[probabilities.argmax()])
        request.state.worker_id = config.worker_id
        return PredictionResponse(
            prediction=SPECIES[class_id], class_id=class_id,
            probabilities={species: float(value) for species, value in zip(SPECIES, probabilities)},
            model=MODEL_NAME, worker_id=config.worker_id, request_id=request.state.request_id,
        )

    def require_model(model_id: str):
        require_running()
        session = application.state.demo_models.get(model_id)
        if not application.state.ready or session is None:
            raise HTTPException(503, "This model is not loaded on this worker; prepare the demo artifacts and restart")
        return session

    @application.post("/predict/house", response_model=HouseResponse)
    def predict_house(payload: HouseRequest, request: Request) -> HouseResponse:
        session = require_model(HOUSE_MODEL)
        values = np.asarray([payload.features()], dtype=np.float32)
        result = session.run(["price_usd"], {"features": values})[0]
        if result.shape != (1, 1) or not np.isfinite(result).all():
            raise HTTPException(500, "Model returned an invalid house price")
        request.state.worker_id = config.worker_id
        return HouseResponse(predicted_price_usd=float(result[0, 0]),
                             worker_id=config.worker_id, request_id=request.state.request_id)

    @application.post("/predict/face", response_model=FaceResponse)
    def predict_face(payload: FaceRequest, request: Request) -> FaceResponse:
        session = require_model(FACE_MODEL)
        try:
            values = face_tensor(payload.image_bytes())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result = session.run(["probabilities"], {"images": values})[0]
        if result.shape != (1, 2) or not np.isfinite(result).all():
            raise HTTPException(500, "Model returned invalid class probabilities")
        labels = ("female", "male")
        request.state.worker_id = config.worker_id
        return FaceResponse(prediction=labels[int(result[0].argmax())],
                            probabilities=dict(zip(labels, map(float, result[0]))),
                            worker_id=config.worker_id, request_id=request.state.request_id)

    return application


app = create_app()
