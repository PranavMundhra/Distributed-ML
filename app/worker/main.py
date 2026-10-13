"""Run with python -m uvicorn app.worker.main:app --port 8001."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.core.config import WorkerSettings
from app.core.logging import install_request_logging
from app.schemas.inference import (
    MODEL_NAME, SPECIES, PredictionRequest, PredictionResponse, WorkerHealth,
)
from app.worker.model import train_model


def create_app(settings: WorkerSettings | None = None) -> FastAPI:
    config = settings if settings is not None else WorkerSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(message)s")
        application.state.model = await run_in_threadpool(train_model)
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            application.state.model = None

    application = FastAPI(title="CPU inference worker", version="0.1.0", lifespan=lifespan)
    application.state.settings = config
    application.state.ready = False
    application.state.model = None
    install_request_logging(application)

    @application.get("/health", response_model=WorkerHealth)
    def health() -> WorkerHealth:
        if not application.state.ready:
            raise HTTPException(503, "Worker model is not ready")
        return WorkerHealth(status="ok", service="worker", ready=True,
                            worker_id=config.worker_id, model=MODEL_NAME)

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
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

    return application


app = create_app()
