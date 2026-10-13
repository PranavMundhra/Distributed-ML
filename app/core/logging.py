"""Request correlation for success, validation errors, and upstream failures."""

import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("inference")
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def install_request_logging(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo raw values: nonfinite numbers cannot be serialized as JSON.
        errors = [
            {key: error[key] for key in ("type", "loc", "msg") if key in error}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid4())
        )
        started = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            LOGGER.info(
                "request_id=%s service=%s method=%s path=%s status=%s elapsed_ms=%.2f worker_id=%s",
                request.state.request_id, app.title, request.method, request.url.path,
                status, (perf_counter() - started) * 1000,
                getattr(request.state, "worker_id", "-"),
            )
