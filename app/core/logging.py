"""Request correlation for success, validation errors, and upstream failures."""

import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

from app.core.lamport import (
    INSTANCE_HEADER, RECEIVED_HEADER, TIME_HEADER, TRACE_HEADER,
    InferenceTrace, event, log_event, parse_time,
)

LOGGER = logging.getLogger("inference")
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class RequestLoggingMiddleware:
    """Record actual ASGI message boundaries without buffering response bodies."""
    def __init__(self, app, application: FastAPI):
        self.app = app
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        app = self.application
        request = Request(scope)
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid4())
        )
        started = perf_counter()
        status = 500
        clock = app.state.lamport
        supplied_times = request.headers.getlist(TIME_HEADER)
        error = None
        try:
            if len(supplied_times) > 1:
                raise ValueError("Supply only one Lamport timestamp")
            remote_time = parse_time(supplied_times[0]) if supplied_times else None
        except ValueError as exc:
            remote_time = None
            error = str(exc)
        received = clock.receive(remote_time) if remote_time is not None else clock.tick()
        request.state.lamport_received = received
        is_prediction = request.url.path in {"/predict", "/predict/house", "/predict/face"} and request.method == "POST"
        is_gateway = hasattr(app.state, "traces")
        if is_prediction:
            incoming_trace = request.headers.get(TRACE_HEADER, "")
            # Each gateway attempt has a fresh ID even if clients reuse request IDs.
            request.state.trace_id = str(uuid4()) if is_gateway or not re.fullmatch(r"[a-f0-9-]{36}", incoming_trace) else incoming_trace
            request.state.clock_events = [event(clock, "request_received", received)]
        log_event(clock, "request_received", received, request.state.request_id, request.url.path)
        async def send_response(message):
            nonlocal status
            if message["type"] != "http.response.start":
                await send(message)
                return
            status = message["status"]
            sent = clock.tick()
            headers = MutableHeaders(scope=message)
            headers[TIME_HEADER] = str(sent)
            headers[RECEIVED_HEADER] = str(received)
            headers[INSTANCE_HEADER] = clock.instance_id
            headers["X-Request-ID"] = request.state.request_id
            if is_prediction:
                headers[TRACE_HEADER] = request.state.trace_id
                if is_gateway:
                    events = request.state.clock_events
                    events.append(event(clock, "response_sent", sent))
                    complete = [item.event for item in events] == [
                        "request_received", "request_forwarded", "worker_request_received",
                        "worker_response_sent", "worker_response_received", "response_sent",
                    ]
                    app.state.traces.add(InferenceTrace(
                        trace_id=request.state.trace_id, request_id=request.state.request_id,
                        status_code=status, complete=complete, events=tuple(events),
                    ))
            log_event(clock, "response_sent", sent, request.state.request_id, request.url.path)
            await send(message)

        try:
            if error:
                await JSONResponse(status_code=400, content={"detail": error})(scope, receive, send_response)
            elif request.url.path == "/predict/face" and request.method == "POST":
                # Bound the streamed JSON body before parsing or base64 decoding.
                chunks = []
                size = 0
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > 7 * 1024 * 1024:
                        await JSONResponse(status_code=413, content={"detail": "Image request is too large; use an image at most 5 MiB"})(scope, receive, send_response)
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
                body = b"".join(chunks)
                chunks.clear()
                delivered = False

                async def replay_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await receive()

                await self.app(scope, replay_receive, send_response)
            else:
                await self.app(scope, receive, send_response)
        finally:
            LOGGER.info(
                "request_id=%s service=%s method=%s path=%s status=%s elapsed_ms=%.2f worker_id=%s",
                request.state.request_id, app.title, request.method, request.url.path,
                status, (perf_counter() - started) * 1000,
                getattr(request.state, "worker_id", "-"),
            )


def install_request_logging(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo raw values: nonfinite numbers cannot be serialized as JSON.
        errors = [
            {key: error[key] for key in ("type", "loc", "msg") if key in error}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.add_middleware(RequestLoggingMiddleware, application=app)
