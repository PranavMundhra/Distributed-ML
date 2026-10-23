"""Bounded periodic heartbeats using the worker's existing HTTP client."""

import asyncio
import logging
from threading import Event

import httpx

from app.core.config import WorkerSettings
from app.core.lamport import LamportClock, clocked_request
from app.schemas.workers import RegisteredWorker, WorkerRegistration

LOGGER = logging.getLogger("inference")


async def register_with_gateway(
    config: WorkerSettings, client: httpx.AsyncClient, registration: WorkerRegistration,
    clock: LamportClock,
) -> RegisteredWorker:
    try:
        async with asyncio.timeout(config.request_timeout_seconds):
            response, _ = await clocked_request(
                client, clock, "POST", f"{str(config.gateway_url).rstrip('/')}/workers/register",
                json=registration.model_dump(mode="json"),
                timeout=config.request_timeout_seconds, request_id=f"register-{clock.instance_id}", peer_id="gateway",
            )
        if response.status_code not in (200, 201):
            raise ValueError(f"Gateway returned HTTP {response.status_code}")
        registered = RegisteredWorker.model_validate(response.json())
        if (registered.worker_id, registered.worker_url, registered.models, registered.instance_id) != (
            registration.worker_id, registration.worker_url, registration.models, registration.instance_id
        ):
            raise ValueError("Gateway returned a mismatched registration")
        if registered.demo_control_available != registration.demo_control_available:
            raise ValueError("Gateway returned mismatched demo capabilities")
        if config.heartbeat_interval_seconds >= registered.heartbeat_timeout_seconds:
            raise ValueError("HEARTBEAT_INTERVAL_SECONDS must be less than gateway HEARTBEAT_TIMEOUT_SECONDS")
    except (TimeoutError, httpx.RequestError, ValueError) as exc:
        raise RuntimeError(f"Worker registration failed: {exc}. Start the gateway and check GATEWAY_URL and heartbeat settings") from exc
    LOGGER.info("worker_registered worker_id=%s gateway_url=%s", config.worker_id, config.gateway_url)
    return registered


async def send_heartbeats(
    config: WorkerSettings, client: httpx.AsyncClient, registration: WorkerRegistration,
    registered: RegisteredWorker,
    clock: LamportClock,
    paused: Event | None = None,
) -> None:
    sequence = registered.heartbeat_sequence
    deadline = min(config.request_timeout_seconds, config.heartbeat_interval_seconds)
    while True:
        await asyncio.sleep(config.heartbeat_interval_seconds)
        if paused is not None and paused.is_set():
            continue
        sequence += 1
        try:
            async with asyncio.timeout(deadline):
                response, _ = await clocked_request(
                    client, clock, "POST", f"{str(config.gateway_url).rstrip('/')}/workers/heartbeat",
                    json={"worker_id": config.worker_id, "instance_id": registration.instance_id,
                          "sequence": sequence},
                    timeout=deadline, request_id=f"heartbeat-{clock.instance_id}-{sequence}", peer_id="gateway",
                )
            if response.status_code == 404:
                # A restarted gateway has an empty registry. Keep this process's
                # sequence increasing so delayed old messages stay stale.
                registered = await register_with_gateway(config, client, registration, clock)
                sequence = max(sequence, registered.heartbeat_sequence)
                continue
            if 400 <= response.status_code < 500:
                LOGGER.error("heartbeat_rejected worker_id=%s status=%s; restart worker to register again",
                             config.worker_id, response.status_code)
                return
            if response.status_code != 200:
                raise ValueError(f"Heartbeat returned HTTP {response.status_code}")
            acknowledged = RegisteredWorker.model_validate(response.json())
            if (acknowledged.worker_id != config.worker_id
                    or acknowledged.instance_id != registration.instance_id
                    or acknowledged.heartbeat_sequence != sequence
                    or acknowledged.available is not True or acknowledged.heartbeat_expired):
                raise ValueError("Gateway returned an invalid heartbeat acknowledgement")
        except (TimeoutError, httpx.RequestError, ValueError, RuntimeError) as exc:
            LOGGER.warning("heartbeat_failed worker_id=%s error=%s", config.worker_id, exc)
