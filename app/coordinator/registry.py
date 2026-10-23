"""In-memory registry for one coordinator process.

Methods run without awaits on the coordinator's event loop, so registration and
selection cannot interleave halfway through an update. Run one Uvicorn process.
"""

from collections.abc import Callable
from collections import deque
from datetime import datetime, timezone
import logging
from time import monotonic

from app.schemas.workers import (
    CompletedPrediction, NodeStatus, RegisteredWorker, WorkerHeartbeat, WorkerRegistration,
)
from app.schemas.inference import MODEL_NAME

LOGGER = logging.getLogger("inference")


class RegistrationConflict(ValueError):
    pass


class UnknownWorker(ValueError):
    pass


class HeartbeatConflict(ValueError):
    pass


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_seconds: float = 8.0, clock: Callable[[], float] = monotonic) -> None:
        self._workers: dict[str, RegisteredWorker] = {}
        self._last_seen: dict[str, float] = {}
        self._next: dict[str, int] = {}
        self._timeout = heartbeat_timeout_seconds
        self._clock = clock
        self.recent_predictions: deque[CompletedPrediction] = deque(maxlen=20)

    def register(self, registration: WorkerRegistration) -> tuple[RegisteredWorker, bool]:
        existing = self._workers.get(registration.worker_id)
        if existing is not None and existing.worker_url != registration.worker_url:
            raise RegistrationConflict("Worker ID is already registered at another address")
        if any(
            worker.worker_url == registration.worker_url and worker.worker_id != registration.worker_id
            for worker in self._workers.values()
        ):
            raise RegistrationConflict("Worker address is already registered with another ID")
        if existing is not None and registration.instance_id in (None, existing.instance_id):
            if (registration.models != existing.models
                    or registration.demo_control_available != existing.demo_control_available):
                raise RegistrationConflict("Restart the worker to change its loaded models")
            # A repeat registration preserves the live process's sequence/identity.
            worker = existing.model_copy(update={"available": None, "heartbeat_expired": False})
        else:
            worker = RegisteredWorker(**registration.model_dump(), heartbeat_timeout_seconds=self._timeout)
        self._workers[worker.worker_id] = worker
        self._last_seen[worker.worker_id] = self._clock()
        return worker, existing is None

    def list_workers(self) -> list[RegisteredWorker]:
        self.expire_workers()
        return list(self._workers.values())

    def get_worker(self, worker_id: str) -> RegisteredWorker | None:
        self.expire_workers()
        return self._workers.get(worker_id)

    def node_statuses(self) -> list[NodeStatus]:
        self.expire_workers()
        now = self._clock()
        return [NodeStatus(worker=worker, lease_remaining_seconds=max(
            0.0, self._timeout - (now - self._last_seen[worker.worker_id])
        )) for worker in self._workers.values()]

    def record_prediction(self, worker: RegisteredWorker, model: str, request_id: str, trace_id: str) -> None:
        # A response from an old process must not increment the replacement's counter.
        current = self._workers.get(worker.worker_id)
        if current is None or current.instance_id != worker.instance_id or current.worker_url != worker.worker_url:
            return
        now = datetime.now(timezone.utc)
        current.completed_predictions += 1
        current.last_prediction_at = now
        self.recent_predictions.append(CompletedPrediction(
            worker_id=worker.worker_id, instance_id=worker.instance_id,
            model=model, request_id=request_id, trace_id=trace_id, completed_at=now,
        ))

    def heartbeat(self, payload: WorkerHeartbeat) -> RegisteredWorker:
        worker = self._workers.get(payload.worker_id)
        if worker is None:
            raise UnknownWorker("Worker is not registered")
        if worker.instance_id != payload.instance_id:
            raise HeartbeatConflict("Worker instance has changed; old heartbeats are rejected")
        if payload.sequence <= worker.heartbeat_sequence:
            raise HeartbeatConflict("Heartbeat sequence must increase")
        recovering = worker.heartbeat_expired or worker.available is False
        self._last_seen[worker.worker_id] = self._clock()
        worker.last_heartbeat_at = datetime.now(timezone.utc)
        worker.heartbeat_sequence = payload.sequence
        worker.heartbeat_expired = False
        worker.available = True
        if recovering:
            LOGGER.info("worker_heartbeat_recovered worker_id=%s", worker.worker_id)
        return worker

    def expire_workers(self) -> list[str]:
        now = self._clock()
        expired = []
        for worker_id, worker in self._workers.items():
            if now - self._last_seen[worker_id] >= self._timeout and not worker.heartbeat_expired:
                worker.heartbeat_expired = True
                worker.available = False
                expired.append(worker_id)
                LOGGER.warning("worker_heartbeat_expired worker_id=%s timeout_seconds=%s", worker_id, self._timeout)
        return expired

    def set_available(self, worker: RegisteredWorker, available: bool) -> None:
        self.expire_workers()
        # Do not overwrite a newer registration with an older in-flight probe.
        if self._workers.get(worker.worker_id) is worker:
            # Successful HTTP probes/predictions cannot extend a heartbeat lease.
            worker.available = available and not worker.heartbeat_expired

    def select(self, model: str = MODEL_NAME) -> RegisteredWorker | None:
        self.expire_workers()
        eligible = [worker for worker in self._workers.values()
                    if worker.available is not False and model in worker.models]
        if not eligible:
            return None
        cursor = self._next.get(model, 0)
        selected = eligible[cursor % len(eligible)]
        self._next[model] = (cursor + 1) % len(eligible)
        return selected
