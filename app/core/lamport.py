"""Process-local logical clocks and explicit HTTP send/receive events.

These counters order messages; heartbeat expiry still uses monotonic time.
"""

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

TIME_HEADER = "X-Lamport-Time"
RECEIVED_HEADER = "X-Lamport-Received"
INSTANCE_HEADER = "X-Lamport-Instance"
TRACE_HEADER = "X-Trace-ID"
LOGGER = logging.getLogger("inference")


def parse_time(value: str) -> int:
    # Bound header parsing and retain exact integers in the browser.
    if not re.fullmatch(r"0|[1-9][0-9]{0,14}", value):
        raise ValueError("Lamport timestamp must be a nonnegative decimal integer of at most 15 digits")
    return int(value)


class LamportClock:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.instance_id = str(uuid4())
        self._value = 0
        self._lock = Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def tick(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def receive(self, remote: int) -> int:
        if type(remote) is not int or remote < 0:
            raise ValueError("Received logical time must be a nonnegative integer")
        with self._lock:
            self._value = max(self._value, remote) + 1
            return self._value


class ClockEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    event: str
    node_id: str
    instance_id: str
    timestamp: int = Field(ge=1)


class InferenceTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    trace_id: str
    request_id: str
    status_code: int
    complete: bool
    events: tuple[ClockEvent, ...]


class TraceStore:
    """Last 200 completed requests, keyed by unique attempt rather than request ID."""
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._items: OrderedDict[str, InferenceTrace] = OrderedDict()
        self._lock = Lock()

    def add(self, trace: InferenceTrace) -> None:
        with self._lock:
            self._items[trace.trace_id] = trace
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, trace_id: str) -> InferenceTrace | None:
        with self._lock:
            return self._items.get(trace_id)


def event(clock: LamportClock, name: str, timestamp: int) -> ClockEvent:
    return ClockEvent(event=name, node_id=clock.node_id,
                      instance_id=clock.instance_id, timestamp=timestamp)


def log_event(clock: LamportClock, name: str, timestamp: int, request_id: str, path: str) -> None:
    LOGGER.info("lamport_event=%s node_id=%s instance_id=%s logical_time=%s request_id=%s path=%s",
                name, clock.node_id, clock.instance_id, timestamp, request_id, path)


@dataclass(frozen=True)
class Exchange:
    sent: int
    received: int
    peer_received: int | None
    peer_sent: int | None
    peer_instance: str | None


async def clocked_request(
    client: httpx.AsyncClient, clock: LamportClock, method: str, url: str, *,
    request_id: str, peer_id: str, trace_id: str | None = None,
    events: list[ClockEvent] | None = None, **kwargs,
) -> tuple[httpx.Response, Exchange]:
    """Advance only this process's clock; a missing legacy peer clock is not invented."""
    sent = clock.tick()
    headers = httpx.Headers(kwargs.pop("headers", {}))
    headers.update({"X-Request-ID": request_id, TIME_HEADER: str(sent), INSTANCE_HEADER: clock.instance_id})
    if trace_id is not None:
        headers[TRACE_HEADER] = trace_id
    if events is not None:
        events.append(event(clock, "request_forwarded", sent))
    log_event(clock, "message_sent", sent, request_id, url)
    response = await client.request(method, url, headers=headers, follow_redirects=False, **kwargs)
    present = [name in response.headers for name in (TIME_HEADER, RECEIVED_HEADER, INSTANCE_HEADER)]
    peer_received = peer_sent = None
    peer_instance = None
    try:
        if any(present):
            if not all(present) or any(len(response.headers.get_list(name)) != 1 for name in (TIME_HEADER, RECEIVED_HEADER, INSTANCE_HEADER)):
                raise ValueError("Incomplete or repeated peer clock headers")
            peer_sent = parse_time(response.headers[TIME_HEADER])
            peer_received = parse_time(response.headers[RECEIVED_HEADER])
            peer_instance = str(UUID(response.headers[INSTANCE_HEADER]))
            if not sent < peer_received < peer_sent:
                raise ValueError("Peer clock does not follow message causality")
            if trace_id is not None and response.headers.get(TRACE_HEADER) != trace_id:
                raise ValueError("Peer returned a mismatched trace ID")
    except ValueError:
        received = clock.tick()
        if events is not None:
            events.append(event(clock, "worker_response_received", received))
        log_event(clock, "invalid_message_received", received, request_id, url)
        raise
    received = clock.receive(peer_sent) if peer_sent is not None else clock.tick()
    if events is not None:
        if peer_received is not None:
            events.extend([
                ClockEvent(event="worker_request_received", node_id=peer_id, instance_id=peer_instance, timestamp=peer_received),
                ClockEvent(event="worker_response_sent", node_id=peer_id, instance_id=peer_instance, timestamp=peer_sent),
            ])
        events.append(event(clock, "worker_response_received", received))
    log_event(clock, "message_received", received, request_id, url)
    return response, Exchange(sent, received, peer_received, peer_sent, peer_instance)
