"""In-memory pub/sub of job progress events for SSE (spec Section 9.1)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

TERMINAL = {"completed", "failed"}


class EventBus:
    """Keeps the latest event per job and fans new events out to subscribers."""

    def __init__(self) -> None:
        """Create an empty bus."""
        self._latest: dict[str, dict[str, Any]] = {}
        self._subs: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    def latest(self, job_id: str) -> dict[str, Any] | None:
        """Most recent event for a job, if any was published in this process."""
        return self._latest.get(job_id)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Record and broadcast an event."""
        self._latest[job_id] = event
        for q in list(self._subs.get(job_id, ())):
            q.put_nowait(event)

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Start receiving events for a job."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subs[job_id].add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Stop receiving events for a job."""
        self._subs.get(job_id, set()).discard(q)

    def forget(self, job_id: str) -> None:
        """Drop all state for a deleted job."""
        self._latest.pop(job_id, None)
        self._subs.pop(job_id, None)
