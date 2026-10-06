"""Shared API state and dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.core.db import JobDB
from app.core.errors import ERROR_TABLE, user_message
from app.core.events import EventBus
from app.core.storage import is_valid_job_id
from app.pipeline.runner import Services

Runner = Callable[[str], Awaitable[None]]


@dataclass
class AppState:
    """Everything the routes share."""

    settings: Settings
    db: JobDB
    bus: EventBus
    services: Services
    runner: Runner
    active_uploads: int = 0
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def schedule(self, job_id: str) -> None:
        """Start a job in the background, keeping a reference so it isn't garbage-collected."""
        task = asyncio.create_task(self.runner(job_id))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


def get_state(request: Request) -> AppState:
    """The app's shared state."""
    return request.app.state.trace


def error_response(code: str) -> JSONResponse:
    """`{"code", "message"}` with the HTTP status from the error table."""
    return JSONResponse(status_code=ERROR_TABLE[code]["http"], content={"code": code, "message": user_message(code)})


async def get_job_or_none(state: AppState, job_id: str) -> dict[str, Any] | None:
    """The job row, or None for unknown or malformed ids."""
    if not is_valid_job_id(job_id):
        return None
    return await state.db.get(job_id)
