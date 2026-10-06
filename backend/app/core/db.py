"""SQLite job index (spec Section 8.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id            TEXT PRIMARY KEY,
  title         TEXT NOT NULL,
  source_file   TEXT NOT NULL,
  file_sha256   TEXT NOT NULL,
  status        TEXT NOT NULL,
  stage         TEXT NOT NULL,
  error_code    TEXT,
  error_message TEXT,
  error_detail  TEXT,
  duration_s    REAL,
  n_decisions   INTEGER DEFAULT 0,
  n_tasks       INTEGER DEFAULT 0,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_sha ON jobs(file_sha256);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""

COLUMNS = (
    "id", "title", "source_file", "file_sha256", "status", "stage", "error_code", "error_message",
    "error_detail", "duration_s", "n_decisions", "n_tasks", "created_at", "updated_at",
)
UPDATABLE = set(COLUMNS) - {"id", "created_at"}


def now_iso() -> str:
    """Current UTC time in ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobDB:
    """Small async wrapper around the jobs table."""

    def __init__(self, path: Path) -> None:
        """Use the SQLite file at `path`."""
        self.path = path

    async def _conn(self) -> aiosqlite.Connection:
        """Open a connection with dict-like rows."""
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def init(self) -> None:
        """Create the table and indexes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as conn:
            await conn.executescript(SCHEMA)
            cur = await conn.execute("PRAGMA table_info(jobs)")
            if "error_detail" not in {r[1] for r in await cur.fetchall()}:  # databases from v1.0
                await conn.execute("ALTER TABLE jobs ADD COLUMN error_detail TEXT")
            await conn.commit()

    async def insert(self, row: dict[str, Any]) -> None:
        """Insert a new job row; timestamps are filled in."""
        ts = now_iso()
        data = {c: row.get(c) for c in COLUMNS}
        data["created_at"] = data["created_at"] or ts
        data["updated_at"] = ts
        data["n_decisions"] = data["n_decisions"] or 0
        data["n_tasks"] = data["n_tasks"] or 0
        cols = ", ".join(COLUMNS)
        marks = ", ".join("?" for _ in COLUMNS)
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", [data[c] for c in COLUMNS])
            await conn.commit()

    async def update(self, job_id: str, **fields: Any) -> None:
        """Update some columns of a job."""
        fields = {k: v for k, v in fields.items() if k in UPDATABLE}
        if not fields:
            return
        fields["updated_at"] = now_iso()
        sets = ", ".join(f"{k} = ?" for k in fields)
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*fields.values(), job_id])
            await conn.commit()

    async def get(self, job_id: str) -> dict[str, Any] | None:
        """One job row, or None."""
        conn = await self._conn()
        try:
            cur = await conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await conn.close()

    async def list(self, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Jobs newest first, and the total count."""
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?", (limit, offset)
            )
            rows = [dict(r) for r in await cur.fetchall()]
            cur = await conn.execute("SELECT COUNT(*) FROM jobs")
            total = (await cur.fetchone())[0]
            return rows, total
        finally:
            await conn.close()

    async def find_completed_by_sha(self, sha: str) -> dict[str, Any] | None:
        """Newest completed job for a file hash (dedupe cache)."""
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT * FROM jobs WHERE file_sha256 = ? AND status = 'completed' ORDER BY created_at DESC LIMIT 1",
                (sha,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await conn.close()

    async def delete(self, job_id: str) -> None:
        """Remove a job row."""
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await conn.commit()

    async def mark_interrupted(self, code: str, message: str) -> list[str]:
        """Fail every queued/running job (used at startup); returns their ids."""
        async with aiosqlite.connect(self.path) as conn:
            cur = await conn.execute("SELECT id FROM jobs WHERE status IN ('queued', 'running')")
            ids = [r[0] for r in await cur.fetchall()]
            await conn.execute(
                "UPDATE jobs SET status='failed', stage='failed', error_code=?, error_message=?, "
                "error_detail='The server restarted while this meeting was processing. Use Retry to continue.', "
                "updated_at=? WHERE status IN ('queued', 'running')",
                (code, message, now_iso()),
            )
            await conn.commit()
        return ids
