"""Job folders and atomic JSON I/O (spec Section 8.1)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def is_valid_job_id(job_id: str) -> bool:
    """True for a uuid4 hex id; rejects anything that could escape the jobs folder."""
    return bool(JOB_ID_RE.fullmatch(job_id))


def job_dir(jobs_root: Path, job_id: str) -> Path:
    """Folder for one job. Raises ValueError for malformed ids."""
    if not is_valid_job_id(job_id):
        raise ValueError(f"invalid job id: {job_id!r}")
    return jobs_root / job_id


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: write a temp file, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """Read JSON, or None if the file is missing or not valid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def slugify(title: str, limit: int = 60) -> str:
    """Safe filename stem: lowercase, non-alphanumerics to '-', at most `limit` chars."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:limit].strip("-") or "meeting"
