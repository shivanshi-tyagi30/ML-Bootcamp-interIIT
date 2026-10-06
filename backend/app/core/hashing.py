"""File hashing for the dedupe cache (spec Section 8.4)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
