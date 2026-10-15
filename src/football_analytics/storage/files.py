from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_immutable(path: Path, data: bytes) -> Path:
    """Publish synced bytes and directory entry; refuse differing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Artifact content conflict: {path}") from None
        # Sync even on identical retries: a previous attempt may have linked
        # complete bytes but failed before persisting the directory entry.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
