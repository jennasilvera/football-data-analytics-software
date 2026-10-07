from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_immutable(path: Path, data: bytes) -> Path:
    """Publish complete bytes atomically; refuse to overwrite differing content."""
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
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
