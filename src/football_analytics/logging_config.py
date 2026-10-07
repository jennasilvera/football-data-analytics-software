"""Structured logs omit credentials and request bodies."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = {
            key: getattr(record, key)
            for key in ("request_id", "method", "path", "status", "duration_ms")
            if hasattr(record, key)
        }
        return json.dumps(
            {
                "time": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                **fields,
            }
        )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.getLogger("football_analytics").handlers = [handler]
    logging.getLogger("football_analytics").setLevel(logging.INFO)
