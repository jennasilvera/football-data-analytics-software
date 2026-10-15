from __future__ import annotations

import logging
import sqlite3
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from football_analytics.api.routes import create_router
from football_analytics.config import Settings
from football_analytics.storage.repository import (
    Repository,
    RepositoryIntegrityError,
)

logger = logging.getLogger("football_analytics.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    repo = Repository(settings.database)
    app = FastAPI(title="Football Data Analytics Software", version="0.3.0")
    app.state.repository = repo

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started, request_id = time.monotonic(), str(uuid4())
        request.state.request_id = request_id
        response = None
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 16384:
                    response = JSONResponse(
                        status_code=413, content={"detail": "Request body too large"}
                    )
                    break
                body.extend(chunk)
            if response is None:
                request._body = bytes(body)
        if response is None:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": (time.monotonic() - started) * 1000,
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def request_invalid(request: Request, exc: RequestValidationError):
        # Raw inputs/context can contain secrets, non-finite numbers, or exceptions.
        # Keep only actionable, JSON-safe validation metadata.
        errors = [
            {key: error[key] for key in ("loc", "msg", "type")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(KeyError)
    async def not_found(request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "Record not found"})

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(RepositoryIntegrityError)
    async def storage_unavailable(
        request: Request, exc: sqlite3.Error | RepositoryIntegrityError
    ):
        logger.error(
            "storage_unavailable",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(exc).__name__,
                "sqlite_errorcode": getattr(exc, "sqlite_errorcode", None),
            },
        )
        return JSONResponse(
            status_code=503, content={"detail": "Storage unavailable; operation not confirmed"}
        )

    router = create_router(repo, settings)
    app.include_router(router, prefix="/api/v1")
    # Compatibility aliases share handlers, authentication and storage.
    app.include_router(router, include_in_schema=False)
    return app
