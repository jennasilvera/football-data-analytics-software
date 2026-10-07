from __future__ import annotations

import logging
import secrets
import time
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.config import Settings
from football_analytics.data.serialization import match_from_dict
from football_analytics.services.forecasting import forecast_match
from football_analytics.services.intelligence import (
    available_records,
    post_match_diagnostics,
    ratings,
    team_report,
)
from football_analytics.storage.repository import Repository

logger = logging.getLogger("football_analytics.api")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=100)
    match_id: str = Field(min_length=1, max_length=100)
    method: str = Field(default="ensemble", max_length=100)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    repo = Repository(settings.database)
    app = FastAPI(title="Football Data Analytics Software", version="0.3.0")
    app.state.repository = repo

    def authorized(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.allow_unauthenticated and settings.api_key is None:
            return
        expected = "Bearer " + (settings.api_key or "")
        if not authorization or not secrets.compare_digest(
            authorization.encode(), expected.encode()
        ):
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started, request_id = time.monotonic(), str(uuid4())
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 16384:
                    return JSONResponse(
                        status_code=413, content={"detail": "Request body too large"}
                    )
            request._body = bytes(body)
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

    @app.exception_handler(KeyError)
    async def not_found(request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "Record not found"})

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        if not repo.health():
            raise HTTPException(status_code=503, detail="Storage health check failed")
        return {"status": "ok", "storage_schema": 1}

    @app.get("/version")
    def version():
        return {
            "version": "0.3.0",
            "api_schema": 1,
            "scope": "official_senior_mens_a",
            "target": "regulation_time",
        }

    auth = [Depends(authorized)]

    def listing(kind: str, limit: int, after: str | None, as_of: datetime | None) -> dict[str, Any]:
        cutoff = as_of or datetime.now(UTC)
        rows = repo.list(kind, limit=limit + 1, after_id=after, as_of=cutoff)
        return {
            "items": [r["payload"] for r in rows[:limit]],
            "limit": limit,
            "as_of": cutoff.isoformat(),
            "next_cursor": rows[limit - 1]["entity_id"] if len(rows) > limit else None,
            "truncated": len(rows) > limit,
        }

    @app.get("/teams", dependencies=auth)
    def teams(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
    ):
        return listing("team", limit, after, as_of)

    @app.get("/teams/{team_id}", dependencies=auth)
    def team(team_id: str):
        return repo.get("team", team_id)["payload"]

    @app.get("/teams/{team_id}/ratings", dependencies=auth)
    def team_ratings(team_id: str):
        from football_analytics.ratings.glicko import rating_history

        repo.get("team", team_id)
        now = datetime.now(UTC)
        return [
            row
            for row in rating_history(available_records(repo, now), as_of=now)
            if row["team_id"] == team_id
        ]

    @app.get("/teams/{team_id}/report", dependencies=auth)
    @app.get("/reports/team/{team_id}", dependencies=auth)
    def team_intelligence(team_id: str):
        return team_report(repo, team_id, datetime.now(UTC))

    @app.get("/matches", dependencies=auth)
    def matches(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
    ):
        return listing("match", limit, after, as_of)

    @app.get("/matches/{match_id}", dependencies=auth)
    def match(match_id: str):
        return repo.get("match", match_id)["payload"]

    @app.get("/predictions", dependencies=auth)
    def predictions(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
    ):
        return listing("forecast", limit, after, as_of)

    @app.post("/predict", dependencies=auth)
    def predict(body: PredictRequest):
        now = datetime.now(UTC)
        model = repo.get("model", body.model_id)["payload"]
        approval = repo.get("model_release", body.model_id)["payload"]
        if approval.get("status") != "approved":
            raise HTTPException(status_code=409, detail="Model is not approved for serving")
        cutoff = datetime.fromisoformat(model["training_cutoff"])
        if (now - cutoff).total_seconds() / 86400 > settings.max_model_age_days:
            raise HTTPException(
                status_code=409, detail="Model history exceeds configured freshness limit"
            )
        source = repo.get("match", body.match_id)["payload"]
        forecast = forecast_match(
            model, match_from_dict(source["match"]), prediction_time=now, method=body.method
        )
        from football_analytics.services.market import attach_market

        forecast = attach_market(repo, forecast, now)
        repo.put("forecast", forecast["forecast_id"], forecast, available_at=now, recorded_at=now)
        return forecast

    @app.get("/ratings", dependencies=auth)
    def rating_list():
        return ratings(repo, datetime.now(UTC))

    @app.get("/ratings/movers", dependencies=auth)
    def movers():
        return sorted(ratings(repo, datetime.now(UTC)), key=lambda r: -abs(r["rating_change"]))

    @app.get("/models", dependencies=auth)
    def models(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
    ):
        cutoff = as_of or datetime.now(UTC)
        rows = repo.list("model", limit=limit + 1, after_id=after, as_of=cutoff)
        return {
            "items": [
                {
                    "model_id": r["entity_id"],
                    "training_cutoff": r["payload"]["training_cutoff"],
                    "calibrated": bool(r["payload"]["transforms"]),
                }
                for r in rows[:limit]
            ],
            "truncated": len(rows) > limit,
            "limit": limit,
            "as_of": cutoff.isoformat(),
            "next_cursor": rows[limit - 1]["entity_id"] if len(rows) > limit else None,
        }

    @app.get("/models/{model_id}/metrics", dependencies=auth)
    def model_metrics(model_id: str):
        repo.get("model", model_id)
        return repo.get("model_metrics", model_id)["payload"]

    @app.get("/reports/match/{match_id}", dependencies=auth)
    def match_report(match_id: str):
        source = repo.get("match", match_id)["payload"]
        forecasts = [
            row["payload"]
            for row in repo.scan("forecast")
            if row["payload"]["match"]["match_id"] == match_id
        ]
        diagnostics = [
            row
            for row in post_match_diagnostics(repo, datetime.now(UTC))
            if row["match_id"] == match_id
        ]
        return {"match": source, "forecasts": forecasts, "post_match": diagnostics}

    @app.get("/diagnostics/upsets", dependencies=auth)
    def upsets():
        return post_match_diagnostics(repo, datetime.now(UTC))

    def watchlist(signal: str):
        now = datetime.now(UTC)
        reports = [team_report(repo, row["entity_id"], now) for row in repo.scan("team", as_of=now)]
        return {
            "method": "descriptive_residual_watch_v1",
            "validated_predictive_signal": False,
            "teams": [r for r in reports if r["watch_signal"] == signal],
        }

    @app.get("/diagnostics/regression-watchlist", dependencies=auth)
    def regression():
        return watchlist("positive_residual")

    @app.get("/diagnostics/breakout-watchlist", dependencies=auth)
    def breakout():
        return watchlist("negative_residual")

    return app
