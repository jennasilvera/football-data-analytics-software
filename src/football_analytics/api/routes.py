"""Native API v1 routes, independent of application middleware and lifecycle."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from football_analytics.config import Settings
from football_analytics.services.intelligence import (
    available_records,
    post_match_diagnostics,
    ratings,
    team_report,
)
from football_analytics.services.serving import ServingConflictError, predict_and_publish
from football_analytics.storage.repository import (
    PublicationConflictError,
    PublicationDeadlineError,
    Repository,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=100)
    match_id: str = Field(min_length=1, max_length=100)
    method: str = Field(default="ensemble", max_length=100)


def create_router(repo: Repository, settings: Settings) -> APIRouter:
    router = APIRouter()

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

    @router.get("/health")
    def health():
        if not repo.health():
            raise HTTPException(status_code=503, detail="Storage health check failed")
        return {"status": "ok", "storage_schema": 1}

    @router.get("/version")
    def version():
        return {
            "version": "0.3.0",
            "api_schema": 1,
            "scope": "official_senior_mens_a",
            "target": "regulation_time",
        }

    auth = [Depends(authorized)]

    def snapshot_boundary(as_of: datetime | None, through_sequence: int | None) -> int:
        if through_sequence is not None:
            if as_of is None:
                raise HTTPException(status_code=422, detail="Snapshot sequence requires as_of")
            if through_sequence > repo.snapshot_sequence():
                raise HTTPException(status_code=422, detail="Snapshot sequence is not committed")
            return through_sequence
        return repo.snapshot_sequence()

    def listing(
        kind: str, limit: int, after: str | None, as_of: datetime | None,
        through_sequence: int | None,
    ) -> dict[str, Any]:
        cutoff = as_of or datetime.now(UTC)
        boundary = snapshot_boundary(as_of, through_sequence)
        rows = repo.list(
            kind, limit=limit + 1, after_id=after, as_of=cutoff, through_sequence=boundary
        )
        return {
            "items": [r["payload"] for r in rows[:limit]],
            "limit": limit,
            "as_of": cutoff.isoformat(),
            "through_sequence": boundary,
            "next_cursor": rows[limit - 1]["entity_id"] if len(rows) > limit else None,
            "truncated": len(rows) > limit,
        }

    @router.get("/teams", dependencies=auth)
    def teams(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
        through_sequence: int | None = Query(None, ge=0, le=2**63 - 1),
    ):
        return listing("team", limit, after, as_of, through_sequence)

    @router.get("/teams/{team_id}", dependencies=auth)
    def team(team_id: str):
        return repo.get("team", team_id)["payload"]

    @router.get("/teams/{team_id}/ratings", dependencies=auth)
    def team_ratings(team_id: str):
        from football_analytics.ratings.glicko import rating_history

        repo.get("team", team_id)
        now = datetime.now(UTC)
        return [
            row
            for row in rating_history(available_records(repo, now), as_of=now)
            if row["team_id"] == team_id
        ]

    @router.get("/teams/{team_id}/report", dependencies=auth)
    @router.get("/reports/team/{team_id}", dependencies=auth)
    def team_intelligence(team_id: str):
        return team_report(repo, team_id, datetime.now(UTC))

    @router.get("/matches", dependencies=auth)
    def matches(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
        through_sequence: int | None = Query(None, ge=0, le=2**63 - 1),
    ):
        return listing("match", limit, after, as_of, through_sequence)

    @router.get("/matches/{match_id}", dependencies=auth)
    def match(match_id: str):
        return repo.get("match", match_id)["payload"]

    @router.get("/predictions", dependencies=auth)
    def predictions(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
        through_sequence: int | None = Query(None, ge=0, le=2**63 - 1),
    ):
        return listing("forecast", limit, after, as_of, through_sequence)

    @router.get("/forecast-batches", dependencies=auth)
    def forecast_batches(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
        through_sequence: int | None = Query(None, ge=0, le=2**63 - 1),
    ):
        return listing("forecast_batch", limit, after, as_of, through_sequence)

    @router.get("/forecast-batches/{run_id}", dependencies=auth)
    def forecast_batch_report(run_id: str):
        return repo.get("forecast_batch", run_id)["payload"]

    @router.post("/predict", dependencies=auth)
    def predict(body: PredictRequest):
        try:
            return predict_and_publish(
                repo, model_id=body.model_id, match_id=body.match_id,
                now=datetime.now(UTC), method=body.method,
                max_model_age_days=settings.max_model_age_days,
            )
        except (ServingConflictError, PublicationConflictError, PublicationDeadlineError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/ratings", dependencies=auth)
    def rating_list():
        return ratings(repo, datetime.now(UTC))

    @router.get("/ratings/movers", dependencies=auth)
    def movers():
        return sorted(ratings(repo, datetime.now(UTC)), key=lambda r: -abs(r["rating_change"]))

    @router.get("/models", dependencies=auth)
    def models(
        limit: int = Query(100, ge=1, le=1000),
        after: str | None = Query(None, max_length=200),
        as_of: datetime | None = None,
        through_sequence: int | None = Query(None, ge=0, le=2**63 - 1),
    ):
        cutoff = as_of or datetime.now(UTC)
        boundary = snapshot_boundary(as_of, through_sequence)
        rows = repo.list(
            "model", limit=limit + 1, after_id=after, as_of=cutoff, through_sequence=boundary
        )
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
            "through_sequence": boundary,
            "next_cursor": rows[limit - 1]["entity_id"] if len(rows) > limit else None,
        }

    @router.get("/models/{model_id}/metrics", dependencies=auth)
    def model_metrics(model_id: str):
        repo.get("model", model_id)
        return repo.get("model_metrics", model_id)["payload"]

    @router.get("/reports/match/{match_id}", dependencies=auth)
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

    @router.get("/diagnostics/upsets", dependencies=auth)
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

    @router.get("/diagnostics/regression-watchlist", dependencies=auth)
    def regression():
        return watchlist("positive_residual")

    @router.get("/diagnostics/breakout-watchlist", dependencies=auth)
    def breakout():
        return watchlist("negative_residual")

    return router
