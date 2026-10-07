"""Declared feature families and typed exogenous observations for research."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from football_analytics.data import CanonicalCatalogs, CanonicalMatchRecord
from football_analytics.data.contracts import assert_pre_match_available
from football_analytics.data.serialization import metadata_from_dict
from football_analytics.domain import CompetitionStage
from football_analytics.domain.locations import GeographicPoint
from football_analytics.features import (
    CompetitionContextFeatureProvider,
    FifaRankingFeatureProvider,
    FifaRankingObservation,
    LegacyEloFeatureProvider,
    MarketSnapshotFeatureProvider,
    MarketSnapshotObservation,
    PlayerAvailabilityObservation,
    PlayerAvailabilityStatus,
    RollingFormFeatureProvider,
    ScheduleRestFeatureProvider,
    SquadAvailabilityFeatureProvider,
)
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureProvider,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)
from football_analytics.features.competition import MatchCompetitionContextObservation
from football_analytics.features.history import completed_record_is_before_cutoff
from football_analytics.features.travel import (
    TeamReferenceLocationObservation,
    TravelContextFeatureProvider,
    VenueLocationObservation,
)
from football_analytics.ratings.glicko import GlickoState, inflate, rating_history
from football_analytics.ratings.legacy_elo import LegacyEloRatingEngine
from football_analytics.ratings.replay import replay_completed_matches

MANUAL_FIELDS = (
    "manager_tenure_days",
    "squad_average_age",
    "top_five_league_players",
    "champions_league_players",
    "captain_available",
    "star_available",
    "goalkeeper_strength_proxy",
    "rotation_risk",
    "timezone_shift_hours",
    "altitude_m",
    "climate_stress_proxy",
    "must_win",
    "draw_utility",
    "elimination_risk",
)


class StrengthProvider:
    provider_id = "historical_strength_v1"

    def __init__(self, records: Sequence[CanonicalMatchRecord], catalogs: CanonicalCatalogs):
        self.records, self.catalogs = records, catalogs

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return (
            *LegacyEloFeatureProvider([]).definitions(),
            *tuple(
                FeatureDefinition(
                    f"glicko.{side}.{metric}",
                    "glicko1_rd50_v1",
                    "strength",
                    f"Glicko {metric} for {side} team",
                )
                for side in ("home", "away")
                for metric in ("rating", "deviation")
            ),
        )

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        records = [
            r for r in self.records if completed_record_is_before_cutoff(r, context.prediction_time)
        ]
        replay = replay_completed_matches(
            records,
            engine=LegacyEloRatingEngine(),
            competition_names={c.competition_id: c.name for c in self.catalogs.competitions},
        )
        values = list(LegacyEloFeatureProvider(replay.snapshots).compute(context))
        states = {r["team_id"]: r for r in rating_history(records, as_of=context.prediction_time)}
        lineage = FeatureLineage(
            source_record_ids=tuple(sorted(r.source_match_id for r in records)),
            model_ids=("glicko1_daily_rd50_per_month_v1",),
        )
        for side, team in (
            ("home", context.match.home_team_id),
            ("away", context.match.away_team_id),
        ):
            observed = states.get(team)
            state = (
                inflate(
                    GlickoState(observed["rating"], observed["deviation"]),
                    (
                        context.prediction_time.date()
                        - date.fromisoformat(observed["effective_date"])
                    ).days,
                )
                if observed
                else GlickoState()
            )
            for metric in ("rating", "deviation"):
                definition = next(
                    d for d in self.definitions() if d.name == f"glicko.{side}.{metric}"
                )
                values.append(
                    FeatureValue(
                        definition,
                        FeatureStatus.OBSERVED if observed else FeatureStatus.IMPUTED,
                        context.prediction_time,
                        getattr(state, metric),
                        missing_reason=None if observed else FeatureMissingReason.NO_HISTORY,
                        imputation_method=None if observed else "declared_glicko_prior",
                        lineage=lineage,
                    )
                )
        return tuple(values)


class ManualContextProvider:
    provider_id = "manual_context_v1"

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = []
        seen = set()
        for row in rows:
            if row["field"] not in MANUAL_FIELDS:
                raise ValueError("Unsupported manual feature field.")
            import math

            if not math.isfinite(float(row["value"])):
                raise ValueError("Manual feature value must be finite.")
            metadata = metadata_from_dict(row["metadata"])
            if not metadata.legal_use_notes or not metadata.source_record_id:
                raise ValueError("Manual intelligence requires legal notes and record identity.")
            key = (row["match_id"], row["team_id"], row["field"], metadata.available_at)
            if key in seen:
                raise ValueError("Ambiguous duplicate manual feature timestamp.")
            seen.add(key)
            value = float(row["value"])
            field = row["field"]
            if field in (
                "manager_tenure_days",
                "top_five_league_players",
                "champions_league_players",
            ):
                if value < 0 or not value.is_integer():
                    raise ValueError(
                        "Manual tenure and player counts must be nonnegative integers."
                    )
            bounds = {
                "squad_average_age": (15, 60),
                "timezone_shift_hours": (-24, 24),
                "altitude_m": (-500, 9000),
            }
            if field not in (
                "manager_tenure_days",
                "top_five_league_players",
                "champions_league_players",
            ):
                lower, upper = bounds.get(field, (0, 1))
                if not lower <= value <= upper:
                    raise ValueError("Manual feature value is outside declared bounds.")
            self.rows.append((row, metadata))

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return tuple(
            FeatureDefinition(
                f"manual.{side}.{name}",
                "v1",
                "manual_context",
                f"Declared {name}; missing rather than inferred",
            )
            for side in ("home", "away")
            for name in MANUAL_FIELDS
        )

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        values = []
        for side, team in (
            ("home", context.match.home_team_id),
            ("away", context.match.away_team_id),
        ):
            for name in MANUAL_FIELDS:
                eligible = []
                for row, metadata in self.rows:
                    if (row["match_id"], row["team_id"], row["field"]) != (
                        context.match.match_id,
                        team,
                        name,
                    ):
                        continue
                    try:
                        assert_pre_match_available(
                            metadata,
                            kickoff_at=context.match.kickoff_at or context.prediction_time,
                            prediction_time=context.prediction_time,
                        )
                    except ValueError:
                        continue
                    eligible.append((row, metadata))
                eligible.sort(key=lambda pair: (pair[1].available_at, pair[1].source_record_id))
                definition = next(
                    d for d in self.definitions() if d.name == f"manual.{side}.{name}"
                )
                if eligible:
                    row, metadata = eligible[-1]
                    values.append(
                        FeatureValue(
                            definition,
                            FeatureStatus.OBSERVED,
                            context.prediction_time,
                            float(row["value"]),
                            lineage=FeatureLineage(
                                source_record_ids=(str(metadata.source_record_id),)
                            ),
                        )
                    )
                else:
                    values.append(
                        FeatureValue(
                            definition,
                            FeatureStatus.MISSING,
                            context.prediction_time,
                            missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                        )
                    )
        return tuple(values)


def build_providers(
    records: Sequence[CanonicalMatchRecord],
    catalogs: CanonicalCatalogs,
    *,
    groups: tuple[str, ...] = ("form",),
    context: dict[str, Any] | None = None,
) -> list[FeatureProvider]:
    context = context or {}
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Feature groups must be non-empty and unique.")
    providers: list[FeatureProvider] = []

    def observations(key: str, cls, convert=None):
        result = []
        for row in context.get(key, []):
            body = {**row, "metadata": metadata_from_dict(row["metadata"])}
            if key == "market" and body.pop("score_basis", "unknown") != "regulation_time":
                raise ValueError("Market features require explicit regulation-time settlement.")
            if convert:
                body = convert(body)
            result.append(cls(**body))
        return result

    for group in groups:
        if group == "form":
            providers.append(ExtendedFormProvider(records))
        elif group == "schedule":
            providers.append(ScheduleRestFeatureProvider(records))
        elif group == "strength":
            providers.append(StrengthProvider(records, catalogs))
        elif group == "competition":
            providers.append(
                CompetitionContextFeatureProvider(
                    catalogs.competitions,
                    match_context=observations(
                        "competition",
                        MatchCompetitionContextObservation,
                        lambda b: {**b, "stage": CompetitionStage(b["stage"])},
                    ),
                )
            )
        elif group == "rankings":
            providers.append(
                FifaRankingFeatureProvider(observations("rankings", FifaRankingObservation))
            )
        elif group == "squad":
            providers.append(
                SquadAvailabilityFeatureProvider(
                    observations(
                        "squad",
                        PlayerAvailabilityObservation,
                        lambda b: {**b, "status": PlayerAvailabilityStatus(b["status"])},
                    )
                )
            )
        elif group == "market":
            providers.append(
                MarketSnapshotFeatureProvider(observations("market", MarketSnapshotObservation))
            )
        elif group == "travel":

            def converter(b):
                return {**b, "location": GeographicPoint(**b["location"])}

            providers.append(
                TravelContextFeatureProvider(
                    venue_locations=observations("venues", VenueLocationObservation, converter),
                    team_reference_locations=observations(
                        "team_locations", TeamReferenceLocationObservation, converter
                    ),
                )
            )
        elif group == "manual":
            providers.append(ManualContextProvider(context.get("manual", [])))
        else:
            raise ValueError(f"Unknown feature group: {group}")
    return providers


class ExtendedFormProvider(RollingFormFeatureProvider):
    """Clean-sheet and failed-to-score coverage alongside 5/10/20 match form."""

    def __init__(self, records):
        super().__init__(records, windows=(5, 10, 20), version="rolling_form_v2")
        self._extra_definitions = tuple(
            FeatureDefinition(
                f"form.{side}.{metric}_{window}",
                "rolling_form_v2",
                "recent_form",
                f"Fraction of last {window} completed matches with {metric}",
            )
            for side in ("home", "away")
            for window in (5, 10, 20)
            for metric in ("clean_sheet_rate", "failed_to_score_rate")
        )

    def definitions(self):
        return (*super().definitions(), *self._extra_definitions)

    def compute(self, context):
        from football_analytics.features.history import team_history_before_cutoff

        values = list(super().compute(context))
        for side, team in (
            ("home", context.match.home_team_id),
            ("away", context.match.away_team_id),
        ):
            history = team_history_before_cutoff(
                self._records, team_id=team, cutoff=context.prediction_time
            )
            for window in (5, 10, 20):
                recent = history[-window:]
                for metric in ("clean_sheet_rate", "failed_to_score_rate"):
                    definition = next(
                        d
                        for d in self._extra_definitions
                        if d.name == f"form.{side}.{metric}_{window}"
                    )
                    if not recent:
                        values.append(
                            FeatureValue(
                                definition,
                                FeatureStatus.MISSING,
                                context.prediction_time,
                                missing_reason=FeatureMissingReason.NO_HISTORY,
                            )
                        )
                        continue
                    goals = [
                        (
                            (r.away_score if r.match.home_team_id == team else r.home_score)
                            if metric == "clean_sheet_rate"
                            else (r.home_score if r.match.home_team_id == team else r.away_score)
                        )
                        for r in recent
                    ]
                    values.append(
                        FeatureValue(
                            definition,
                            FeatureStatus.OBSERVED,
                            context.prediction_time,
                            sum(g == 0 for g in goals) / len(goals),
                            lineage=FeatureLineage(
                                source_record_ids=tuple(r.source_match_id for r in recent)
                            ),
                        )
                    )
        return tuple(values)
