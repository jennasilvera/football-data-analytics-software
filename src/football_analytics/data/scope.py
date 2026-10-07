from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GenderCategory(StrEnum):
    """Gender category attached to a source match observation."""

    MEN = "men"
    WOMEN = "women"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TeamLevel(StrEnum):
    """Team level attached to a source match observation."""

    SENIOR_A = "senior_a"
    OLYMPIC_U23 = "olympic_u23"
    YOUTH = "youth"
    B_TEAM = "b_team"
    SELECT = "select"
    CLUB = "club"
    ACADEMY = "academy"
    UNKNOWN = "unknown"


class ScopeDecision(StrEnum):
    """Whether a source observation belongs in the platform's formal scope."""

    INCLUDE = "include"
    EXCLUDE = "exclude"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ScopeAssessment:
    """Result of evaluating a source observation against formal product scope."""

    decision: ScopeDecision
    reason: str


def assess_senior_mens_a_scope(
    *,
    gender: GenderCategory,
    team_level: TeamLevel,
    official: bool | None,
) -> ScopeAssessment:
    """Evaluate senior men's A-international eligibility.

    Unknown metadata is reviewed rather than guessed. Explicitly out-of-scope
    categories are excluded.
    """

    if gender is GenderCategory.UNKNOWN:
        return ScopeAssessment(
            decision=ScopeDecision.REVIEW,
            reason="gender_unknown",
        )

    if team_level is TeamLevel.UNKNOWN:
        return ScopeAssessment(
            decision=ScopeDecision.REVIEW,
            reason="team_level_unknown",
        )

    if official is None:
        return ScopeAssessment(
            decision=ScopeDecision.REVIEW,
            reason="official_status_unknown",
        )

    if gender is not GenderCategory.MEN:
        return ScopeAssessment(
            decision=ScopeDecision.EXCLUDE,
            reason=f"gender_out_of_scope:{gender.value}",
        )

    if team_level is not TeamLevel.SENIOR_A:
        return ScopeAssessment(
            decision=ScopeDecision.EXCLUDE,
            reason=f"team_level_out_of_scope:{team_level.value}",
        )

    if not official:
        return ScopeAssessment(
            decision=ScopeDecision.EXCLUDE,
            reason="unofficial_match",
        )

    return ScopeAssessment(
        decision=ScopeDecision.INCLUDE,
        reason="senior_mens_a_international",
    )
