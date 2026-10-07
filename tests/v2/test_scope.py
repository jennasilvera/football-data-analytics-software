from __future__ import annotations

from football_analytics.data import (
    GenderCategory,
    ScopeDecision,
    TeamLevel,
    assess_senior_mens_a_scope,
)


def test_senior_mens_a_match_is_in_scope() -> None:
    assessment = assess_senior_mens_a_scope(
        gender=GenderCategory.MEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
    )

    assert assessment.decision is ScopeDecision.INCLUDE


def test_womens_match_is_excluded() -> None:
    assessment = assess_senior_mens_a_scope(
        gender=GenderCategory.WOMEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
    )

    assert assessment.decision is ScopeDecision.EXCLUDE
    assert assessment.reason == "gender_out_of_scope:women"


def test_u23_match_is_excluded() -> None:
    assessment = assess_senior_mens_a_scope(
        gender=GenderCategory.MEN,
        team_level=TeamLevel.OLYMPIC_U23,
        official=True,
    )

    assert assessment.decision is ScopeDecision.EXCLUDE
    assert assessment.reason == "team_level_out_of_scope:olympic_u23"


def test_unknown_scope_metadata_requires_review() -> None:
    assessment = assess_senior_mens_a_scope(
        gender=GenderCategory.UNKNOWN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
    )

    assert assessment.decision is ScopeDecision.REVIEW
