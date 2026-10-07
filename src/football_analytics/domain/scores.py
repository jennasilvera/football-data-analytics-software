"""Declared meaning of source goals; never infer periods from score values."""
from enum import StrEnum


class ScoreBasis(StrEnum):
    UNKNOWN = "unknown"
    REGULATION_TIME = "regulation_time"
    AFTER_EXTRA_TIME = "after_extra_time"
    INCLUDING_SHOOTOUT = "including_shootout"
    SHOOTOUT_ONLY = "shootout_only"


REGULATION_TARGET_POLICY_ID = "regulation_time_goals_and_1x2_v1"


def require_regulation_score(basis: ScoreBasis, *, match_id: str) -> None:
    if basis is not ScoreBasis.REGULATION_TIME:
        raise ValueError(
            f"Match {match_id}: regulation-time research requires score_basis="
            f"regulation_time; received {basis.value}. Supply documented regulation "
            "scores; extra-time and shootout scores cannot be converted automatically."
        )
