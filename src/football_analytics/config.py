from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database: Path = Path("outputs/platform.sqlite3")
    api_key: str | None = None
    allow_unauthenticated: bool = False
    max_model_age_days: int = 90

    def __post_init__(self) -> None:
        if not self.allow_unauthenticated and (not self.api_key or len(self.api_key) < 24):
            raise ValueError(
                "Set FOOTBALL_API_KEY to at least 24 characters or "
                "explicitly enable local development mode."
            )
        if self.max_model_age_days <= 0:
            raise ValueError("Maximum model age must be positive.")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            Path(os.getenv("FOOTBALL_DATABASE", "outputs/platform.sqlite3")),
            os.getenv("FOOTBALL_API_KEY"),
            os.getenv("FOOTBALL_DEV_NO_AUTH") == "1",
            int(os.getenv("FOOTBALL_MAX_MODEL_AGE_DAYS", "90")),
        )
