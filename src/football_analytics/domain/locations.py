from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeographicPoint:
    """A validated WGS84-style latitude/longitude point in decimal degrees."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        latitude = float(self.latitude)
        longitude = float(self.longitude)

        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("Geographic coordinates must be finite.")
        if latitude < -90.0 or latitude > 90.0:
            raise ValueError("latitude must be between -90 and 90 degrees.")
        if longitude < -180.0 or longitude > 180.0:
            raise ValueError("longitude must be between -180 and 180 degrees.")

        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)


@dataclass(frozen=True, slots=True)
class Venue:
    """Canonical football venue identity.

    Coordinates are optional because venue identity and geospatial enrichment
    can arrive from different governed sources.
    """

    venue_id: str
    name: str
    location: GeographicPoint | None = None
    country_code: str | None = None

    def __post_init__(self) -> None:
        venue_id = self.venue_id.strip()
        name = self.name.strip()

        if not venue_id:
            raise ValueError("venue_id must not be blank.")
        if not name:
            raise ValueError("name must not be blank.")

        object.__setattr__(self, "venue_id", venue_id)
        object.__setattr__(self, "name", name)

        if self.country_code is not None:
            country_code = self.country_code.strip().upper()
            if len(country_code) != 2 or not country_code.isalpha():
                raise ValueError(
                    "country_code must be a two-letter alphabetic code."
                )
            object.__setattr__(self, "country_code", country_code)
