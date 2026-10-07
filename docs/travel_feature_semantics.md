# Travel and Venue Feature Semantics

## Purpose

The V2 travel-context layer provides point-in-time geographic context without
claiming to know information that the source data does not contain.

The first implementation deliberately models **reference distance**, not actual
team travel.

## What the feature means

`travel.home.reference_distance_km` and
`travel.away.reference_distance_km` are great-circle distances between:

1. a governed geographic reference point for the national team; and
2. a governed geographic observation for the target match venue.

The distance is calculated with the haversine formula and the IUGG mean Earth
radius.

## What the feature does not mean

It is not:

- flight distance;
- actual route distance;
- distance from the team's current training camp;
- distance from the squad's departure airport;
- accumulated travel during the international window; or
- jet-lag exposure.

Those concepts require itinerary or camp-location observations with their own
source, availability timestamp, provenance, and legal-use rules.

## Point-in-time policy

A location observation is eligible only when:

- `available_at` is known;
- `available_at <= prediction_time`; and
- `leakage_risk == safe`.

A future, review-only, high-risk, or unknown-availability observation is not
silently substituted.

If the match has no canonical `venue_id`, the provider returns a missing
travel-distance feature rather than inferring a venue from competition,
country, team name, or home/away designation.

## Neutral-site context

`context.match.neutral_site` comes directly from the canonical Match contract.
It is not inferred from distance or geography.

## Research use

Reference distance should be treated as a contextual baseline. Before it is
promoted into a production forecasting model, its incremental value should be
tested through chronological ablation and calibration analysis. Actual travel
and timezone-displacement features should be added only when defensible
point-in-time itinerary data is available.
