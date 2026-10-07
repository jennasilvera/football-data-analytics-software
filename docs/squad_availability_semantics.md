# Squad Availability Feature Semantics

## Purpose

The V2 squad-availability layer converts source-provided player availability
assessments into auditable team-level features for a specific target match.

The design deliberately removes the legacy assumption that descriptive status
labels should be converted into hard-coded numeric weights.

## Source observation

Each `PlayerAvailabilityObservation` is tied to:

- canonical match ID;
- canonical team ID;
- stable player ID;
- descriptive status;
- explicit availability probability in [0, 1];
- optional baseline expected minutes;
- source/provenance metadata.

Multiple observations for one player are allowed over time. The latest safe
observation available at the prediction cutoff is used.

## Status is descriptive

The V2 provider does not map:

- probable -> 0.85;
- questionable -> 0.50;
- doubtful -> 0.25; or
- unknown -> 0.50.

Those legacy mappings are modeling assumptions, not source facts.

If a source or separate model wants to assign probabilities from statuses, that
transformation should be versioned and evaluated independently before its
outputs enter `PlayerAvailabilityObservation`.

## Features

For each team the provider emits:

- player observation count;
- mean availability probability;
- expected unavailable player count;
- projected-minutes coverage ratio;
- projected-minutes-weighted availability.

It also emits home-minus-away differences for mean availability and
minutes-weighted availability.

## Projected minutes

`baseline_expected_minutes` is optional.

When only some observed players have a minutes projection,
`minutes_coverage_ratio` exposes that partial coverage.

When no positive projected-minute denominator exists,
`minutes_weighted_availability` remains missing rather than being filled with a
default value.

## Point-in-time policy

An observation is eligible only when:

- `available_at` is known;
- `available_at <= prediction_time`; and
- `leakage_risk == safe`.

A later injury update, lineup announcement, suspension confirmation, or recovery
report cannot affect a historical forecast made before that information was
available.

## Research policy

Availability features should be evaluated with chronological ablation and
calibration analysis. They should not directly alter forecast probabilities
through an unvalidated adjustment rule.

A later player-strength layer may add source-defined or model-derived player
value, but its model/version must be explicit rather than stored as an
unqualified "importance rating."
