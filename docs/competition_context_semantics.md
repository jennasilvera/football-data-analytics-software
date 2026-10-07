# Competition Context Feature Semantics

## Purpose

The V2 competition-context layer encodes competition identity and match stage
without parsing tournament names or assigning subjective importance weights.

## Competition kind

Competition kind is taken from the canonical `Competition.kind` enum.

The provider emits one-hot features for every supported `CompetitionKind`.
A competition called "World Cup Something" does not become a World Cup feature
unless its canonical kind is explicitly `world_cup`.

This removes the legacy dependence on string matching such as:

- tournament name contains "World Cup";
- tournament name contains "friendly"; or
- tournament name contains a confederation championship token.

Those heuristics remain part of legacy Elo parity only and should not define V2
feature semantics.

## Confederation scope

`competition.confederation_specific` indicates whether the canonical
competition is associated with one confederation.

It does not infer team membership or historical confederation affiliation.

## Match stage

Stage information is supplied through
`MatchCompetitionContextObservation` with the same point-in-time metadata used
elsewhere in V2.

Supported generic stages are:

- group or league;
- knockout;
- playoff;
- final; and
- other.

These categories are intentionally generic enough to support multiple senior
international competition formats without encoding tournament-specific labels
into model columns.

## Point-in-time eligibility

Stage observations are usable only when:

- `available_at` is known;
- `available_at <= prediction_time`; and
- `leakage_risk == safe`.

If only future or unsafe stage information exists, stage features remain missing
with a temporal-integrity reason.

## Leg number

A positive leg number is emitted only when the source explicitly provides one.

If a valid stage observation exists but no leg number applies, the feature is
missing with `not_applicable`; the system does not invent `1` for one-off
matches.

## Research policy

This layer intentionally does not contain a competition "importance score."
Any importance weighting should be evaluated as a separate modeling hypothesis
through chronological backtesting and ablation rather than embedded as an
unexamined data-contract assumption.
