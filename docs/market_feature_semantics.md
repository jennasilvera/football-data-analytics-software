# Market Feature Semantics

## Purpose

The V2 market layer treats prices as a research benchmark and contextual signal,
not as an instruction to bet.

The core feature layer contains no staking, Kelly sizing, edge threshold, or
automatic decision policy.

## Snapshot observation

Each `MarketSnapshotObservation` contains:

- canonical match ID;
- market source ID;
- home/draw/away decimal odds;
- full source/provenance metadata.

A snapshot is eligible only when it was safely available by the prediction
cutoff.

## De-vigging

Raw implied probabilities are:

```text
p_raw = 1 / decimal_odds
```

For a three-way market, their sum commonly exceeds 1 because of bookmaker
margin.

V2 normalizes the three raw implied probabilities by their total before using
them as outcome-belief features.

The original excess is preserved separately as
`market.consensus.average_overround`.

This prevents bookmaker margin from being silently interpreted as outcome
probability.

## Multiple sources

For each market source, the provider chooses:

- the earliest cutoff-eligible snapshot as that source's opening observation;
- the latest cutoff-eligible snapshot as its current observation.

The consensus is the arithmetic mean of the de-vigged latest probabilities
across sources.

The source count is emitted explicitly so downstream research can distinguish a
one-source signal from a broader consensus.

## Movement

Opening-to-current movement is measured in fair probability space, not raw odds
space.

For each outcome:

```text
movement = current_consensus_fair_probability
           - opening_consensus_fair_probability
```

With one snapshot per source, movement is zero rather than inferred from data
that does not exist.

## Freshness

`market.consensus.oldest_source_snapshot_age_hours` reports the age of the
stalest latest source snapshot included in the consensus.

Using the oldest source is deliberately conservative; a single fresh source
does not hide stale contributors.

## Point-in-time policy

A snapshot is usable only when:

- `available_at` is known;
- `available_at <= prediction_time`; and
- `leakage_risk == safe`.

Closing prices observed after the forecast cutoff cannot enter pre-match
features for that historical forecast.

## Research policy

Market features should be evaluated through chronological ablation,
calibration, and market-vs-model comparison.

The core platform should remain capable of generating forecasts without market
data so market comparison can remain a meaningful independent benchmark.
