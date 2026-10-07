# Native independent Poisson model card

## Status and intended use

Model family: `poisson`. Specification: `independent_poisson_v1`.
Implementation version: `legacy_parity_v1`. Artifact schema: 2.

This is a migrated research baseline for senior men's national-team football.
It estimates goal-count rates, scoreline probabilities, and three-way outcome
probabilities from canonical scored matches. It is not a calibrated production
forecast, a wagering recommendation, or an event-data expected-goals model.
There is no empirical claim of predictive skill from the bundled synthetic data.

## Estimation

For the training matches, let μ be total goals divided by total team appearances.
For team i, attack strength is mean goals scored / μ and defence weakness is mean
goals conceded / μ. Both are fitted solely from the declared training sample.

For home team h and away team a:

```text
λ_home = μ × attack_h × defence_weakness_a × venue_multiplier
λ_away = μ × attack_a × defence_weakness_h / venue_multiplier
```

The venue multiplier is 1 at neutral venues and 1.10 otherwise. The default
expected-goal rates are clipped to [0.20, 5.00]. These parameters intentionally
preserve the legacy algorithm; they are not estimated optimal values.

Home and away goals are independent Poisson variables. The reported grid is
0…10 goals per side by default. Grid probabilities and outcome probabilities are
normalized **conditional on both scores lying inside that grid**. Every forecast
reports `omitted_tail_probability` from the unnormalized joint distribution, so
truncation is visible. Increasing the grid limit reduces this omitted mass.
Expected goals are the original Poisson rates, not means of the truncated grid.

The modal score uses ascending home/away goals to break exact ties. Outcome
entropy is measured in nats. Entropy measures concentration, not calibrated
confidence or parameter uncertainty. This version supplies no uncertainty interval
or confidence category because neither has yet been validated.

## Contracts and temporal integrity

- Inputs are resolved canonical teams and scored completed matches.
- Duplicate/reversed fixtures and invalid goal targets are rejected.
- Every training result must be eligible by the declared training cutoff.
- Date-only sources use the existing conservative next-UTC-day policy.
- Forecast time must be at or after model training cutoff and no later than
  kickoff; date-only forecasts must precede the match date.
- Unseen teams cause an explicit error; there is no league-average fallback.
- Zero-goal training datasets fail explicitly rather than inventing a goal rate.
- Cancelled/postponed fixtures cannot be forecast through this model.

The model is fitted once per temporal fold. It does not update attack/defence
within that fold. Rolling-form classifiers can use newly eligible historical
features during the evaluation window; model fitting remains fixed for all
families. This difference in forecast information is intentional and must be
considered when interpreting comparisons.

## Reproducibility and artifacts

Goal-dataset identity includes exact goal counts (not only win/draw/loss), team
IDs, competition, venue neutrality, match time, target availability, and source
record/version. Model identity includes dataset identity, training cutoff,
configuration, global rate and every fitted team strength/appearance count.

`models/poisson_model_<sha256>.json` stores complete portable inference state.
Loading reconstructs the typed model and verifies its content-derived identity.
No pickle or executable object is deserialized. This detects accidental edits;
it is not a signed authenticity guarantee. Identical writes are idempotent and
differing existing bytes are never overwritten.

Changing future results cannot change earlier fitted rates or forecasts. An
upstream file/version digest may nevertheless change provenance identities.
Runtime versions are captured by the surrounding research report, not included
in the model's portable-state identity.

## Validation

Regression tests compare native expected goals, every grid cell, and three-way
probabilities with the legacy implementation for neutral and home venues across
several grid sizes. Additional tests cover leakage, unseen teams, invalid goals,
reversed duplicates, finite rates, truncation mass, JSON round trips, tampering,
and paired chronological comparison. `make demo-v2-comparison` runs the full
four-model workflow in CI.

## Limitations and next research

The estimator has no opponent-adjusted likelihood fitting, shrinkage, recency
weights, squad input, low-score dependence correction, or posterior uncertainty.
Sparse and imbalanced national-team schedules can distort raw attack/defence
ratios. Appearances are retained to expose sparse fitted histories.

The target is **regulation time including stoppage time**, excluding extra time
and shootouts. Native research rejects unknown or non-regulation score bases;
adapters preserve the declared basis for audit. Source assertions still require
independent verification; a label does not prove provider accuracy. See
[score target contract](score_target_contract.md). Next-day availability is a research assumption, not evidence
of the source's publication history or absence of retrospective corrections.

Future work should assess shrinkage and time decay, source revision
contracts, low-score dependence, nested temporal calibration and ensemble
training, interval coverage, and licensed-source evaluation. A more complex
model must beat the simple training-only class-frequency benchmark on held-out
data before any promotion decision.
