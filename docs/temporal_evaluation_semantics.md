# Temporal Evaluation Semantics

## Purpose

V2 forecasting uses two distinct temporal questions that must never be collapsed into one timestamp:

1. **When could the forecast features have been known?**
2. **When could the realized match outcome have been used as a training label?**

A historical row can be safe for feature construction and still unsafe for model training if its final result was not yet eligible at the training cutoff.

## The two clocks

Every historical supervised example carries `prediction_time`, `target_available_at`, and `target_availability_basis`.

For a training fold with cutoff `t`, an example is eligible only when:

```text
prediction_time < t
target_available_at <= t
```

The first condition prevents future feature state from entering the fold. The second prevents an unresolved future outcome from becoming a historical label.

## Completed-result eligibility

The default policy is versioned as `completed_result_eligibility_v1`.

### Exact kickoff + defensible source result timestamp

When an exact kickoff is known and the source provides a valid `available_at` timestamp for the completed result, that source timestamp is used. It must be strictly after kickoff.

### Missing result timestamp

When a defensible result-availability timestamp is unavailable, V2 does not invent a final-whistle time or assume a fixed match duration. The result becomes eligible at the start of the next UTC calendar day.

This is deliberately conservative. It is an **eligibility boundary**, not a claim that the source actually published the result at midnight.

### Date-only historical matches

A date-only match cannot establish within-day chronology. The default policy therefore retains the conservative next-UTC-day boundary unless an explicit source availability timestamp is later still.

## Feature-history invariant

Rolling form and other completed-match history use the same result-eligibility policy as supervised model training. A final score cannot enter a later feature vector merely because the earlier match had already kicked off.

```text
match started
match completed
final result demonstrably eligible
```

These are not treated as equivalent events.

## Backtest invariant

Expanding- and rolling-window fold builders first identify examples whose prediction timestamps belong to the historical training window. They then filter those candidates by target availability.

If the declared minimum training size would have been met except that some targets were not yet available, the fold is skipped with `insufficient_available_targets`, and the audit record includes `unavailable_target_count`.

## Reproducibility identity

Target availability is part of the research artifact identity. The content-addressed model dataset hash includes the feature-set ID, prediction-cutoff policy ID, result-eligibility policy ID, imputation policy, canonical columns, match IDs, prediction timestamps, target-availability timestamps and bases, targets, feature values, and applied imputations.

Model-training metadata records the same result-eligibility policy, and temporal backtest IDs are derived from the fold policy plus the exact train/evaluation dataset identities.

Changing temporal eligibility semantics therefore changes the dataset/model/backtest identity instead of silently reusing an old artifact name.

## Research interpretation

This policy intentionally favors defensible chronology over maximum sample reuse. A future source with trustworthy historical final-result publication timestamps can support a more precise, separately versioned policy without changing the meaning of `completed_result_eligibility_v1`.

## Non-goals

This policy does not claim to reconstruct exact final-whistle timestamps, stoppage-time duration, source publication latency when unreported, or local-time chronology for date-only records. Those unknowns remain unknown rather than being replaced by false precision.
