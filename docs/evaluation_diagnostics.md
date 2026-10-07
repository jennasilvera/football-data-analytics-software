# Held-out evaluation diagnostics

Every native `research` run now emits descriptive diagnostics for its exact
held-out forecast population. This includes raw models, temperature-scaled
variants and learned ensembles. It does not refit models or select a winner.

## Outputs

Research JSON schema 5 adds a `diagnostics` array keyed by backtest run ID and
model specification ID. Each entry includes:

- Aggregate metrics recomputed from the prediction rows.
- Competition, venue (neutral/home), calendar-year and team slices.
- Exact match IDs, outcome counts, metrics and a small-sample flag for each slice.
- Per-match probabilities, actual and predicted outcomes, realized scores,
  probability assigned to the outcome, log loss, Brier score, RPS and entropy.

The CLI prints a `diagnostics_report` path to an immutable `.diagnostics.md`
sidecar. It contains all slices and up to ten matches with the largest log losses
per model. Full per-match data remains in JSON. Run `make demo-v2-nested` to see
the diagnostics for all nine estimators.

The default display minimum is 30; override with `--min-diagnostic-sample N`.
Small slices remain visible with their sample counts and a warning. Passing the
threshold does not establish statistical power, calibration or reliability.

## Interpretation

Competition, venue and year slices partition the evaluated matches. Their
sample-weighted losses reconstruct the aggregate loss. Team slices overlap:
each fixture contributes to both participants. Summing those counts gives twice
the number of fixtures, and pooling team rows would double-count the sample.
Team slice accuracy measures forecasts of that team's matches, not its win rate.
Home/draw/away counts always retain the fixture perspective, including on neutral
sites. Historical confederation membership is not inferred from current catalogs.

Log loss uses the existing natural-log convention with an outcome probability
floor of 1e-15. Brier is the sum over three classes, and RPS uses home/draw/away
ordering with two cumulative errors averaged. Entropy is in nats; zero
probabilities contribute zero entropy. A surprising outcome is not automatically
a causal model failure, and a correct favorite can still have a high loss.

These reports contain descriptive point estimates. They provide no confidence
intervals, significance tests, causal explanation, feature attribution or drift
alarms. Evaluating many slices and then choosing a model from the best-looking
ones risks selection bias. Model selection needs a predeclared protocol and an
untouched test period.

## Validation and provenance

Diagnostic joins reject missing or duplicate canonical match IDs, repeated
evaluation forecasts, train/evaluation overlap, mismatched fold/model IDs,
non-regulation scores, invalid goals, targets conflicting with canonical results,
and predictions outside their pre-match/fold-start boundaries. Prediction rows
are the metric source; cached summary metrics are not trusted. No inner join
silently reduces coverage.

The builder accepts the canonical records used in the research run. Its digest
includes the exact diagnostic rows, slice metrics, sample threshold, target
policy and backtest identity. Reordering source records does not change the
report. A later canonical correction that changes a result can cause the join
to fail; rebuild the research run from the corrected governed source rather than
editing an old report. Revision-aware historical source storage remains pending.

Tests cover loss formulas including zero-probability outcomes, aggregate/slice
reconciliation, overlapping team counts, invalid joins, threshold behavior and
idempotent CLI output. Existing CI native demos exercise all diagnostics paths.
