# Temporal probability calibration and ensembles

Native research can now fit one temperature per base model and a convex ensemble
of the uncalibrated base probabilities. All outputs refer to regulation-time
home/draw/away outcomes. These are research estimators, not a claim of better
real-world forecasts.

## Run

```bash
make demo-v2-nested
```

This runs four base models, four temperature-scaled variants and one ensemble on
the same outer evaluation matches. The synthetic demo uses a 21-day inner
holdout and a minimum of five available results to exercise the code. That small
sample is not an acceptable justification for deployment. The CLI's default
minimum is 30, which also does not establish adequate statistical power.

For a governed source, use the existing `research --model all` arguments plus
`--postprocess-days DAYS --min-postprocess COUNT`. Without `--postprocess-days`,
the existing uncalibrated workflow is unchanged.

## Chronology and information boundaries

For each outer cutoff C and declared holdout duration H:

1. Fit inner base models at C-H, using only results available by C-H.
2. Generate pre-match base probabilities in [C-H, C).
3. Retain only holdout outcomes available by C. Audit excluded late results.
4. Fit temperatures and weights on those held-out predictions and outcomes.
5. Refit base models at C, then apply the frozen transforms during outer evaluation.

Each fitting row records its prediction time, target-availability time, actual
outcome and probability vector. Base evidence records spec/model IDs, fitting
cutoff and training match IDs. Validation rejects duplicate held-out IDs,
in-sample predictions, base models fitted after prediction, late targets, missing
members, insufficient observations and samples without all three outcome classes.
No identity-transform fallback conceals a fitting failure.

Every inner fitting label must be in the outer training population. For rolling
windows, the inner base window is shortened by H so its lower bound equals the
outer training lower bound. H must therefore be shorter than that window. Every
declared outer fold must be valid; nested research does not silently skip it.

Date-only prediction cutoffs can precede the match date. A prediction inside an
inner holdout may therefore concern a result unavailable at C even under the
next-day policy. Such rows are explicitly excluded from transform fitting.
Inner backtest metrics remain descriptive audit output and do not select weights,
temperatures, holdout length or outer models. Only the eligible fitting rows
enter the optimization. Outer labels enter scoring, not fitted transforms.

Rolling-form features can update from newly available results during outer
evaluation under the existing pre-match policy. Consequently, changing an early
outer result can affect later outer features, but cannot alter that fold's
already fitted base parameters, temperature or ensemble weights.

## Estimators

Temperature scaling uses `softmax(log(max(p, 1e-12)) / T)`, with T constrained to
[0.25, 4]. It minimizes held-out multiclass negative log likelihood. A bounded
scalar search is compared against both boundaries and T=1. Zero inputs receive
the declared numerical floor. This single scalar cannot correct arbitrary
class-specific bias.

The ensemble minimizes held-out negative log likelihood plus
`0.01 * sum((w - uniform_weight)^2)`, with non-negative weights summing to one.
It uses SLSQP initialized at uniform weights; invalid, non-converged or inferior
to-initialization solutions fail explicitly. The API permits a predeclared
non-negative regularization strength. The ensemble fits raw base probabilities,
not probabilities calibrated on the same holdout.

Temperatures and weights are transferred from the inner fitted models to the
same model specifications refitted at C. This assumes calibration and relative
performance remain useful after refitting; outer evaluation measures that
assumption. Hyperparameter/holdout selection still needs a separate design and
an untouched final test period. There is no automatic model promotion.

## Artifacts and replay

Research JSON schema 5 retains `nested_holdout` with inner backtests, exact eligible
fitting samples, unavailable IDs, transform state and outer base-model bindings.
The ordinary `runs` and comparison include all nine estimators. Each derived run
has calibration diagnostics and its own experiment manifest. Combined model IDs
bind the transform ID to the specific outer base model IDs. Evaluation dataset
IDs include the actual outer input probability records.

Immutable `probability_transform_<sha256>.json` artifacts contain validated JSON,
not pickle. Loading verifies the target policy, temporal evidence, configuration
and content identity. The digest detects content changes; it is not a signature.
Use `load_probability_transform(path)` and
`model.predict({spec_id: OutcomeProbabilities(...)}, prediction_time=...)` for
probability-level replay. The mapping must contain exactly the fitted member
specifications, and prediction time cannot precede fitting. The caller supplies
the base probabilities; the transform is not a complete standalone fixture
forecaster. Native classifier persistence remains a separate milestone.

Derived outputs contain calibrated/combined 1X2 probabilities only. They do not
claim the original Poisson score grid, expected goals or modal score are also
calibrated. Original Poisson outputs remain available in the raw baseline run.

## Verification and limitations

Tests cover portable replay and tampering, invalid temporal evidence, missing
classes, sample minima, simplex constraints, zero probabilities, repeated-run
identity, late-result exclusion, rolling population bounds, CLI persistence and
outer-label isolation. CI also runs the full nine-estimator synthetic workflow.

Real-source revisions, market settlement compatibility, sample-size uncertainty,
subgroup reliability and drift still require validation. A lower synthetic
log loss demonstrates neither generalization nor commercial readiness.
