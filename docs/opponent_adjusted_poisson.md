# Opponent-adjusted Poisson research model

The original Poisson model estimates strength from each team's average goals for
and against. Those averages mix ability with schedule difficulty. The opt-in
`opponent_adjusted_poisson_v1` challenger fits every team's attack and defence
jointly, conditioning each match's goal rates on both participants.

For home team i and away team j:

```text
log(lambda_home) = baseline + attack_i + defence_weakness_j + home_effect * (not neutral)
log(lambda_away) = baseline + attack_j + defence_weakness_i
```

Attack and defence coefficients each sum to zero. Positive attack increases goals;
positive defence weakness increases goals conceded. Only the home side receives
the learned home effect. Neutral fixtures receive neither a home boost nor an away
penalty. These are statistical goal rates, not shot-based xG.

## Estimation and numerical controls

Each match contributes two conditionally independent Poisson observations. The
objective is the sum of `exp(eta) - goals * eta`, plus half the declared ridge
penalty times the sum of squared raw attack, raw defence and home coefficients.
Constant log-factorials are omitted. The intercept is not penalized. Team effects
are centered inside the objective; the ridge penalty identifies the otherwise
redundant raw mean. Sparse effects therefore shrink toward zero.

| Control | Default / behavior |
|---|---|
| Ridge penalty | 5.0, strictly positive; summed likelihood, not mean likelihood |
| Optimizer | Deterministic L-BFGS-B with analytic gradient and fixed initialization |
| Iteration limit | 1000; nonconvergence fails the run without a model artifact |
| Solver tolerance | 1e-9 for function/gradient termination controls |
| Baseline log-rate bounds | -5 to 3 |
| Raw team-effect bounds | -3 to 3 before centering |
| Home-effect bounds | -2 to 2 |
| Forecast rate bounds | 0.20 to 5.00 goals per team; applied at inference only |
| Score grid | 0 through 10 goals per side; normalized conditional on bounds |
| Tail handling | Omitted probability is reported explicitly |

The artifact records objective value, iterations and the number of fitted raw
parameters at a bound. Bound hits and inference clipping need scientific review;
solver convergence does not establish model adequacy. All-zero scoring samples
fail explicitly. Configuration is declared in the experiment manifest and model
artifact. Custom configurations are available through the typed model-spec API;
the CLI uses the documented defaults.

## Temporal integrity, coverage and replay

Training accepts only canonical completed regulation-time scores available by the
fold cutoff. It rejects unavailable results, duplicate/reversed fixtures, invalid
goals and ambiguous score targets. It never silently filters supplied training data.
Rows and team IDs are sorted before fitting. Dataset identity includes exact goal
values, team/site context, availability and source provenance.

Prediction requires pre-match time at or after the training cutoff, known teams,
and neither cancelled nor postponed fixtures. The training opponent graph is
retained as connected components. Cross-component comparisons fail explicitly:
ridge shrinkage is not evidence that disconnected schedules are comparable. These
failures stop the evaluation rather than changing the comparison sample silently.
Connectivity alone does not guarantee strong identification; sparse, bipartite or
otherwise unbalanced schedules may remain dominated by the declared penalty.

Portable JSON models validate contracts, numeric state, zero-sum constraints,
component partitions and content identity. No executable deserialization is used.
The existing `predict-poisson --model-artifact PATH ...` command accepts either the
legacy-parity or joint model. Nested calibration and serving defaults are unchanged.

```bash
make demo-v2-opponent-poisson
# Equivalent research selection with your governed source and temporal flags:
football-analytics research ... --model poisson-comparison --paired-uncertainty
# Single challenger:
football-analytics research ... --model opponent-poisson
```

Both models use identical temporal folds and canonical targets. Comparison reports
contain proper outcome scores, calibration, paired deltas, diagnostics and full
score forecasts. The first model is the existing baseline. Synthetic parameter
recovery tests use deliberately unequal schedules; other tests check analytic
gradients, deterministic replay, future-label invariance, missing coverage,
regularization and solver failure.

## Evidence required before promotion

Synthetic tests demonstrate implementation behavior only. Evaluate real licensed
publication-aware data on an untouched holdout, examine subgroup calibration and
schedule coverage, tune the penalty inside training-only validation, and compare
against frequency, discriminative models and eligible market quotes. Conditional
independence, constant team effects, absence of time decay and competition-specific
scoring levels, numerical bounds, refit calibration transfer and uncertainty
coverage remain research limitations. No automatic promotion or superiority claim
is made from this challenger.
