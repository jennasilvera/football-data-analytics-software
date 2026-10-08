# Competition frequency baseline

This native benchmark estimates regulation-time 1X2 probabilities from the training
sample, grouped by canonical competition ID. It provides a stronger contextual
reference than one global frequency vector without introducing team-strength features.

For training counts `N_k`, total `N`, positive smoothing `a`, competition counts
`n_gk`, total `n_g`, and nonnegative prior strength `s`:

- Global probabilities: `p_k = (N_k + a) / (N + 3a)`.
- Observed competition: `p_gk = (n_gk + s * p_k) / (n_g + s)`.
- Competition absent from training: use the global `p_k`.

Defaults are `a=1` and `s=10`, recorded in each model's training specification.
These are declared benchmark settings, not empirically optimized parameters or a
validated uncertainty model. `s=0` gives unsmoothed within-competition frequencies;
that can produce zero probabilities for unseen outcomes. Sparse groups shrink toward
the global distribution when `s>0`. The global distribution includes that group's
training observations; this is an empirical shrinkage rule, not a fully Bayesian model.

## Temporal and identity controls

Every fold recomputes global and competition counts from its eligible training
sample. Future evaluation outcomes cannot change its model identity or predictions.
Expanding and rolling windows share the existing target-availability contracts.
Competition codes are nominal labels, never treated as a distance or ordinal scale.
The code mapping derives from the explicit canonical catalog and its hash is part
of the feature definition. Catalog ordering does not change the mapping; a changed
catalog changes the input contract. Unknown canonical competitions fail validation;
known catalog competitions without training observations use the global fallback.

The model uses only competition identity. Explicit feature-family/context overrides
are rejected instead of implying that additional predictors affect the benchmark.
Global-versus-competition comparisons require identical training and evaluation
identities. Report the competition slices and sample sizes alongside aggregate scores.
The sample dataset has only one competition, so its demo establishes execution and
pairing rather than multi-competition predictive benefit.

## Commands

```bash
make demo-v2 V2_MODEL=competition-frequency
make demo-v2 V2_MODEL=competition-comparison V2_EXTRA_ARGS=--paired-uncertainty
```

The second command compares global and competition frequency models on the same
folds. Existing `--model all` and calibrated/ensemble defaults retain their original
four members. Python clients can call `competition_frequency_spec`,
`train_competition_frequency` and `export_classifier`; portable JSON preserves
within-group probabilities and the global fallback and validates input contracts.
The model is available for native research and typed replay; it is not silently
inserted into existing operational forecast bundles.

No historical confederation membership is inferred from a present-day team catalog.
Confederation baselines still require effective-dated, publication-governed inputs.
