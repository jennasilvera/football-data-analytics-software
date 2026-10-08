# Historical confederation-pair frequency baseline

This benchmark estimates regulation-time home/draw/away probabilities by the
**ordered pair of home and away confederations**. Reversing home/away changes the
pair. Membership comes from explicit historical releases, not the current team catalog.

## Fitting rule

For eligible training outcomes, let `N_k` be global class counts and `N` the total.
With positive smoothing `a`, `p_k = (N_k + a) / (N + 3a)`. For a known pair `g`
with counts `n_gk` and total `n_g`, use
`p_gk = (n_gk + s * p_k) / (n_g + s)` with nonnegative prior strength `s`.
Defaults `a=1` and `s=10` are recorded in model metadata. They are declared benchmark
settings, not empirical optima or a validated uncertainty model.

All training rows contribute to the global distribution. Rows with unknown membership
do **not** form an artificial fitted unknown group. At prediction time, unknown
membership or a known pair absent from training returns global `p_k`. Known sparse
pairs shrink toward global probabilities. With `s=0`, an observed pair can assign
zero probability to outcomes absent from that pair's training history.

## Temporal integrity

Each historical feature row resolves both teams for its match date using the latest
complete membership release published by its prediction cutoff. Even if a revision
is known by a later fold-training date, it does not retrospectively replace the input
that would have been known for that historical forecast. Outcome eligibility and
expanding/rolling fold boundaries follow the existing research policy.

The 36 pair codes are nominal identities in fixed alphabetical confederation order;
zero denotes the explicit global fallback. The mapping is hashed into the feature
version. Known codes are observed values; unknown codes carry an imputed status,
reason and named fallback method. Selected membership release IDs remain in feature
lineage, and CLI research reports embed the complete input releases and diagnostics.
Appending an unavailable future release cannot alter existing feature values, fitting
identities or predictions; its existence still changes the full report's input provenance.

## Use and validation

```bash
make demo-v2-confederations
# The target compares global and confederation frequencies, with historical slice reports.
```

Native `research` accepts `--model confederation-frequency` or
`--model confederation-comparison`, both requiring `--membership-history FILE.json`.
An explicitly empty history is permitted: every forecast then uses the global baseline.
Additional feature-group/context overrides are rejected. Membership IDs must belong
to the supplied canonical team catalog.

Python clients can use `confederation_frequency_spec`, `train_confederation_frequency`,
`ConfederationPairProvider` and `export_classifier`. Portable JSON validates pair codes,
probability distributions, artifact identity and exact feature contracts. Existing
operational bundles and the four-member default ensemble do not automatically adopt
this research model. Any adoption requires a new declared evaluation and review.

The demo's membership timestamps are synthetic. Tests establish formulas, temporal
boundaries, unknown/unseen fallback, orientation, lineage and portable replay. They do
not establish real confederation-specific skill. Obtain licensed, verifiable historical
publication archives and adequate held-out samples before drawing performance conclusions.
