# Experiment Registry

## Purpose

V2 research artifacts need stable identities that can be compared, reproduced, and audited independently of local filenames or database-generated UUIDs.

The experiment registry records what was evaluated. It is not the model binary store and does not decide which model is promoted to production.

## Experiment manifest

An `ExperimentManifest` is an immutable, content-addressed description of one evaluated modeling experiment.

Its identity includes:

- experiment-manifest schema version
- temporal backtest run ID
- optional calibration report ID
- full model specification, family, version, random seed, and parameters
- split policy
- feature-set identity
- historical prediction-cutoff policy
- completed-result eligibility policy
- imputation policy
- aggregate evaluation metrics
- fold IDs
- training-run IDs
- fitted model IDs
- train/evaluation dataset IDs
- optional calibration-bin count and macro ECE
- optional code revision

Changing any identity-bearing field changes `experiment_id`.

## Why metrics are identity-bearing

An experiment manifest is intended to be self-verifying. Reported metrics are therefore included in the content hash rather than treated as mutable annotations.

If a stored metric is edited without recomputing the complete manifest identity, loading the manifest fails.

## Calibration binding

When a calibration report is attached, the manifest builder reconstructs calibration diagnostics from the backtest's own out-of-sample predictions using the same bin count.

The supplied calibration report must have the exact deterministic report ID of that reconstruction. Equal sample size alone is not sufficient.

This prevents a calibration report from another experiment from being attached accidentally to an otherwise valid backtest.

## Local JSON registry

`JsonExperimentRegistry` is the first persistence implementation.

It provides:

- deterministic `<experiment_id>.json` paths
- ID-format validation
- atomic temporary-file replacement
- filesystem flush before replacement
- idempotent writes for identical manifests
- stable sorted listing
- deterministic manifest verification on read
- registry-path ID verification

The JSON backend is suitable for local research, CI artifacts, and reproducible examples. The `ExperimentRegistry` protocol keeps callers independent of the persistence backend.

## Future backends

A later SQLite or PostgreSQL implementation should preserve the same immutable registry semantics rather than inventing new experiment identities.

Database rows may add operational metadata such as insertion time, actor, environment, or promotion status, but those fields must not silently alter the meaning of the content-addressed experiment manifest.

## Separation from model artifacts

The experiment registry does not yet serialize trained sklearn estimators. Model-binary persistence needs a separate artifact contract including serialization format, library/runtime versions, artifact digest, and compatibility policy.

That separation is intentional: research metadata can be made reproducible before choosing a production model serialization strategy.
