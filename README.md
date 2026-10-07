# Football Data Analytics Software

[![Football Data Analytics CI](https://github.com/jennasilvera/football-data-analytics-software/actions/workflows/ci.yml/badge.svg)](https://github.com/jennasilvera/football-data-analytics-software/actions/workflows/ci.yml)

Football Data Analytics Software is a research-grade analytics and forecasting platform under active development for **senior men’s national-team football**. It estimates pre-match probabilities and evaluates forecast quality with reproducible data and model lineage.

The project is a maintainable analytical system with explicit data contracts, reproducible research, leakage controls, model/version lineage, and interfaces that can support both research and production workflows.

## Scope

Primary scope:

- Senior men's A-international matches
- FIFA World Cup and qualification
- Confederation championships and qualification
- Nations League competitions
- International friendlies
- Other recognized senior men's national-team competitions

Explicitly out of scope for the core dataset unless modeled separately:

- Women's internationals
- Olympic/U-23 matches
- Youth internationals
- B teams and select teams
- Club and academy football
- Unofficial matches

Unknown scope metadata should be reviewed or quarantined rather than guessed.

## Current Repository State

The repository currently contains two architectural generations.

### Legacy implementation: `wc_forecast`

The existing system remains operational while the replacement architecture is introduced. It includes:

- Historical results ingestion and validation
- Custom Elo ratings
- Poisson expected-goals modeling
- Logistic regression, gradient boosting, and random-forest baselines
- Rolling and chronological backtests
- Feature ablation and model-selection utilities
- Calibration and uncertainty analysis
- Market-implied probability and closing-line-value analysis
- Player-availability and market-movement signals
- Monte Carlo group-stage simulation
- SQLite-backed feature, model, and prediction persistence
- FastAPI service components
- CLI workflows
- Automated tests and GitHub Actions CI

This package is retained during migration so useful behavior can be preserved and tested rather than discarded in a greenfield rewrite.

### V2 implementation: `football_analytics`

The integrated V2 foundation is on `master`. Native scoreline modeling and paired model comparison extend that foundation while preserving the legacy pipeline.

The integrated foundation currently includes:

- Canonical Team, Competition, and Match domain types
- Explicit exact-kickoff vs date-only temporal precision
- Source provenance and temporal metadata
- `available_at` / `ingested_at` point-in-time contracts
- Leakage-risk classification
- Senior men's A-international scope validation
- Canonical team and competition entity resolution
- Explicit alias handling
- Unresolved-record quarantine
- Competition-agnostic tabular ingestion
- Normalized / excluded / quarantined batch outputs
- Deterministic source snapshot hashes
- Canonical catalog loading
- Legacy-results migration adapters
- Replaceable rating-engine contracts
- Legacy Elo parity adapter
- Deterministic canonical rating replay
- Immutable rating snapshots
- Point-in-time feature contracts
- Explicit observed / missing / imputed feature states
- Competition, squad availability, travel, and market snapshot feature providers
- Native class-frequency, logistic, histogram gradient boosting, and Poisson models
- Paired model comparison with identical training and evaluation samples
- Competition/team/venue/year diagnostics and auditable per-match forecast losses
- Nested temporal temperature scaling and learned convex probability ensembles
- Explicit regulation-time score targets; unknown, extra-time and shootout scores fail closed
- Portable Poisson model artifacts and scoreline forecast replay
- Target-availability-safe temporal backtesting and calibration diagnostics
- Content-addressed experiment manifests and a JSON registry
- Executable V2 research service and CLI with auditable reports
- Separate prediction-time and target-availability semantics for supervised examples
- Conservative completed-result eligibility when exact publication time is unknown
- Content-addressed V2 model datasets with temporal/imputation policy identity
- Native logistic-regression and histogram-gradient-boosting baselines
- Expanding- and rolling-window temporal fold contracts
- Log loss, multiclass Brier score, Ranked Probability Score, and accuracy
- Deterministic per-fold model training and out-of-sample backtest artifacts

The migration policy is simple: **preserve proven behavior, replace unsafe contracts, and change modeling assumptions only after parity is measurable.**

## Core Engineering Principles

### 1. Point-in-time correctness

A model feature must be demonstrably available at the prediction cutoff.

For a forecast at time `t`:

```text
feature.available_at <= t <= kickoff_at
```

If historical availability is unknown, the observation may remain useful for archival or post-match analysis, but it cannot silently become a pre-match feature.

### 2. No invented precision

Historical sources frequently provide only a match date.

The platform represents that as date-only data rather than inventing a midnight kickoff. The same rule applies to publication timestamps, team identities, competition mappings, and other metadata.

### 3. Canonical entities before modeling

Source labels are not model identities.

Team and competition names are resolved to canonical entities before downstream ratings, features, or forecasts are produced. Unknown entities are quarantined instead of receiving a synthetic identity or default rating.

### 4. Leakage prevention is architectural

Chronological ordering alone is not enough.

The V2 design couples data values to source, event time, availability time, ingestion time, lineage, and leakage risk so temporal validity can be enforced at feature-build and backtest boundaries.

### 5. Research and production concerns are separated

Model research should not be entangled with:

- Source ingestion
- Entity resolution
- Persistence
- API transport
- CLI orchestration
- Market decision policy

The system is being decomposed into explicit interfaces so those layers can evolve independently.

### 6. Migration before modification

The current Elo implementation is first migrated behind a stable V2 rating contract and parity-tested against the legacy behavior.

Only after parity exists should research changes such as alternative K-factors, Glicko-style uncertainty, competition weighting, or new rating systems be evaluated.

## Architecture Direction

```text
External / historical sources
        |
        v
Source-specific adapters
        |
        v
Source observations + provenance
        |
        v
Scope validation
        |
        v
Canonical entity resolution
        |
        +--> excluded records
        |
        +--> quarantined records
        |
        v
Canonical matches
        |
        +--> rating replay / snapshots
        |
        +--> point-in-time feature providers
        |
        v
Versioned feature vectors
        |
        v
Probabilistic models
        |
        v
Chronological / rolling evaluation
        |
        +--> calibration
        +--> diagnostics
        +--> research slices
        |
        v
Forecast services / APIs / analytical outputs
```

Market prices can be used as research benchmarks, calibration references, and comparative signals. Betting or staking policy is intentionally treated as an optional downstream extension rather than the core identity of the platform.

## Repository Structure

Both `wc_forecast` and `football_analytics` are retained. The integrated V2 branch provides a native research command alongside the operational legacy CLI.

```text
.
├── data/
│   ├── sample/
│   ├── raw/
│   └── processed/
├── docs/
├── outputs/
├── reports/
├── src/
│   ├── wc_forecast/         # Current operational implementation
│   └── football_analytics/  # Added by the staged V2 reimplementation
├── tests/
│   └── v2/                  # Added by the staged V2 reimplementation
├── .github/workflows/
├── Makefile
└── pyproject.toml
```

The native research CLI and the legacy CLI are both available. See the [implementation status](docs/implementation_status.md) for completed capabilities and remaining requirements.

## Quickstart: Native V2 Research

After installing the dependencies below, run:

```bash
make demo-v2
make demo-v2-comparison
python -m football_analytics research --help
```

This runs canonical ingestion → point-in-time form features → temporal backtest
→ calibration diagnostics → experiment registration. It writes a content-addressed
JSON report containing predictions, fold metrics, source/catalog provenance, and
runtime versions under `outputs/v2-research/`.

The sample is synthetic and validates the workflow, not predictive performance.
The CLI exposes rolling-form classifiers, a training-only class-frequency baseline,
and an independent Poisson score model. `research --model all` produces a paired
comparison report and replayable Poisson artifacts. Other feature providers remain
available through their typed interfaces. See [workflow semantics and limitations](docs/v2_research_workflow.md).

## Quickstart: Existing Operational Pipeline

Python 3.12+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run quality checks:

```bash
ruff check .
pytest
```

Run the existing reproducible pipeline:

```bash
make demo
```

The legacy CLI remains available during migration:

```bash
python -m wc_forecast health
python -m wc_forecast ingest-results data/sample/historical_results_sample.csv
python -m wc_forecast build-elo
python -m wc_forecast build-features
python -m wc_forecast backtest-logistic
python -m wc_forecast report-backtest
python -m wc_forecast predict-poisson Argentina France
python -m wc_forecast report-match Argentina France
```

These commands validate the existing system. They are not the final V2 interface design.

## Validation Strategy

The project favors chronological and point-in-time evaluation over random train/test splits.

Current and planned evaluation includes:

- Expanding-window backtesting
- Rolling-origin evaluation
- Log loss
- Multiclass Brier score
- Ranked Probability Score
- Calibration and reliability diagnostics
- Feature ablation
- Hyperparameter tuning
- Competition-level slices
- Confederation-level slices
- Temporal robustness checks
- Model-disagreement and entropy diagnostics
- Market benchmark comparison
- Prediction and model version lineage

Forecast probabilities are estimates, not guarantees.

## Data Governance

Data sources should have documented:

- Provider/source identity
- Access method
- Source version or snapshot
- Event timestamp where known
- Availability timestamp where known
- Ingestion timestamp
- License or legal-use notes
- Transformation assumptions
- Canonical entity mappings
- Leakage risk

The V2 source layer is designed so unknown metadata remains unknown rather than being replaced by convenient defaults.

## Implementation Status

The [requirements map](docs/implementation_status.md) distinguishes native V2,
legacy-only, and planned capabilities. The [Poisson model card](docs/poisson_model_card.md)
records assumptions, temporal controls, and limitations. Learned recalibration,
validated uncertainty intervals, native team intelligence, and commercial deployment
remain unfinished.

## Reimplementation Strategy

The project is being migrated in controlled slices.

### Foundation

- Canonical domain
- Temporal/provenance contracts
- Entity resolution
- Formal product scope
- Generic source adapters
- Quarantine/exclusion paths
- Canonical catalogs
- Data snapshot hashing

### Rating migration

- Replaceable rating protocol
- Legacy Elo adapter
- Immutable rating snapshots
- Deterministic replay
- Legacy parity tests
- Ambiguous historical chronology checks

### Feature architecture

- Point-in-time feature definitions
- Feature-set versioning
- Missingness semantics
- Imputation lineage
- Rating features
- Form and context providers
- FIFA-ranking provider
- Schedule/rest/travel features
- Leakage regression tests

### Modeling and evaluation

Implemented foundation:

- Native V2 probabilistic model interfaces
- Logistic-regression and histogram-gradient-boosting baselines
- Content-addressed training datasets and deterministic training identities
- Expanding- and rolling-window split policies
- Target-availability-safe temporal training folds
- Accuracy, log loss, multiclass Brier score, and Ranked Probability Score
- Per-fold and aggregate out-of-sample evaluation

Next research layers:

- Poisson migration
- Calibration diagnostics and calibration layer
- Ensemble research using out-of-sample base predictions
- Competition/confederation evaluation slices
- Market benchmark comparison
- Persisted experiment/model artifact registry

### Product interfaces

- Application/service layer
- Modular CLI
- Versioned FastAPI routes
- Analytical dashboard
- Scheduled data refresh and forecast workflows
- Monitoring and alerting

The detailed migration plan is maintained with the V2 reimplementation branch and is merged with the architecture it describes.

## Development Policy

A change is not considered complete only because it adds a model or feature.

Professional-grade changes should also address, where applicable:

- Input contract
- Temporal semantics
- Provenance
- Failure behavior
- Tests
- Reproducibility
- Backward compatibility
- Observability
- Documentation
- Research validation

Large migrations are intentionally split into reviewable pull requests rather than accumulated into one unbounded rewrite.

## CI

GitHub Actions runs:

1. Ruff linting
2. V2 static type checks with mypy
3. Full pytest suite
4. Existing end-to-end sample forecasting pipeline

The legacy pipeline remains in CI during V2 migration to catch regressions while replacement components are introduced.

## Research and Commercial Direction

The architecture is intended to support work beyond one competition or one prediction surface, including:

- National-team strength estimation
- Match probability forecasting
- Team and competition analytics
- Schedule and travel effects
- Player availability
- Tournament simulation
- Model calibration
- Market benchmarking
- Research APIs
- Forecast audit trails
- Model and data versioning
- Reproducible experiment tracking

Commercial use would require additional work around licensed data, reliability SLOs, persistence infrastructure, authentication, observability, operational support, and deployment architecture. The repository is being designed so those concerns can be added without rewriting the research core.

## Important Limitations

The repository is under active architectural migration.

The existing `wc_forecast` system contains useful tested functionality, but several legacy assumptions are being replaced, including:

- World Cup-specific fixture defaults
- Implicit team aliases
- Default ratings for unresolved forecast teams
- Weak source-level availability semantics
- Coupled feature/model state
- Broad CLI orchestration responsibilities

Do not interpret demo outputs as evidence of guaranteed predictive performance or betting profitability.

## License and Data Use

Before production or commercial use, verify the license and permitted use of every upstream dataset.

Source provenance and legal-use notes should be stored alongside ingestion definitions rather than assumed from a URL or file name.

Native calibrated/ensemble research: `make demo-v2-nested`. See
[temporal postprocessing](docs/temporal_postprocessing.md) for fitting chronology,
artifact replay and limitations.

Every native research run also writes [evaluation diagnostics](docs/evaluation_diagnostics.md),
including sample counts, slice metrics and the largest match losses.
