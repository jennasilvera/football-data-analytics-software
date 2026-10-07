# V2 Reimplementation Plan

## Purpose

This document defines the controlled migration from the current `wc_forecast`
implementation to **Football Data Analytics Software**, a research-grade and
commercially extensible analytics platform for senior men's A-international
football.

The V2 effort is a reimplementation, not a greenfield rewrite. Existing code is
retained when it is correct, tested, and compatible with the new contracts.
Legacy code remains operational on `master` while the staged V2 components are
reconciled on `reimplementation/v2-integrated-research`. The executable research
workflow and remaining gaps are documented in [V2 research workflow](v2_research_workflow.md).

## Baseline audited

Baseline commit:

```text
5f1252f Add market movement analytics
```

The current repository already contains meaningful engineering work:

- historical results and fixture ingestion
- custom Elo ratings
- rolling form features
- logistic regression, histogram gradient boosting, and random forest models
- Poisson scoreline modeling
- ensemble probabilities
- rolling-origin backtests
- feature ablation and logistic tuning
- calibration and closing-line-value reports
- market movement and player-availability signals
- prediction ledgers and SQLite-backed storage
- a lightweight model registry and feature store
- FastAPI endpoints
- a static dashboard
- GitHub Actions CI
- more than 30 test modules

V2 must build on these assets rather than discard them.

## Architectural gaps

### 1. No canonical domain layer

The legacy package passes team names, tournament names, match identifiers, dates,
and status values primarily as strings and dataframe columns. There is no
authoritative representation of teams, competitions, matches, venues, rankings,
or canonical entity IDs.

V2 introduces `football_analytics.domain` first. All later adapters and models
must depend on canonical domain objects or schemas instead of source-specific
strings.

### 2. No first-class point-in-time data contract

Current leakage prevention is strongest inside chronological Elo and rolling-form
replay. That is useful but insufficient for a platform that will ingest rankings,
odds, squads, injuries, suspensions, and manual intelligence.

The existing feature store records `feature_date` and `created_at`, but not the
more important `available_at` timestamp that answers: *could the model actually
have known this at prediction time?*

V2 requires every predictive observation to carry provenance and temporal
metadata including:

- source
- source record ID where available
- source version where available
- event time where meaningful
- available-at time
- ingestion time
- leakage-risk classification
- legal-use notes where applicable

### 3. Data ingestion is still World Cup-oriented

`normalize_world_cup_fixtures` defaults missing tournaments to `FIFA World Cup`
and missing neutral-site values to `True`. The API also contains World Cup 2026
default paths.

V2 ingestion must be competition-agnostic and validate formal scope:

Included:
- FIFA World Cup and qualifiers
- continental championships and qualifiers
- Nations League competitions
- international friendlies
- FIFA Series
- regional senior tournaments
- intercontinental playoffs

Excluded:
- clubs
- women's matches
- Olympic/U-23 matches
- youth matches
- B teams
- unofficial/select/academy teams

### 4. Entity resolution is implicit

Legacy fixture forecasting contains hand-coded team aliases and fallback Elo
ratings. V2 requires a canonical entity-resolution subsystem with stable IDs,
aliases, source mappings, unresolved-entity quarantine, and audit logs.

A missing team mapping must not silently become a generic 1500 rating in a
production forecast.

### 5. Feature generation is coupled to model state

`build_match_features.py` owns Elo replay, form replay, tournament context,
defaults, result columns, and target construction in one module.

V2 will use independent point-in-time feature providers. A feature provider must:

1. declare the data it consumes;
2. declare feature names and versions;
3. compute values as of a prediction timestamp;
4. preserve provenance;
5. expose missingness rather than silently hiding all gaps behind neutral defaults.

### 6. Model naming and interfaces are too narrow

`train_logistic_regression` can train three different classifier families. V2
needs a model protocol and registry so model type, feature set, calibration layer,
training window, artifact version, and evaluation metadata are explicit.

### 7. Backtesting is useful but incomplete for research-grade governance

The rolling-origin backtester is a strong reusable base. It must be expanded to
support:

- expanding-window and rolling-window protocols
- ranked probability score
- calibration error and reliability diagnostics
- competition and confederation slices
- favorite/underdog and draw-specific analysis
- market benchmark comparisons
- feature ablation linked to experiment IDs
- stored fold definitions and data snapshot hashes

### 8. API and CLI need service boundaries

The FastAPI service is useful but currently exposes a small World Cup-oriented
surface and directly calls file-based forecasting functions.

The CLI is approximately 90 KB and has accumulated many commands in one module.

V2 will move orchestration into application services so FastAPI, CLI, scheduled
jobs, notebooks, and dashboards call the same tested service layer.

### 9. Storage needs temporal and repository abstractions

SQLite is appropriate for local development and reproducible demos and should be
retained initially. Storage code should move behind repository interfaces so a
later PostgreSQL deployment does not alter domain or modeling code.

### 10. Betting/decision modules should be isolated from the core analytics product

Market benchmarking is a core research capability. Stake sizing and betting
policy are not required for the primary Football Data Analytics Software product.
Legacy strategy/staking modules should remain available as an optional research
extension, not define the platform architecture or public API.

## Migration decisions

| Legacy component | Decision | V2 destination / treatment |
|---|---|---|
| `data/ingest_results.py` | REWRITE | source adapter + canonical match normalizer |
| `data_sources/international_results.py` | ADAPT | source-specific adapter with provenance |
| `data/ingest_fixtures.py` | REWRITE | competition-agnostic fixture ingestion |
| `models/elo.py` | ADAPT | `ratings/elo.py` behind rating protocol |
| `features/build_features.py` | REWRITE | point-in-time feature providers |
| `models/classifier.py` | ADAPT | generic model trainer implementations |
| `models/poisson.py` | ADAPT | score model implementation |
| `models/ensemble.py` | ADAPT | calibrated ensemble layer |
| `validation/rolling_backtest.py` | ADAPT | unified evaluation/backtest engine |
| `validation/feature_ablation.py` | ADAPT | experiment/ablation runner |
| `reports/calibration.py` | ADAPT | evaluation calibration module |
| `reports/forecast_audit.py` | ADAPT | richer forecast provenance audit |
| `storage/database.py` | ADAPT | local SQLite repository implementation |
| `storage/feature_store.py` | REWRITE | point-in-time feature store |
| `storage/model_registry.py` | ADAPT | governed model/experiment registry |
| `storage/prediction_store.py` | ADAPT | forecast repository |
| `api.py` | REWRITE | versioned FastAPI routers + services |
| `cli.py` | REWRITE | thin modular CLI commands + services |
| `simulation/group_stage.py` | ISOLATE | optional tournament simulation module |
| `strategy/*` | ISOLATE | optional market-research extension |
| legacy reports/docs | ADAPT/ARCHIVE | V2 docs and reproducible report generation |

## Migration architecture

```text
External sources
      |
      v
Source adapters
      |
      v
Raw immutable observations
      |
      v
Entity resolution + canonicalization
      |
      v
Temporal/provenance validation
      |
      +-------------------+
      |                   |
      v                   v
Rating snapshots     Point-in-time features
      |                   |
      +---------+---------+
                |
                v
           Model registry
                |
                v
             Training
                |
                v
            Calibration
                |
                v
        Forecast artifact
                |
       +--------+---------+
       |        |         |
       v        v         v
      API      CLI      Reports
                |
                v
     Backtesting / diagnostics
```

## V2 package migration rule

The new package is `football_analytics`.

The existing `wc_forecast` package remains intact during migration. New V2
modules may wrap tested legacy algorithms temporarily, but new domain, temporal,
storage, evaluation, and service contracts must not depend on World Cup-specific
defaults.

The legacy package will be removed only after parity tests prove that each retained
capability has migrated successfully.

## Phase sequence

### Phase 1 — canonical domain and temporal contracts

Deliverables:
- team, competition, and match domain types
- timezone-safe kickoff timestamps
- source/provenance metadata
- explicit `available_at` semantics
- leakage-risk classifications
- pre-match availability validation
- contract tests

### Phase 2 — canonical data layer

Deliverables:
- source registry
- raw observation envelopes
- entity resolution
- stable team and competition IDs
- senior men's A-international scope validation
- fixture/result normalization
- data lineage
- quarantine for unresolved or excluded rows

### Phase 3 — rating migration

Deliverables:
- rating protocol
- Elo adapter migrated from legacy implementation
- immutable rating snapshots
- rating history
- no-fallback production behavior
- Glicko-style uncertainty implementation

### Phase 4 — point-in-time feature system

Deliverables:
- feature definitions and versions
- Elo/Glicko/FIFA strength providers
- rolling form providers
- competition/venue/rest providers
- missingness and uncertainty features
- temporal feature-store schema

### Phase 5 — model and experiment system

Deliverables:
- baseline model protocol
- logistic regression
- gradient boosting
- Poisson
- calibrated ensemble
- training run metadata
- model artifact registry
- reproducible seeds and dataset hashes

### Phase 6 — evaluation and governance

Deliverables:
- expanding and rolling backtests
- Brier score, log loss, RPS
- calibration diagnostics
- competition/confederation slices
- market benchmark comparisons
- ablation and error analysis
- model cards and experiment reports

### Phase 7 — product interfaces

Deliverables:
- modular CLI
- versioned FastAPI routers
- team/match/rating/prediction/report endpoints
- application service layer
- research dashboard

### Phase 8 — deployment readiness

Deliverables:
- Dockerfile and compose stack
- separate CI workflows for lint/test/data/model checks
- PostgreSQL-ready repository implementation
- observability
- deployment guide
- commercial-readiness review

## Acceptance rule

A V2 module is not complete because it reproduces an old output. It is complete
only when it has:

- typed contracts
- temporal integrity
- provenance
- deterministic tests
- documented assumptions
- no hidden World Cup-only behavior
- a migration/parity test when replacing legacy behavior
