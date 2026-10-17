# Football Data Analytics Software

Football Data Analytics Software is an analytics and forecasting platform for senior men’s national-team football. The system combines rating models, calibrated machine learning, expected-goals modeling, market benchmarking, and post-match diagnostics to produce reproducible football intelligence across international matches.

[![CI](https://github.com/jennasilvera/football-data-analytics-software/actions/workflows/ci.yml/badge.svg)](https://github.com/jennasilvera/football-data-analytics-software/actions/workflows/ci.yml)

The native `football_analytics` package provides an executable research and operational platform. The retained `wc_forecast` package preserves legacy workflows and regression coverage. **Synthetic demonstrations verify software behavior and not predictive performance or commercial readiness.** The native implementation is complete; remaining real-data and production release gates are listed in the [completion and handoff](docs/implementation_status.md#completion-and-handoff).

## Scope and data integrity

Official senior men's A-internationals: World Cup and qualifiers, continental tournaments and qualifiers, Nations Leagues, friendlies, FIFA Series, recognized regional competitions and playoffs. Clubs, women, youth, Olympic/U23, B/select teams and unofficial matches are excluded. Unknown identities or scope are quarantined, not guessed.

Targets are **regulation-time goals and 1X2 outcomes**. Unknown score bases, extra-time totals and shootout tallies cannot silently become regulation targets. Exact kickoff and date-only records have different cutoff rules. Features carry availability, missingness and source lineage. Operational revisions are append-only and visible only after both publication and recording times.

## What is implemented

- Canonical ingestion, explicit aliases, scope validation, duplicate detection, source hashes, reviewable data-onboarding reports and paginated match revision history.
- Elo replay, Glicko-1 rating/deviation history, inactivity uncertainty and team reports.
- Declared form, strength, schedule, competition, rankings, squad, travel, market and manual feature families.
- Frequency, logistic regression, histogram gradient boosting and independent Poisson models; nested temporal temperature calibration and convex ensembles.
- Expanding/rolling backtests, paired comparisons, calibration diagnostics, sample-aware slices, family ablation, block-bootstrap loss comparisons, market benchmarks and descriptive feature drift.
- Portable JSON model artifacts, content-addressed experiments, model review records and forecast audit trails.
- Versioned authenticated FastAPI service, operational CLI, Streamlit dashboard, SQLite persistence, online backup and recovery acceptance.
- Pinned dependencies, nonroot containers, structured request logging and CI for tests, research, operations and container startup.

Forecasts contain 1X2 probabilities, Poisson expected goals and score grid, entropy, feature drivers, model/data identities, exact operational input revision references, freshness and market disagreement when available. Match-probability intervals remain explicitly **unvalidated/null**. Glicko intervals describe the rating scale. Logistic drivers are model contributions, not causal explanations or ensemble attribution.

See [reliability validation](docs/reliability_validation.md) for process-failure, concurrency, dependency and recovery requirements.

## Install and verify

Python 3.12 is the tested runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
ruff check .
mypy src/football_analytics
pytest
make demo-v2
make demo-v2-comparison
make demo-v2-nested
make demo-v2-opponent-poisson
make demo-platform
```

Research demonstrations write content-addressed reports under `outputs/v2-research/`. `demo-platform` uses an isolated temporary database and explicitly date-shifted synthetic data, then checks the complete native operational path and recovery. It does not approve a real deployment model. `make demo` runs the retained legacy pipeline.

## Use the platform

```bash
football-analytics research --help
football-analytics train --help
football-analytics validate-data --help
football-analytics ingest-results --help
football-analytics predict --help
football-analytics match-history --help
football-analytics forecast-lineage --help
```

Daily fixture windows can be forecast atomically with `predict-batch`; durable run IDs support retries and the dashboard exposes batch audits. See the [native daily runbook](docs/daily_forecast_runbook.md).

Forecast views include an on-demand input-revision report and JSON download, with an explicit distinction between historical lineage and current approval.

The dashboard’s Match history workspace lets you select stored matches, inspect source provenance and revisions at a historical cutoff, and page through a fixed snapshot.

The operational flow is ingestion → training candidate → chronological evaluation → `import-research` → explicit `approve-model` review → stored-fixture prediction. Research linkage verifies source/catalog/context identity, metric consistency and serving-family coverage. Approval binds the exact model and evidence revisions; new evidence requires re-review, and `revoke-model` records a serving stop. Approval records a human decision; it does not establish statistical validity.

```bash
export FOOTBALL_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
football-analytics serve --database outputs/platform.sqlite3
# In a second terminal with the same environment:
football-analytics dashboard
```

Use the dashboard's password field for the API key. The dashboard defaults to `http://127.0.0.1:8000/api/v1`. Canonical API routes use the `/api/v1` prefix; the original unversioned URLs remain compatibility aliases. `/health` and `/version` are public; business endpoints require `Authorization: Bearer <key>`. Interactive API schemas are at `/docs`. Container setup and model operations are in the [deployment guide](docs/deployment_guide.md).

## Architecture and documentation

| Need | Reference |
|---|---|
| Components and operational boundaries | [Architecture](docs/architecture.md) |
| Exact fields, features and missingness | [Data dictionary](docs/data_dictionary.md) |
| Provenance and historical revision limits | [Data lineage](docs/data_lineage.md) |
| Model assumptions and validation | [Model card](docs/model_card.md), [Poisson](docs/poisson_model_card.md), [temporal calibration](docs/temporal_postprocessing.md) |
| Research commands and artifacts | [Research workflow](docs/v2_research_workflow.md), [diagnostics](docs/evaluation_diagnostics.md) |
| API and CLI | [API reference](docs/api_reference.md), [deployment](docs/deployment_guide.md) |
| Requirement coverage and remaining gates | [Implementation status](docs/implementation_status.md), [commercial readiness](docs/commercialization_notes.md) |

The native implementation is organized under `src/football_analytics/{data,domain,features,ratings,models,evaluation,experiments,services,storage,api,reports}`. Transport code delegates to typed domain and research services; model inference does not deserialize executable pickle files.

## Operating and research limits

The supplied data is synthetic. Licensed production feeds, verified historical publication/revision archives, an untouched real-data evaluation period and agreed service SLOs are external acceptance requirements. Training from a flat historical CSV requires an explicit historical-availability assumption. The model bundle freezes its training history and feature context; ingestion alone does not refresh it. Retrain and review candidates before their configured age limit (90 days by default).

SQLite and a shared API key target a single-instance research service. Internet-facing or multi-tenant deployments need managed TLS, user identity/authorization, rate limiting, monitoring, incident ownership and tested capacity. Containers are deployment artifacts; no public service is automatically provisioned by this repository. Consult [commercial readiness](docs/commercialization_notes.md) before describing the product as validated for commercial use.

Every upstream dataset needs a documented license and permitted use. Market comparisons are analytical benchmarks; they make no claim of betting profitability.


The opt-in [opponent-adjusted Poisson challenger](docs/opponent_adjusted_poisson.md)
jointly estimates attack, defence and home effects. Run `--model poisson-comparison`
for chronological comparison against the existing goal baseline; exported JSON
artifacts support standalone replay. This research option does not change serving
models or establish a real-data performance advantage.
