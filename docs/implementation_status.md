# Implementation status against the product requirements

Native platform release 0.3.0 completes an executable implementation of the professional upgrade sequence. This map distinguishes software delivery from empirical and operational acceptance. Core scope is official senior men's A-international football with regulation-time targets.

| Product module | Native implementation | Remaining acceptance boundary |
|---|---|---|
| Ingestion, validation, entity resolution | Canonical catalogs, aliases, source hashes, tabular/results/fixtures/rankings/odds imports, scope/duplicate checks and atomic batches | Licensed provider integrations, real coverage and source archives |
| Data and revision integrity | UTC provenance, leakage controls, append-only bitemporal SQLite revisions and as-of reads | Unknown historical revisions cannot be reconstructed |
| Ratings | Elo replay/parity, Glicko-1 periods, deviation/inactivity and rating history | Football-specific tuning and uncertainty coverage |
| Features | Form 5/10/20, strength, schedule, competition, rankings, squad, travel, market, manual context; status/lineage; persisted form vectors | Real exogenous input availability; specialized research signals |
| Forecasting/calibration | Frequency, logistic, histogram boosting, Poisson, held-out temperature transforms and convex ensemble; portable JSON replay | Real-data performance, validated match-probability intervals |
| Research evaluation | Expanding/rolling folds, proper scores, reliability, paired comparison, team/competition/year/venue slices, publication-aware historical confederation slices, family ablation, block loss intervals, cutoff market joins, drift | Verified membership publication archives and untouched real holdout |
| Diagnostics/intelligence | Actual-result surprisal, residuals, miss classification, Glicko movements, form/schedule reports, descriptive watchlists and next stored forecast | Causal explanations and validated future trend signals |
| API | All requested endpoint families, authentication, size limits, stable collection pagination, structured logs | Tenant identity, internet edge policy and measured capacity |
| CLI | Ingestion through candidate training/review/prediction, ratings, reports, drift and backup | Provider-specific scheduled job ownership |
| Dashboard | API-backed overview, ratings, forecasts, team/model diagnostics and post-match views | User research and production-scale interaction testing |
| Governance/docs | Experiment/model identity, source-matched research association, explicit approval, model card, field/feature dictionary, lineage and runbook | Independent scientific/legal sign-off |
| Deployment | Pinned dependencies, nonroot containers, persistent volume, health checks, operational/recovery demo and CI | Provisioned host, TLS, monitoring, incident owners, recovery SLOs |

## Acceptance commands

```bash
ruff check .
mypy src/football_analytics
pytest
make demo-v2
make demo-v2-comparison
make demo-v2-nested
make demo-platform
# Container acceptance is also exercised by GitHub Actions:
docker compose up --build --detach --wait
```

Native research demos emit immutable JSON/manifests and Markdown diagnostics. The platform demo uses only temporary synthetic data and checks validation, ingestion, nested research, training, approval, feature storage, CLI/API prediction, ratings, intelligence, model card and backup restore. Legacy regression workflows remain in CI.

## Deliberate interface and scope decisions

Operational forecasts reference canonical stored model/match IDs. Elo is a rating engine and feature provider rather than an invented calibrated 1X2 output. Global and [competition-specific frequency baselines](competition_frequency_baseline.md) are implemented, including sparse-group shrinkage and unseen-group fallback. Historical confederation baselines still require effective-dated inputs. Ranking-only classifier experiments use the ranking feature family; missing observations are explicit. Match-probability intervals are null until validated. Additional manually supplied football context is timestamped, not inferred from hindsight.

These distinctions prevent placeholder outputs from being presented as completed research. See [commercial readiness](commercialization_notes.md) for the evidence needed to release the software commercially and [model card](model_card.md) for modeling limitations.
