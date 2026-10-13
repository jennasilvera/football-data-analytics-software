# Implementation status against the product requirements

Native platform release 0.3.0 completes an executable implementation of the professional upgrade sequence. This map distinguishes software delivery from empirical and operational acceptance. Core scope is official senior men's A-international football with regulation-time targets.

| Product module | Native implementation | Remaining acceptance boundary |
|---|---|---|
| Ingestion, validation, entity resolution | Canonical catalogs, aliases, source hashes, tabular/results/fixtures/rankings/odds imports, scope/duplicate checks and atomic batches | Licensed provider integrations, real coverage and source archives |
| Data and revision integrity | UTC provenance, leakage controls, append-only bitemporal SQLite revisions and as-of reads | Unknown historical revisions cannot be reconstructed |
| Ratings | Elo replay/parity, Glicko-1 periods, deviation/inactivity and rating history | Football-specific tuning and uncertainty coverage |
| Features | Form 5/10/20, strength, schedule, competition, rankings, adjacent-publication ranking trends/buckets, squad, travel, market, manual context; status/lineage; persisted form vectors | Real exogenous input availability; specialized research signals |
| Forecasting/calibration | Frequency, logistic, histogram boosting, legacy-parity and joint opponent-adjusted Poisson, held-out temperature transforms and convex ensemble; portable JSON replay | Real-data performance, validated match-probability intervals |
| Research evaluation | Expanding/rolling folds, proper scores, reliability, paired comparison, team/competition/year/venue slices, publication-aware historical confederation slices, family ablation, block loss intervals, prediction/opening/closing market joins and paired bookmaker movements, drift | Verified membership publication archives and untouched real holdout |
| Diagnostics/intelligence | Actual-result surprisal, residuals, miss classification, Glicko movements, form/schedule reports, descriptive watchlists and next stored forecast | Causal explanations and validated future trend signals |
| API | All requested endpoint families, authentication, size limits, stable collection pagination, structured logs | Tenant identity, internet edge policy and measured capacity |
| CLI | Ingestion through candidate training/review/prediction, atomic forecast batches with durable retries, ratings, reports, drift and backup | Provider-specific scheduled job ownership |
| Dashboard | API-backed overview, ratings, forecasts, team/model diagnostics and post-match views, snapshot-preserving collection navigation and batch audits | User research and production-scale interaction testing |
| Governance/docs | Experiment/model identity, source/catalog/context-matched research with metric verification, exact-revision approval/revocation, model card, field/feature dictionary, lineage and runbook | Independent scientific/legal sign-off |
| Deployment | Pinned dependencies, nonroot containers, persistent volume, health checks, operational/recovery demo and CI | Provisioned host, TLS, monitoring, incident owners, recovery SLOs |

## Acceptance commands

```bash
ruff check .
mypy src/football_analytics
pytest
make demo-v2
make demo-v2-comparison
make demo-v2-nested
make demo-v2-opponent-poisson
make demo-platform
# Container acceptance is also exercised by GitHub Actions:
docker compose up --build --detach --wait
```

Native research demos emit immutable JSON/manifests and Markdown diagnostics. The platform demo uses only temporary synthetic data and checks validation, ingestion, nested research, training, evidence replacement/re-review/revocation, feature storage, CLI/API prediction, ratings, intelligence, model card and backup restore. Legacy regression workflows remain in CI.

## Deliberate interface and scope decisions

Operational forecasts reference canonical stored model/match IDs. Elo is a rating engine and feature provider rather than an invented calibrated 1X2 output. Global and [competition-specific frequency baselines](competition_frequency_baseline.md) are implemented, including sparse-group shrinkage and unseen-group fallback. The historical confederation-pair baseline is also implemented with explicit publication-governed membership inputs and global fallback. Verified real membership archives remain required. The [opponent-adjusted Poisson challenger](opponent_adjusted_poisson.md) jointly fits attack/defence and home effects in each chronological fold; it is opt-in research and does not replace serving defaults. Ranking-only classifier experiments use the ranking feature family; missing observations are explicit. Match-probability intervals are null until validated. Additional manually supplied football context is timestamped, not inferred from hindsight.

These distinctions prevent placeholder outputs from being presented as completed research. See [commercial readiness](commercialization_notes.md) for the evidence needed to release the software commercially and [model card](model_card.md) for modeling limitations.


## Release acceptance hardening

Research import and approval fail closed when primary run metrics/calibration cannot
be reproduced or manifest/feature/model coverage is inconsistent. Re-import returns
a model to candidate status atomically; serving checks the reviewed evidence and
model record IDs at the request timestamp. Audit history survives revocation and backup.
Single-match API/CLI serving and batch publication verify reviewed model, evidence,
approval and fixture revisions inside the write transaction, withholding publication
if they changed during computation or lock acquisition. Approval and research import
likewise reject concurrent changes to model, evidence or release state, including
creation of previously absent records. Review recording times are assigned after the
write lock is acquired, preserving the publication order of delayed revocations.
Verified backups are published atomically without replacing existing files.

Tests cover tampered predictions/metrics/lineage, replaced evidence, concurrent
reviews, delayed revocations, forecast deadlines, durable retries, backup failure
recovery and UI cursor navigation. Auxiliary narrative/diagnostic outputs
are not independent proof of the primary research sample.

This completes the native software acceptance pass. Empirical validation and a live
commercial deployment remain separate projects requiring real licensed data,
independent evaluation and operator-selected infrastructure. The table above retains
those boundaries explicitly; synthetic acceptance does not close them.


## Completion and handoff

The native implementation and its repository-contained acceptance work are complete.
The CI workflow is the release gate: lint, native static types, the complete test
suite, all research demonstrations, operational governance/forecast/recovery
acceptance, the retained legacy workflow and container smoke checks must pass on
the exact proposed commit before merge.

The remaining work requires external inputs and evidence:

| Next gate | Required input | Acceptance evidence |
|---|---|---|
| Real data onboarding | Selected provider, access and permitted uses; publication/revision archives | Coverage, identity, timing and regulation-score audit |
| Empirical validation | Frozen real training/evaluation data and untouched holdout | Reproduced scores, calibration and subgroup results; justified model selection |
| Live deployment | Chosen host and identity system, TLS, secret lifecycle and operators | Capacity/security review, monitoring, incident ownership and off-host recovery drill |
| Commercial release | Data redistribution/training rights and product/support decisions | Independent scientific and legal sign-off |

These gates are not replaced by adding more synthetic tests or optional model
families. Start the next phase by selecting the real data source and intended
deployment environment; no provider subscription, public deployment or production
model approval is implied by completion of the software implementation.
