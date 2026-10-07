# Implementation status against the product requirements

This map tracks the supplied Football Data Analytics Software requirements as
of the native evaluation diagnostics milestone. A legacy implementation does not
imply that the corresponding native V2 capability is complete. The intended
scope remains official senior men's A-international football only.

| Product area | Native V2 status | Next acceptance gate |
|---|---|---|
| Ingestion, validation, entity resolution | Canonical catalogs, tabular/legacy adapters, scope assessment, quarantine, snapshot hashes, explicit score basis and regulation-only targets | Governed real-source adapters; separate period tallies and revisions; source freshness |
| Temporal integrity | Provenance, cutoff-safe features, target availability and chronological folds | Historical revisions and actual publication-time datasets |
| Team ratings | Elo migration/parity and snapshots | Glicko-style uncertainty, rating comparison and persisted history services |
| Features | Form, rest, rankings, competition, travel, squad availability, market snapshots | Manager/manual intelligence, opponent adjustment, coverage monitoring and persistent feature store |
| Models | Class-frequency, logistic, histogram boosting, independent Poisson, nested temperature scaling and convex ensemble | Shrinkage/uncertainty research; real-data calibration validation |
| Model artifacts | Immutable portable Poisson/transform JSON plus experiment manifests | Versioned classifier artifacts, promotion metadata and environment compatibility |
| Evaluation | Expanding/rolling backtests, log loss/Brier/RPS, calibration diagnostics, paired comparison, competition/team/venue/year slices | Historical confederation slices, ablation, benchmark joins, interval coverage and drift |
| Match forecasts | Poisson replay with expected goals, grid, mode, entropy and tail mass | Unified multi-model service; explanatory drivers; calibrated intervals and freshness reports |
| Reports | Auditable research JSON, model comparison Markdown, Poisson model card | Team/competition intelligence and complete field/feature dictionary |
| Post-match diagnostics | Native per-match probability losses, entropy and largest-loss reports | Feature attribution, rating-movement and trend analysis |
| CLI | Research, all-model comparison and Poisson replay; installed `football-analytics` entry point | Modular ingestion, ratings, training, unified prediction and reporting commands |
| API | Legacy FastAPI only | Native versioned routers backed by application services and repositories |
| Dashboard | Legacy only | Native research dashboard consuming governed artifacts |
| Deployment | CI, static typing, regression and end-to-end demos | Pinned deployment environment, containers, service configuration, logging, operational persistence and health checks |
| Commercial readiness | Architecture and limitations documented | Licensed data, authentication, operational SLOs, backup/recovery and support processes |

## Current commands and expected outputs

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
make demo-v2
make demo-v2-comparison
python -m football_analytics research --help
football-analytics predict-poisson --help
ruff check .
mypy src/football_analytics
pytest
```

The comparison demo evaluates class frequency, logistic regression, histogram
gradient boosting and Poisson on the same folds. It writes research JSON,
comparison Markdown, four experiment manifests, and one Poisson model artifact
per valid fold under `outputs/v2-research/`. The sample has 20 evaluated matches
across two folds. Scores are demonstration outputs, not research conclusions.

Use an emitted Poisson artifact with canonical IDs to replay inference:

```bash
football-analytics predict-poisson \
  --model-artifact outputs/v2-research/models/poisson_model_<digest>.json \
  --match-id example-arg-bra --home-id ARG --away-id BRA \
  --competition-id friendly --match-date 2020-07-01 \
  --prediction-time 2020-06-30T12:00:00Z --venue neutral
```

The digest placeholder must be replaced with a path printed by the research
command. The fixture is illustrative, not a historical claim. Date-only match
cutoffs, canonical IDs, training cutoff and known team histories are enforced.

## Planned implementation sequence

1. **Source governance.** Explicit score-basis contracts now reject unknown and
   non-regulation targets. Add revision contracts; verify real sources against scope, license and timestamp
   requirements. Test mixed score bases, late revisions and exact publication times.
2. **Calibration validation.** Nested holdout fitting and leakage tests are implemented.
   Validate transfer to refitted models on governed real sources, assess sample
   requirements and subgroup reliability, and reserve an untouched final test period.
3. **Research diagnostics.** Slice metrics and per-match forecast losses are implemented.
   Add market benchmark joins, ablations, historical confederation metadata and
   paired uncertainty with an explicit dependence/resampling policy.
4. **Ratings and intelligence.** Add uncertainty-aware ratings, schedule-adjusted
   trends and team reports; validate sparse-team behavior and rating replay.
5. **Unified inference and storage.** Add classifier artifact support and native
   forecast repositories, then application services for match/team outputs.
6. **API and dashboard.** Implement the required endpoint families over those
   services, followed by the dashboard. Test API contracts and historical replay.
7. **Deployment.** Add container/environment locking, configuration/logging,
   operational persistence, authentication and recovery tests. Document actual
   operating limits before describing the system as commercially ready.

Each step is a tested implementation milestone, not a directory of placeholders.
The existing migration plan describes package boundaries and the legacy
retain/adapt/rewrite decisions. The native workflow and model card document the
assumptions of functionality available today.

## Common failures and intended behavior

| Failure | Action |
|---|---|
| Unresolved or excluded rows | Review rejection CSVs and fix source/catalog governance; no partial-sample training |
| Unseen team in a Poisson fold | Collect adequate earlier history or predeclare an eligible cohort; no automatic fallback or selective row drop |
| Class missing from classifier training | Move the predeclared training window or use the frequency benchmark; do not borrow later labels |
| Overlapping evaluation windows | Use disjoint windows so pooled matches are counted once |
| No valid temporal folds | Inspect training minima and available-target dates; the run fails explicitly |
| Model artifact identity conflict | Investigate modified/corrupt files; the immutable writer refuses overwrite |
| Model comparisons use different samples | Re-run against the same declared source and folds; no silently intersected sample |

Real-data performance, commercial readiness and completed native parity must each
be demonstrated separately. None follows from the size of the codebase or the
synthetic test suite.
