# Native API and CLI reference

Start `football-analytics serve --database PATH`. FastAPI publishes the precise request/response schemas at `/openapi.json` and interactive documentation at `/docs`. Version 0.3.0 uses API schema 1 and regulation-time targets.

Business routes require `Authorization: Bearer <FOOTBALL_API_KEY>`. The key must have at least 24 characters. Only explicitly configured local development can disable authentication. The service uses one operator key; it does not implement per-user authorization or tenant isolation.

| Method/path | Result |
|---|---|
| GET /health | SQLite health; 503 on failed check |
| GET /version | Software/API version, scope and target |
| GET /teams; /matches; /predictions; /models | Paginated collection |
| GET /teams/{team_id}; /matches/{match_id} | Canonical stored object |
| GET /teams/{team_id}/ratings | As-of replayed Glicko history |
| GET /teams/{team_id}/report; /reports/team/{team_id} | Rating, recent form, schedule and residual intelligence |
| GET /ratings; /ratings/movers | Current Glicko state; largest absolute last-period movements |
| GET /models/{model_id}/metrics | Linked research report |
| GET /reports/match/{match_id} | Fixture/result, stored forecasts and post-match diagnostics |
| GET /diagnostics/upsets | Latest pre-match forecast residuals, sorted by surprisal |
| GET /diagnostics/regression-watchlist | Positive-residual descriptive watchlist |
| GET /diagnostics/breakout-watchlist | Negative-residual descriptive watchlist |
| POST /predict | Approved stored-model forecast for a stored scheduled fixture |

POST body:

```json
{"model_id":"forecast_bundle_<digest>","match_id":"canonical-match-id","method":"ensemble"}
```

Use the actual IDs printed by ingestion/training. Method can be a bundle's base spec ID, `calibrated_<spec_id>` or `ensemble` when present. The server supplies prediction time and enforces pre-match eligibility, approval, model age, known identities and artifact integrity. Maximum body size is 16 KiB. Success includes `X-Request-ID`; structured request logs omit credentials and bodies.

Collections accept `limit` (1–1000), `after` (previous `next_cursor`) and `as_of` (previous response's UTC timestamp). Retain the same `as_of` across pages to obtain a consistent historical view. Missing objects return 404, missing credentials 401, invalid requests/contracts 422, unapproved/stale serving candidates 409 and oversized bodies 413. Model lists expose metadata rather than full training bundles.

## CLI operations

All operational commands accept `--database`; default comes from `FOOTBALL_DATABASE` or `outputs/platform.sqlite3`. Commands that emit files use immutable publication and refuse conflicting output replacement.

| Command | Purpose |
|---|---|
| ingest-results / ingest-fixtures | Validate canonical CSV and atomically record catalogs/matches |
| ingest-rankings / ingest-odds | Validate timestamped JSON observation arrays |
| validate-data | Validate source and catalogs without importing records |
| build-features | Materialize and persist as-of form features for a stored fixture |
| train | Fit all native model families and optional transforms into a candidate bundle |
| research / backtest | Temporal model evaluation; `backtest` is an alias |
| calibrate | Research alias requiring nested holdout options |
| import-research / approve-model | Validate research consistency and serving-family coverage; bind human review to exact revisions |
| revoke-model | Append a revocation with required review reason; stop serving |
| predict | Forecast stored model/fixture, enforcing approval and freshness |
| update-ratings | Replay and persist current Glicko states |
| team-report / match-report / model-card | Auditable JSON reports using canonical IDs |
| drift-report | Compare JSON arrays of materialized features from two samples |
| backup | Online SQLite backup and integrity check |
| serve / dashboard | Native API and Streamlit interface |

`train --historical-availability-assumed` explicitly acknowledges flat-file historical availability. `--uncalibrated` creates raw models only, so select an actual base method at prediction time. `--feature-groups` and `--feature-context` declare research/training inputs. Prediction uses the bundle's frozen context; it does not silently substitute newly ingested features.

The implementation uses canonical `--model-id`, `--match-id` and `--team` IDs rather than ambiguous free-form team-name arguments. Elo is a replayable rating engine and strength feature family; it is not falsely exposed as a standalone calibrated 1X2 model.


## Review lifecycle

Import schema-5 research generated for the same source, catalogs, feature groups and
frozen context. Each evaluated run must reproduce its fold/aggregate metrics and
calibration from its predictions and match its manifest lineage. Research must cover
the bundle's base models/features and any calibration/ensemble families.

`approve-model --model-id ID --reason TEXT` binds the decision to the exact model and
metrics repository record IDs. Importing any new research revision atomically sets
status back to `candidate`, even when the report contents are identical. Legacy
approvals without revision bindings require a new review. Replaced evidence and
revoked approvals return HTTP 409 from prediction and fail CLI prediction.
`revoke-model --model-id ID --reason TEXT` records a revocation without deleting audit
history. A later explicit review can approve the model again.

These checks establish artifact consistency and declared coverage. A reviewer must
still verify source truth, hyperparameter suitability, independent holdout evidence,
source licensing and fitness for the intended use. No metric threshold auto-promotes.

Dashboard collection views provide Previous, Next and Refresh controls. Navigation
retains the API snapshot timestamp; Refresh starts a new snapshot. Counts are per-page,
not total inventory counts.
