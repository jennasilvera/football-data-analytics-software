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
| GET /forecast-batches; /forecast-batches/{run_id} | Paginated durable batch audits or one run |
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

Collections accept `limit` (1–1000), `after` (previous `next_cursor`), `as_of` (previous response's UTC timestamp), and `through_sequence`. Retain both snapshot fields across pages for a consistent committed historical view. Missing objects return 404, missing credentials 401, invalid requests/contracts 422, unapproved/stale serving candidates or publication conflicts 409 and oversized bodies 413. Model lists expose metadata rather than full training bundles.

SQLite failures and stored-content integrity failures during requests return 503 with a generic storage-unavailable
message and an X-Request-ID. Server logs correlate that ID with the SQLite error
code and exception type, without including the database exception text. The
response does not confirm whether a write committed; inspect stored results before
retrying a write. There is no automatic retry or promised retry interval. Storage
failure during application startup prevents startup rather than serving a healthy
response. A failed storage health probe also returns 503.

Single-match API and CLI serving resolve model, review, fixture and market inputs at
the request snapshot. Before publication, the write transaction checks the exact
model/evidence/approval/fixture revisions and the fixture's pre-match deadline.
Changed inputs or a deadline crossed during computation or lock acquisition publish
no forecast (API 409; CLI exit 2). Successful forecasts become available at computation
completion, not retroactively at the input snapshot. Single-match requests have no
durable retry key; use `predict-batch` when an immutable run audit is required.

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
| predict-batch / batch-report | Atomically forecast an inclusive UTC date window; inspect durable retry audit |
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

Approval and research import also compare the exact model, evidence and release
revisions inside the write transaction. An intervening revocation, re-review,
model replacement or evidence import rejects the stale operation without writing
a decision or partial evidence. Absence is checked too: two first-time imports
cannot silently replace each other. A conflict exits 2; inspect the latest records
and rerun the review or import deliberately. Revocation remains an unconditional
operator action and is not blocked by an evidence change. Review/import recording
timestamps are assigned after acquiring the write lock. A delayed revocation
therefore sorts after an approval that published first, even when the revocation
request started earlier. The declared review/input timestamp is retained separately;
as-of queries cannot see the decision before it was recorded.

These checks establish artifact consistency and declared coverage. A reviewer must
still verify source truth, hyperparameter suitability, independent holdout evidence,
source licensing and fitness for the intended use. No metric threshold auto-promotes.

Dashboard collection views provide Previous, Next and Refresh controls. Navigation
retains the API snapshot timestamp and committed sequence; Refresh starts a new snapshot. Counts are per-page,
not total inventory counts.

Batch operation and exit/retry semantics are covered in the [native daily runbook](daily_forecast_runbook.md).


Publication verifies the content behind guarded revision IDs while holding the
write lock. Duplicate writes also verify the existing content before treating the
write as idempotent. A malformed or mismatched stored payload is a storage-integrity
failure, even if its stored record ID has not changed. The transaction rolls back;
the source record is preserved for investigation rather than silently repaired.

### Stable collection pagination

Teams, matches, predictions, forecast batches, and models return `as_of` and
`through_sequence`. Preserve both values together with `next_cursor` as `after`
when navigating forward or backward. The timestamp controls temporal visibility;
the sequence excludes commits made after the first page, including historical
imports and revisions. Supplying a sequence requires an explicit timestamp and a
committed boundary in the current database. Refresh by omitting both values.
Timestamp-only clients remain supported but do not preserve a committed snapshot.
Tokens apply to the same append-only database history; discard them after database
replacement or restore. This does not provide snapshots across separate collections
or protection against direct database edits. Repository scans freeze the sequence
across their internal pages, and the dashboard retains both fields until Refresh.
