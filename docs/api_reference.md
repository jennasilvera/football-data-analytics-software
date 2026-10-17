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

Publication guards retain the exact approval record that passed validation. They
never reread a newer release merely to choose the guard identity: a revocation or
replacement approval committed after validation causes a publication conflict.
Single forecasts return 409; batches record a failed audit and withhold forecasts.

Stored content corruption is a database failure (`RepositoryIntegrityError`, a
`sqlite3.DatabaseError`), not a request validation error or approval conflict. This
classification is preserved through model review, fixture reads, and odds
enrichment: the API returns a correlated generic 503, and the CLI exits 2. Batch
processing aborts without writing forecasts or a terminal batch audit, allowing
the same run ID to be retried after storage is recovered. Ordinary domain failures
continue to produce durable failed-batch audits.

Request validation errors return 422 with `detail` entries containing `loc`, `msg`,
and `type`; raw `input` and `ctx` are omitted. This keeps invalid non-finite numeric
inputs JSON-safe and avoids echoing submitted values. POST bodies are limited to
16,384 bytes, including streamed chunks without Content-Length. Oversized requests
return 413 with the same X-Request-ID header and completion log as other responses.
The buffer checks each chunk before appending it and stops reading when over limit.
Ingress timeouts, network buffering, and connection cleanup remain deployment concerns.

Immutable CLI output and model files sync both file bytes and the containing
directory before acknowledging publication. If directory sync fails after linking,
the command reports failure while a complete final file may remain. Retry the
same bytes at the same path to recheck content and sync the directory; differing
bytes remain a conflict. This assumes a provisioned durable output directory and
a filesystem supporting hard links and directory fsync. Newly created ancestor
directories and machine/storage loss require deployment-level durability controls.

## API v1 and compatibility

The canonical HTTP base is `http://127.0.0.1:8000/api/v1`; endpoint paths elsewhere
in this reference are relative to that base. For example, use `/api/v1/teams` and
`/api/v1/predict`. `/api/v1/health` and `/api/v1/version` are public. `/docs` and
`/openapi.json` remain at the application root and document the versioned contract.
The `api_schema` field is 1; this routing change preserves payload schemas.

Every original unversioned URL remains an alias to the same router and services,
with identical authentication, request limits, exception handling and database.
Existing clients and health probes can keep their URLs. Pagination snapshot fields
can be carried between the canonical routes and compatibility aliases. No alias
removal date is declared. New clients should use `/api/v1`.

The dashboard defaults to the v1 base, and Compose sets `FOOTBALL_API_URL` to
`http://api:8000/api/v1`. An explicit `FOOTBALL_API_URL` is treated as a complete
base URL, so existing deployments configured with an unversioned base continue to
use the compatibility routes. Middleware/lifecycle stay in `api/main.py`; route
registration and request contracts live in `api/routes.py`.

## Data onboarding validation

`validate-data` emits a deterministic JSON canonicalization report, including when
rows are rejected. Use `--data-kind results` (default) or `--data-kind fixtures`;
`--results` names the input CSV in either case. Source, catalog and scope declarations
are the same as ingestion. `--output` optionally publishes the immutable report.

Reports contain source SHA-256, catalog identity, report identity, counts, normalized
match identities, and excluded/quarantined row identities with reasons. An empty
batch has `valid: false` and `batch_errors: ["empty_batch"]`. Exit codes are 0 for
valid input, 1 for an emitted invalid report, and 2 for parsing, catalog, storage or
output-publication failure. Validation does not ingest records. Ingestion still
refuses batches with any excluded/quarantined observations atomically.

This report covers canonicalization of the supplied batch. It does not certify
cross-import uniqueness, historical publication availability, licensing, predictive
quality or serving eligibility. Malformed source schemas or values that cannot form
observations fail before a report can be produced. Report IDs do not include a run
time, so repeating validation with identical data, catalogs and policy is stable.

## Match revision history

Authenticated `GET /api/v1/matches/{match_id}/history` returns revision envelopes
with `record_id`, `sequence`, `available_at`, `recorded_at`, and the original stored
`payload`, including source provenance when present. This inspects observed storage
history rather than reconstructing uncollected provider revisions. The unversioned
compatibility route is also available.

Supply `as_of` to inspect a historical UTC cutoff; both availability and recording
times must be eligible. The response includes `as_of`, `through_sequence`, `limit`,
`truncated`, and `next_cursor`. Results are ordered by ascending commit sequence,
which can differ from timestamp order for backdated imports. To continue, pass
`next_cursor` as `after_sequence` together with both snapshot fields. Continuation
without the snapshot fields is rejected. Limits are 1–1,000 (default 100). A match
with no visible revisions returns an empty page. Old revisions are content-verified;
corruption returns 503 even when the latest revision is intact.

The CLI exposes the same service and snapshot fields:

```bash
football-analytics match-history --database outputs/platform.sqlite3 \
  --match-id canonical-match-id --limit 100
# Continue with values returned by the first call:
football-analytics match-history --database outputs/platform.sqlite3 \
  --match-id canonical-match-id --as-of 2026-10-15T20:00:00+00:00 \
  --through-sequence 120 --after-sequence 40
```

CLI `--output` uses immutable publication; invalid parameters exit 2. Snapshot
sequences belong to the same database history and should be discarded after a
replacement or restore. A backup preserving that history retains sequence values.

The dashboard exposes this endpoint in the **Match history** workspace. Select a
match from the paginated collection, optionally enter an ISO cutoff with timezone,
and inspect a revision with its source metadata. History pages contain up to 25
revisions. Previous/Next retain both snapshot fields; Refresh starts a new snapshot
for the selected match and cutoff. Match selection and cutoff values have separate
navigation state. Invalid cutoff input is rejected locally before requesting history.

## Forecast input lineage

New operational single and batch forecasts include `operational_lineage` with the
exact model, research evidence, approval, and fixture record IDs guarded at
publication. This field contributes to `forecast_id`; pure research forecasts
remain independent of repository records. Existing stored forecasts are unchanged.

Authenticated `GET /api/v1/predictions/{forecast_id}/lineage` (also available via
the compatibility route) resolves those exact revisions, including superseded
ones. It returns compact record identities and availability/recording timestamps;
selected odds record references are included when market enrichment is present.
It does not return full training bundles. Use `football-analytics forecast-lineage
--database outputs/platform.sqlite3 --forecast-id <id>` for the same report.

`status: resolved` means referenced records exist and pass content verification;
it does not mean the model is currently approved or scientifically validated.
Revocations and later fixture changes do not rewrite historical lineage. Missing,
corrupt, or incorrectly typed references fail as storage integrity errors (API 503,
CLI exit 2). Older/research forecasts without operational references return
`status: unavailable` and `reason: legacy_or_research_forecast`; no references are
inferred retrospectively. Backup preserves historical references and their records.

In the dashboard **Match forecasts** workspace, select a forecast and choose
**Inspect forecast inputs** to resolve its references on demand. The resulting
revision table can be downloaded as JSON. Legacy/research forecasts display an
explicit unavailable message. Each inspection requests the current audit from the
API; prior successful results are not reused after a failed request or after
selecting another forecast. Historical resolution does not assert current approval.
