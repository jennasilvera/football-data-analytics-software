# Native daily forecasting runbook

This workflow operates the `football_analytics` platform for official senior men's
national-team matches. The retained World Cup-specific commands are documented in
[the legacy runbook](legacy_world_cup_runbook.md).

## Before the daily run

1. Validate and ingest results, fixtures and any licensed odds using the canonical
   catalogs. Archive the source releases and their publication/revision evidence.
2. Select a reviewed model bundle. Refresh its training snapshot through governed
   research and approval when needed; new result imports do not update frozen bundles.
3. Choose an explicit inclusive UTC fixture-date window and a unique run ID. The
   same run ID belongs permanently to one model/method/window/freshness request.

The default ensemble requires a calibrated bundle. Use a supported base method for
an uncalibrated bundle. All commands accept `--database PATH`.

```bash
football-analytics predict-batch \
  --database outputs/platform.sqlite3 \
  --model-id forecast_bundle_REPLACE_WITH_REVIEWED_ID \
  --run-id daily-2026-10-13-am \
  --from-date 2026-10-13 --through-date 2026-10-16 \
  --output outputs/daily-2026-10-13-am.json

football-analytics batch-report \
  --database outputs/platform.sqlite3 --run-id daily-2026-10-13-am
```

Use actual dates and the registered model ID. The CLI supplies current UTC time;
it does not offer an operational backdating option. Run IDs allow letters, digits,
dots, colons, hyphens and underscores, up to 128 characters.

## Selection and publication

The run resolves the reviewed model, evidence, fixtures and eligible odds at one
start timestamp. The model must meet the configured freshness limit (90 days by
default, adjustable with `--max-model-age-days`). Exact model, evidence, approval
and match repository record IDs are retained in the audit; market record IDs remain
in each enriched forecast.

Within the date window, completed, postponed and cancelled fixtures are recorded as
skipped. Scheduled fixtures at/past exact kickoff are skipped; date-only fixtures
must still be before their UTC match day. Fixtures outside the window or not yet
known at the snapshot are not part of this run. A window exceeding 1000 stored
fixtures fails explicitly; reduce the window instead of truncating the population.

All eligible fixtures are computed before publication. A single invalid fixture or
forecast failure withholds the other computed forecasts. The database commits either
all forecasts and a completed run audit, or a failed audit with no batch forecasts.
Pre-match deadlines are checked after computation and inside the write transaction,
including after acquiring the write lock and immediately before commit. If the run
misses a deadline, its forecast writes are rolled back and a failure audit is retained.

The snapshot timestamp describes forecast inputs. The completion timestamp and
repository availability describe the computed output; no output is recorded as
available at the earlier input snapshot. Commit visibility is governed by SQLite's
transaction, not a guarantee of subsecond distribution to external consumers.

## Review the result

| Field/state | Meaning |
|---|---|
| completed | Forecasts and audit committed together; may include zero eligible fixtures |
| failed | Audit retained; no forecasts from this batch committed |
| forecasted | Published forecast ID for an eligible fixture |
| skipped | Non-scheduled or already closed pre-match window |
| failed fixture | Its data, model evaluation or publication window failed |
| withheld | Computed candidate discarded because the batch failed |
| errors | Run-level review, selection or publication failure details |

`predict-batch` exits 0 for completed runs (including empty runs), 1 for durable failed
runs, and 2 for invalid requests, database errors or output-file failures. A failed run
still prints its JSON audit. A failure writing the optional output file may happen
after database commit: inspect `batch-report` or retry the same request before
assuming the run did not publish.

The authenticated API exposes paginated `GET /forecast-batches` and
`GET /forecast-batches/{run_id}`. The dashboard's **Forecast batches** page shows
counts, fixture outcomes, request parameters and review lineage. These are read-only
views; batch execution remains an operator CLI operation.

## Retries, concurrency and recovery

Repeat the exact command and run ID after a lost response. A committed run returns
its original audit without fitting or predicting again. This also applies after
source updates, revocation or kickoff: a retry retrieves historical output, not a
new forecast. Reusing the ID with different request parameters is rejected.

Failed runs are immutable too. Repair the upstream issue and use a **new run ID**
for a fresh attempt and input snapshot. If a process dies before its transaction
commits, no partial forecasts or run guard remain; retrying can compute a new snapshot.
If it dies after commit, the stored guard makes the retry return the winning result.
Concurrent workers using the same ID can compute redundantly, but a serialized
transaction permits only one publication and its related forecast set.

Storage errors can prevent a failure audit itself from being written. Do not treat
missing audit records as successful runs. Route CLI nonzero exits to an
operator-configured alert channel and inspect storage health. Use the online backup
command and [recovery procedure](deployment_guide.md); backups retain run guards,
audits and linked forecasts, so completed-run retries survive restoration.

No cron job, paid feed or external alert recipient is installed automatically. Select
an operator-owned scheduler, prevent arbitrary ingestion/review overlap and measure
batch duration/capacity against real match windows before production use.
