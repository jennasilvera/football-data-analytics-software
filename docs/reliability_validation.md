# Reliability validation

Changes to storage, governance, forecasting, dependencies and recovery must be
reviewed as interacting components. A unit-level success is insufficient when the
behavior depends on another process, a write lock, a retry or a restored database.

## Required invariants and evidence

| Interaction | Required invariant | Regression coverage |
|---|---|---|
| Forecast publication + run audit | Both commit or neither commits | SQL failure injection, deadline rollback and process termination before commit |
| Commit + response loss + retry | Retry returns the committed winner without additional forecasts | Worker terminated after commit; retry before and after backup restore |
| Multiple workers + same run ID | One winner and one matching forecast set | Four independent spawned processes, synchronized before publication |
| Approval + concurrent revocation + publication | A captured approval cannot authorize publication after its revision changes | Revocation from a different process; rejection and restored revoked state |
| Active writer + online backup | Snapshot excludes the unfinished transaction | Backup taken while a worker is paused after inserting forecasts but before its audit |
| Review + lock acquisition | Publication order survives reversed request-start order | Delayed revocation and real lock-wait timestamp tests |
| Backup + filesystem failure/concurrent destination | No partial final file and no overwrite | Copy, integrity, sync, symlink and concurrent-publisher tests |
| API + unavailable SQLite | Structured failure, request correlation and successful recovery | Real write-lock contention, health/read errors and subsequent write success |
| SQLite capacity + atomic write | Exhaustion cannot retain a partial batch | SQLITE_FULL induced with a page limit; rollback and recovery after removing the limit |
| Backup + complete revision history | Structurally valid storage cannot conceal altered historical records | Superseded/future-visible payload tampering, malformed JSON and nonfinite values abort publication |
| Publication + unchanged IDs + altered content | Guarded inputs and duplicate records must still match their hashes | Tampering after snapshot selection blocks both write paths; a corrupt duplicate rolls back earlier inserts |
| Installed Python packages | Declared dependency requirements are mutually satisfied | python -m pip check after CI installs the pinned lockfile and editable package |

The process tests use SQLite's actual WAL transactions and independent spawned
Python processes. A test-only SQL trigger pauses publication inside the transaction;
it is removed after recovery. Fixtures isolate storage/governance identities and
do not claim to validate model fitting or predictive quality. Those components have
separate research and end-to-end operational acceptance tests.

    python -m pip check
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest tests/v2/test_process_reliability.py -q
    pytest

Tests use pipe handshakes and bounded waits instead of timing sleeps. Workers are
terminated and joined in cleanup, with kill as a fallback. CI jobs have a 15-minute
limit so a broken interaction cannot leave an indefinitely running job. All process
tests run in the normal suite.

## Scope of the evidence

These tests cover process termination, transaction rollback, publication arbitration,
governance revision consistency and backup/restore behavior on the test filesystem.
They do not simulate host power loss, kernel failure, filesystem corruption,
network filesystems, multi-host storage, clock regression or production traffic.

Production acceptance still needs an operator-selected environment and measured
targets: concurrency and fixture volume, p95/p99 latency, lock-timeout frequency,
physical disk exhaustion behavior (beyond the tested SQLite page limit), backup
duration, recovery point/time objectives and
off-host restoration. Run dependency/security review and representative load/fault
tests on that environment before claiming those requirements are met.

For each future cross-component change, identify the invariant, reproduce the
failure boundary, add a regression at that boundary, and run the affected end-to-end
acceptance path. Passing a dependency consistency check does not establish security
or behavioral compatibility; the API, dashboard, research and recovery workflows
remain necessary.

Pagination regressions hold a real SQLite write transaction open during the first
API page, then commit new historical entities and revisions before the next page.
All five collection routes must retain the original view; a fresh request sees the
commit. A 1,001-record scan tests writes between internal pages, and dashboard
navigation tests preserve both timestamp and committed sequence through navigation
and reset them on Refresh. Empty snapshots and invalid boundaries are covered.

Approval-read races are exercised for both single and batch forecasts: a separate
SQLite connection replaces the release after it is read for validation but before
publication. Both revocation and replacement approval must conflict against the
exact validated record. Single forecasts write nothing; batches retain a failed
audit with forecasts withheld and stable retry results.

Corruption classification is tested across model, metrics, approval, fixture, and
odds reads through the API, batch service, and CLI. Hash-mismatched records must
remain storage failures, return a correlated 503 or CLI exit 2, and leave no
forecast or terminal batch audit. These tests distinguish infrastructure failure
from a validly read but rejected domain input.

Request-boundary tests cover non-finite numeric values and overflow, malformed JSON,
unknown fields, exact/over-limit body sizes, and authentication precedence. A direct
ASGI stream without Content-Length crosses the body limit in three chunks and must
stop reading before dispatch, with a correlated 413 response. Validation responses
retain field/error metadata while omitting raw inputs and parser context.

Immutable output/model publication syncs the completed staging file before linking
the final name, then syncs its parent directory before returning success. Identical
retries also sync the directory. Fault injection covers file-sync failure (no final
file), directory-sync failure (complete file retained, error propagated), and retry
recovery; concurrent writers test identical and conflicting content with no partial
outputs or leftover staging files on normal completion. These are syscall ordering
and failure tests, not a power-loss or filesystem qualification test.

API reimplementation acceptance enumerates every route to verify canonical v1 and
unversioned alias authentication parity, unique canonical OpenAPI operations,
shared validation/body-limit/storage errors, and cross-prefix snapshot pagination.
The trained-model serving lifecycle runs through both prefixes; operational demos
and container checks exercise v1, while existing root-route regression coverage
protects compatibility. Dashboard navigation uses the versioned default base.

Data onboarding acceptance exercises valid and rejected results, duplicate fixtures,
unknown identities, empty batches, fixture-only schemas, and out-of-scope decisions.
Validation reports are stable across retries and leave operational records empty;
ingestion still refuses rejected batches. Unparseable rows exit 2 without a partial
report. Validation does not certify data rights or cross-import consistency.

Match-history acceptance covers superseded records, as-of visibility, backdated
commits after the first page, retained sequence values after backup, API
authentication on both route prefixes, CLI continuation, invalid sequence/UTC
parameters and hash verification of older revisions. Stable pages require both
timestamp and committed-sequence snapshot fields.

Dashboard match-history tests use Streamlit AppTest backed by the actual API and
SQLite store. They verify multi-page snapshots despite a later backdated write,
refresh visibility, independent match selection, historical cutoffs, invalid local
input, empty states, source metadata and storage-error presentation. Storage errors
must not appear as empty revision histories.

Operational forecast lineage tests cover single and batch publication with and
without odds, content identity including revision references, historical resolution
after revocation/fixture correction, CLI and both API prefixes, and backup recovery.
Missing or corrupt referenced approvals fail as storage errors; older forecasts
report missing lineage explicitly. Exact revision lookup enforces temporal visibility.
