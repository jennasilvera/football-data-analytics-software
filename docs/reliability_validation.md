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
