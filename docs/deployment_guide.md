# Deployment and recovery runbook

## Local or container deployment

Install with `requirements.lock` and `pip install -e . --no-deps` on Python 3.12. The exact dependency graph is pinned for repeatable CI/container installation; this is not a platform-independent hash lock. Rebuild and run CI when updating dependencies.

Generate a unique API key with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`. Put it in a local `.env` based on `.env.example`, protect the file and do not commit it. Then:

```bash
docker compose up --build --detach --wait
curl --fail http://127.0.0.1:8000/health
# Open http://127.0.0.1:8501 and enter the API key.
```

Compose binds both services to loopback, runs UID 10001 with dropped capabilities, a read-only root filesystem and temporary writable mounts. SQLite lives on the `football-data` named volume. Inspect `docker compose config` before operating. Public exposure needs a separately managed TLS reverse proxy, identity policy and rate limiting. Repository CI builds and checks the containers; it does not publish a live endpoint.

| Variable | Default / meaning |
|---|---|
| FOOTBALL_DATABASE | outputs/platform.sqlite3 locally; /data/platform.sqlite3 in API container |
| FOOTBALL_API_KEY | Required secret, minimum 24 characters |
| FOOTBALL_MAX_MODEL_AGE_DAYS | 90, positive integer |
| FOOTBALL_DEV_NO_AUTH | 0; 1 explicitly allows local unauthenticated mode |
| FOOTBALL_API_URL | http://127.0.0.1:8000 for dashboard; http://api:8000 in compose |

## Data and model refresh

1. Confirm provider permissions, official senior-men's scope, canonical aliases and regulation-time scores. Maintain source archives and revision/publication evidence.
2. Run `validate-data`, then `ingest-results` and `ingest-fixtures` with explicit source/legal notes. Ranking/odds JSON imports require publication times; odds require `score_basis: regulation_time`.
3. Run predeclared `research`/`backtest` folds (including recent held-out seasons and competitions). Review coverage, calibration, comparisons, ablation and market evidence.
4. Run `train` with a UTC cutoff and explicit flat-file availability acknowledgment. Save the printed model ID. The bundle freezes history and optional feature context.
5. `import-research --model-id ID --input REPORT` links source-matched evidence. Check families/features/folds manually, then `approve-model --model-id ID --reason 'review evidence...'`.
6. Forecast a stored future fixture using `predict --model-id ID --match-id ID`. The default ensemble requires calibrated training. New imports do not update old bundles automatically.
7. After results arrive, replay ratings and inspect reports. Evaluate observed coverage and losses before approving replacement models.

Each command accepts `--database PATH`. Use `--help` for source/catalog arguments. Jobs should run in a scheduler under the operator's ownership, with failures routed to an agreed alert channel. Do not overlap arbitrary refresh jobs or auto-promote candidates solely on one metric.

## Backup and recovery

```bash
football-analytics backup --database outputs/platform.sqlite3 --output backups/platform-YYYYMMDD.sqlite3
```

Use a new destination. The command uses SQLite's online backup API (including committed WAL state) and checks database integrity. Copy the resulting standalone snapshot to encrypted off-host storage with appropriate retention. A local snapshot alone is not disaster recovery.

To restore: stop writers and API, retain the failed database and its WAL/SHM files for investigation, copy the verified snapshot to a **new database path**, set `FOOTBALL_DATABASE` or `--database` to that path, and restart. Check `/health`, counts, one model and representative forecasts against the backup. Never overwrite an open database or leave WAL files associated with a different database. `make demo-platform` verifies record equality after restoring a snapshot into a separate repository; periodically rehearse the same process with representative production volumes.

## Operations and incidents

Collect JSON request logs, status/error rates, latency, disk/WAL size, source age, model age, missingness, drift and observed forecast losses. No external alert integration or numerical SLO is presumed. Document owners, RPO/RTO, retention and capacity after measurement. Health checks test storage rather than predictive quality.

Missing approval/metrics, stale history and pre-match timing violations fail closed. Repair upstream source/catalog problems rather than bypassing the temporal or score policy. If storage/artifact integrity fails, stop serving affected data, preserve evidence and restore a verified snapshot. Rotate a compromised API key, restart both clients/services, and inspect access logs. SQLite is a single-host storage choice; benchmark before introducing multiple writers or larger deployments.
