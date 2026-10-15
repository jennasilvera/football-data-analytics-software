"""SQLite append-only, bitemporal JSON records and atomic backup."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from builtins import list as List
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_analytics.data.contracts import ensure_utc
from football_analytics.models.postprocessing import content_id


class RepositoryIntegrityError(sqlite3.DatabaseError):
    """A stored record cannot be decoded or no longer matches its content identity."""


class PublicationDeadlineError(ValueError):
    """A time-sensitive transaction missed its final publication deadline."""


class PublicationConflictError(ValueError):
    """A reviewed input changed before publication acquired its write lock."""


class Repository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
            ).fetchone()
            if exists and db.execute("SELECT version FROM schema_version").fetchall() != [(1,)]:
                raise ValueError("Unsupported repository schema version.")
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, entity_id TEXT NOT NULL,
                    record_id TEXT NOT NULL UNIQUE,
                    available_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(kind, entity_id, available_at, recorded_at)
                );
                CREATE INDEX IF NOT EXISTS record_lookup ON records(kind, entity_id, recorded_at);
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO schema_version VALUES (1);
            """)
            if db.execute("SELECT version FROM schema_version").fetchall() != [(1,)]:
                raise ValueError("Unsupported repository schema version.")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            with db:
                yield db
        finally:
            db.close()

    def put(
        self,
        kind: str,
        entity_id: str,
        payload: dict[str, Any],
        *,
        available_at: datetime,
        recorded_at: datetime | None = None,
    ) -> str:
        return self.put_many([(kind, entity_id, payload, available_at)], recorded_at=recorded_at)[0]

    def put_many(
        self,
        items: list[tuple[str, str, dict[str, Any], datetime]],
        *,
        recorded_at: datetime | None = None,
        expected_records: list[tuple[str, str, str | None]] | None = None,
        deadline: datetime | None = None,
    ) -> list[str]:
        deadline = ensure_utc(deadline) if deadline is not None else None
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            # Default recording time is assigned after lock acquisition, not at
            # request start; delayed decisions must not sort behind earlier writes.
            recorded_at = ensure_utc(recorded_at or datetime.now(UTC))
            prepared = _prepare_records(items, recorded_at)
            _check_publication_inputs(db, expected_records, recorded_at)
            if deadline is not None and datetime.now(UTC) >= deadline:
                raise PublicationDeadlineError("Pre-match publication deadline passed.")
            _insert_records(db, prepared)
            if deadline is not None and datetime.now(UTC) >= deadline:
                raise PublicationDeadlineError("Pre-match publication deadline passed.")
        return [row[2] for row in prepared]

    def publish_once(
        self, kind: str, entity_id: str, payload: dict[str, Any],
        related: list[tuple[str, str, dict[str, Any], datetime]],
        *, available_at: datetime, recorded_at: datetime, deadline: datetime | None = None,
        expected_records: list[tuple[str, str, str | None]] | None = None,
    ) -> dict[str, Any]:
        """Atomically publish an immutable request result and its related records.

        Writers using this operation serialize the existence check and insertion.
        A racing retry receives the winning payload; a different request is rejected.
        """
        deadline = ensure_utc(deadline) if deadline is not None else None
        if not payload.get("request_id"):
            raise ValueError("One-time publication requires a request identity.")
        if any(k == kind and identity == entity_id for k, identity, _, _ in related):
            raise ValueError("Related records cannot replace the publication guard.")
        prepared = _prepare_records(
            [*related, (kind, entity_id, payload, available_at)], recorded_at
        )
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT record_id, available_at, recorded_at, payload FROM records "
                "WHERE kind=? AND entity_id=? ORDER BY sequence LIMIT 1", (kind, entity_id),
            ).fetchone()
            if existing:
                identity, available, recorded, raw = existing
                result: dict[str, Any] = _decode_record(
                    kind, entity_id, identity, available, recorded, raw
                )["payload"]
                if result.get("request_id") != payload["request_id"]:
                    raise ValueError("Run ID already belongs to a different request.")
                return result
            _check_publication_inputs(db, expected_records, recorded_at)
            if deadline is not None and datetime.now(UTC) >= deadline:
                raise PublicationDeadlineError("Pre-match publication deadline passed.")
            _insert_records(db, prepared)
            if deadline is not None and datetime.now(UTC) >= deadline:
                raise PublicationDeadlineError("Pre-match publication deadline passed.")
        return payload

    def snapshot_sequence(self) -> int:
        """Return the committed high-water mark for this append-only database."""
        with self.connection() as db:
            return int(db.execute("SELECT COALESCE(MAX(sequence), 0) FROM records").fetchone()[0])

    def list(
        self,
        kind: str,
        *,
        as_of: datetime | None = None,
        entity_id: str | None = None,
        limit: int = 1000,
        after_id: str | None = None,
        through_sequence: int | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000:
            raise ValueError("Repository limit must lie in [1, 10000].")
        if through_sequence is not None and (
            type(through_sequence) is not int or not 0 <= through_sequence < 2**63
        ):
            raise ValueError("Invalid snapshot sequence.")
        cutoff = ensure_utc(as_of or datetime.now(UTC)).isoformat()
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT entity_id, record_id, available_at, recorded_at, payload FROM (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id
                    ORDER BY recorded_at DESC, sequence DESC) AS rank
                  FROM records WHERE kind=? AND available_at<=? AND recorded_at<=?
                    AND (? IS NULL OR sequence<=?)
                    AND (? IS NULL OR entity_id=?)
                ) WHERE rank=1 AND (? IS NULL OR entity_id>?) ORDER BY entity_id LIMIT ?
            """,
                (kind, cutoff, cutoff, through_sequence, through_sequence,
                 entity_id, entity_id, after_id, after_id, limit),
            ).fetchall()
        result = []
        for entity, identity, available, recorded, raw in rows:
            body = _decode_record(kind, entity, identity, available, recorded, raw)
            result.append({"record_id": identity, **body})
        return result

    def history(
        self, kind: str, entity_id: str, *, as_of: datetime,
        through_sequence: int, after_sequence: int = 0, limit: int = 100,
    ) -> List[dict[str, Any]]:
        """Read revisions in committed sequence order within a frozen boundary."""
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("Repository limit must lie in [1, 10000].")
        for value in (through_sequence, after_sequence):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("Invalid history sequence.")
        cutoff = ensure_utc(as_of).isoformat()
        with self.connection() as db:
            rows = db.execute(
                "SELECT sequence, record_id, available_at, recorded_at, payload FROM records "
                "WHERE kind=? AND entity_id=? AND available_at<=? AND recorded_at<=? "
                "AND sequence>? AND sequence<=? ORDER BY sequence LIMIT ?",
                (kind, entity_id, cutoff, cutoff, after_sequence, through_sequence, limit),
            ).fetchall()
        return [
            {"sequence": sequence, "record_id": identity,
             **_decode_record(kind, entity_id, identity, available, recorded, raw)}
            for sequence, identity, available, recorded, raw in rows
        ]

    def scan(self, kind: str, *, as_of: datetime | None = None) -> List[dict[str, Any]]:
        cutoff = ensure_utc(as_of or datetime.now(UTC))
        boundary = self.snapshot_sequence()
        result: List[dict[str, Any]] = []
        cursor = None
        while True:
            page = self.list(
                kind, as_of=cutoff, after_id=cursor, limit=1000, through_sequence=boundary
            )
            result.extend(page)
            if len(page) < 1000:
                return result
            cursor = page[-1]["entity_id"]

    def get(self, kind: str, entity_id: str, *, as_of: datetime | None = None) -> dict[str, Any]:
        rows = self.list(kind, entity_id=entity_id, as_of=as_of)
        if not rows:
            raise KeyError(entity_id)
        return rows[0]

    def backup(self, destination: Path) -> None:
        """Verify a staged SQLite snapshot, then publish without replacing any file."""
        if destination.resolve() == self.path.resolve() or destination.exists():
            raise ValueError("Backup destination must be a new file.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".backup-", dir=destination.parent) as staging:
            temporary = Path(staging) / "snapshot.sqlite3"
            with self.connection() as source:
                target = sqlite3.connect(temporary)
                try:
                    source.backup(target)
                    # Produce one self-contained file, including when the source uses WAL.
                    mode = target.execute("PRAGMA journal_mode=DELETE").fetchone()
                    if mode != ("delete",):
                        raise sqlite3.DatabaseError("Backup could not leave WAL mode.")
                    if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                        raise sqlite3.DatabaseError("Backup integrity check failed.")
                    # Verify every revision, including superseded and future-visible
                    # records that ordinary latest/as-of reads may never encounter.
                    for row in target.execute(
                        "SELECT kind, entity_id, record_id, available_at, recorded_at, "
                        "payload FROM records ORDER BY sequence"
                    ):
                        _decode_record(*row)
                finally:
                    target.close()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise ValueError("Backup destination must be a new file.") from None
            # Persist the new directory entry as well as the database bytes.
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def health(self) -> bool:
        with self.connection() as db:
            return bool(db.execute("PRAGMA quick_check").fetchone() == ("ok",))


def _prepare_records(
    items: list[tuple[str, str, dict[str, Any], datetime]], recorded_at: datetime,
) -> list[tuple[str, str, str, str, str, str]]:
    recorded = ensure_utc(recorded_at).isoformat()
    prepared = []
    for kind, entity_id, payload, available_at in items:
        available = ensure_utc(available_at).isoformat()
        if not kind.strip() or not entity_id.strip() or available > recorded:
            raise ValueError("Invalid identity or availability in repository batch.")
        body = {
            "kind": kind,
            "entity_id": entity_id,
            "available_at": available,
            "recorded_at": recorded,
            "payload": payload,
        }
        identity = content_id("record_", body)
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        prepared.append((kind, entity_id, identity, available, recorded, serialized))
    return prepared


def _check_publication_inputs(
    db: sqlite3.Connection,
    expected_records: list[tuple[str, str, str | None]] | None,
    recorded_at: datetime,
) -> None:
    # An expected ID of None asserts that the record is still absent.
    # Check after acquiring the write lock, on this same transaction. No
    # writer can replace a reviewed input between this check and commit.
    cutoff = max(ensure_utc(recorded_at), datetime.now(UTC)).isoformat()
    for expected_kind, expected_entity, expected_id in expected_records or []:
        current = db.execute(
            "SELECT record_id, available_at, recorded_at, payload FROM records "
            "WHERE kind=? AND entity_id=? "
            "AND available_at<=? AND recorded_at<=? "
            "ORDER BY recorded_at DESC, sequence DESC LIMIT 1",
            (expected_kind, expected_entity, cutoff, cutoff),
        ).fetchone()
        if (current[0] if current is not None else None) != expected_id:
            raise PublicationConflictError(
                f"Publication input changed: {expected_kind}/{expected_entity}."
            )
        if current is not None:
            _decode_record(expected_kind, expected_entity, *current)



def _decode_record(
    kind: str, entity_id: str, identity: str, available: str, recorded: str, raw: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Record payload must be an object.")
        body = {
            "kind": kind, "entity_id": entity_id, "available_at": available,
            "recorded_at": recorded, "payload": payload,
        }
        if identity != content_id("record_", body):
            raise ValueError("Record hash mismatch.")
    except (ValueError, TypeError) as exc:
        raise RepositoryIntegrityError("Repository content integrity failure.") from exc
    return body



def _insert_records(
    db: sqlite3.Connection, prepared: list[tuple[str, str, str, str, str, str]],
) -> None:
    for row in prepared:
        existing = db.execute(
            "SELECT kind, entity_id, record_id, available_at, recorded_at, payload "
            "FROM records WHERE record_id=?", (row[2],),
        ).fetchone()
        if existing is not None:
            # Idempotency is valid only if the stored content still matches its ID.
            _decode_record(*existing)
        else:
            db.execute(
                "INSERT INTO records(kind,entity_id,record_id,available_at,recorded_at,"
                "payload) VALUES(?,?,?,?,?,?)", row,
            )
