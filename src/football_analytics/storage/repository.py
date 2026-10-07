"""SQLite append-only, bitemporal JSON records and atomic backup."""

from __future__ import annotations

import json
import sqlite3
from builtins import list as List
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_analytics.data.contracts import ensure_utc
from football_analytics.models.postprocessing import content_id


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
    ) -> list[str]:
        recorded = ensure_utc(recorded_at or datetime.now(UTC)).isoformat()
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
        with self.connection() as db:
            for row in prepared:
                if not db.execute("SELECT 1 FROM records WHERE record_id=?", (row[2],)).fetchone():
                    db.execute(
                        "INSERT INTO records(kind,entity_id,record_id,available_at,recorded_at,"
                        "payload) VALUES(?,?,?,?,?,?)",
                        row,
                    )
        return [row[2] for row in prepared]

    def list(
        self,
        kind: str,
        *,
        as_of: datetime | None = None,
        entity_id: str | None = None,
        limit: int = 1000,
        after_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000:
            raise ValueError("Repository limit must lie in [1, 10000].")
        cutoff = ensure_utc(as_of or datetime.now(UTC)).isoformat()
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT entity_id, record_id, available_at, recorded_at, payload FROM (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id
                    ORDER BY recorded_at DESC, sequence DESC) AS rank
                  FROM records WHERE kind=? AND available_at<=? AND recorded_at<=?
                    AND (? IS NULL OR entity_id=?)
                ) WHERE rank=1 AND (? IS NULL OR entity_id>?) ORDER BY entity_id LIMIT ?
            """,
                (kind, cutoff, cutoff, entity_id, entity_id, after_id, after_id, limit),
            ).fetchall()
        result = []
        for entity, identity, available, recorded, raw in rows:
            payload = json.loads(raw)
            body = {
                "kind": kind,
                "entity_id": entity,
                "available_at": available,
                "recorded_at": recorded,
                "payload": payload,
            }
            if identity != content_id("record_", body):
                raise ValueError("Repository content integrity failure.")
            result.append({"record_id": identity, **body})
        return result

    def scan(self, kind: str, *, as_of: datetime | None = None) -> List[dict[str, Any]]:
        cutoff = ensure_utc(as_of or datetime.now(UTC))
        result: List[dict[str, Any]] = []
        cursor = None
        while True:
            page = self.list(kind, as_of=cutoff, after_id=cursor, limit=1000)
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
        if destination.resolve() == self.path.resolve() or destination.exists():
            raise ValueError("Backup destination must be a new file.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()

    def health(self) -> bool:
        with self.connection() as db:
            return bool(db.execute("PRAGMA quick_check").fetchone() == ("ok",))
