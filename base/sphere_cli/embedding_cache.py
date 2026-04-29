from __future__ import annotations

import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    vector_dim INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT,
    last_accessed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_content_hash ON embedding_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(provider, model_name);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class PersistentEmbeddingCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def count(self) -> int:
        row = self._connect().execute("SELECT COUNT(*) AS c FROM embedding_cache").fetchone()
        return int(row["c"] if row else 0)

    def get_many(self, cache_keys: Iterable[str]) -> dict[str, list[float]]:
        unique_keys = list(dict.fromkeys(key for key in cache_keys if key))
        if not unique_keys:
            return {}
        conn = self._connect()
        rows: list[sqlite3.Row] = []
        for batch in self._batched(unique_keys, size=400):
            placeholders = ",".join(["?"] * len(batch))
            rows.extend(
                conn.execute(
                    f"SELECT cache_key, embedding_blob FROM embedding_cache WHERE cache_key IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
            )
        return {
            str(row["cache_key"]): self._decode_embedding(row["embedding_blob"])
            for row in rows
        }

    def put_many(
        self,
        rows: Iterable[tuple[str, str, str, list[float], str, str]],
    ) -> None:
        payload = list(rows)
        if not payload:
            return
        stamp = _now_iso()
        conn = self._connect()
        conn.executemany(
            """
            INSERT INTO embedding_cache (
                cache_key, content_hash, normalized_text, embedding_blob, vector_dim,
                provider, model_name, created_at, last_accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                normalized_text=excluded.normalized_text,
                embedding_blob=excluded.embedding_blob,
                vector_dim=excluded.vector_dim,
                provider=excluded.provider,
                model_name=excluded.model_name,
                last_accessed_at=excluded.last_accessed_at
            """,
            [
                (
                    cache_key,
                    content_hash,
                    normalized_text,
                    sqlite3.Binary(self._encode_embedding(embedding)),
                    len(embedding),
                    provider,
                    model_name,
                    stamp,
                    stamp,
                )
                for cache_key, content_hash, normalized_text, embedding, provider, model_name in payload
            ],
        )
        conn.commit()

    def _encode_embedding(self, embedding: list[float]) -> bytes:
        values = array("f", embedding)
        return values.tobytes()

    def _decode_embedding(self, blob: bytes) -> list[float]:
        values = array("f")
        values.frombytes(blob)
        return list(values)

    def _batched(self, items: list[str], size: int) -> list[list[str]]:
        return [items[index : index + size] for index in range(0, len(items), size)]
