"""Ephemeral RAM diary — SQLite in /dev/shm for zero-persistence by default.

All data lives in `/dev/shm/ghost_diary.db` (or the path set by
``GHOST_DIARY_PATH``).  The OS wipes it on reboot, guaranteeing ephemerality.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_PATH = os.environ.get("GHOST_DIARY_PATH", "/dev/shm/ghost_diary.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    tick        INTEGER NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'thought',
    content     TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS insights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    source_from INTEGER NOT NULL,
    source_to   INTEGER NOT NULL,
    content     TEXT    NOT NULL
);
"""


@dataclass
class DiaryEntry:
    tick: int
    role: str       # 'thought' | 'action' | 'reflection' | 'dream' | 'error'
    content: str
    metadata: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: int | None = None


class RamDiary:
    """Thread-safe SQLite diary living entirely in RAM (/dev/shm)."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._open()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def wipe(self) -> None:
        """Destroy all entries and insights (used after Janitor compression)."""
        assert self._conn is not None
        self._conn.execute("DELETE FROM entries")
        self._conn.execute("DELETE FROM insights")
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def append(self, entry: DiaryEntry) -> int:
        assert self._conn is not None
        cur = self._conn.execute(
            "INSERT INTO entries (ts, tick, role, content, metadata) VALUES (?,?,?,?,?)",
            (
                entry.ts,
                entry.tick,
                entry.role,
                entry.content,
                json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def add_insight(self, content: str, source_from: int, source_to: int) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO insights (ts, source_from, source_to, content) VALUES (?,?,?,?)",
            (time.time(), source_from, source_to, content),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def recent(self, n: int = 20) -> list[DiaryEntry]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM entries ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [_row_to_entry(r) for r in reversed(rows)]

    def all_entries(self) -> list[DiaryEntry]:
        assert self._conn is not None
        rows = self._conn.execute("SELECT * FROM entries ORDER BY id ASC").fetchall()
        return [_row_to_entry(r) for r in rows]

    def insights(self) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM insights ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def entry_count(self) -> int:
        assert self._conn is not None
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


def _row_to_entry(row: sqlite3.Row) -> DiaryEntry:
    return DiaryEntry(
        id=row["id"],
        ts=row["ts"],
        tick=row["tick"],
        role=row["role"],
        content=row["content"],
        metadata=json.loads(row["metadata"]),
    )
