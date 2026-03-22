"""
SQLite-backed lineage store.
All lineage records persist to a local .db file.

Default path: ./lineage.db (current working directory).
Always pass db_path explicitly in scripts and tests to avoid
writing to an unexpected location.
"""

import sqlite3
import json
import os
from typing import Optional, List, Dict


# Default is current working directory so it's visible and easy to inspect.
# In production pipelines, always pass an explicit db_path.
DEFAULT_DB_PATH = os.path.join(os.getcwd(), "lineage.db")


class LineageStore:
    """
    SQLite-backed store for lineage records.

    Args:
        db_path: Path to the SQLite file. Defaults to ./lineage.db.
                 Pass an explicit path in all scripts and tests.

    Example:
        store = LineageStore(db_path="experiments/run_01/lineage.db")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        # Ensure parent directory exists
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS steps (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL,
                step_name    TEXT NOT NULL,
                fn_module    TEXT,
                fn_qualname  TEXT,
                input_hashes TEXT,
                output_hash  TEXT,
                duration_ms  REAL,
                started_at   TEXT,
                status       TEXT,
                error        TEXT,
                tags         TEXT
            );

            CREATE TABLE IF NOT EXISTS pipelines (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id  TEXT NOT NULL,
                name         TEXT NOT NULL,
                started_at   TEXT,
                ended_at     TEXT,
                status       TEXT DEFAULT 'running'
            );
        """)
        self._conn.commit()

    def log_step(self, *, run_id, step_name, fn_module, fn_qualname,
                 input_hashes, output_hash, duration_ms, started_at,
                 status, error, tags):
        self._conn.execute("""
            INSERT INTO steps
              (run_id, step_name, fn_module, fn_qualname, input_hashes,
               output_hash, duration_ms, started_at, status, error, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, step_name, fn_module, fn_qualname,
            json.dumps(input_hashes), output_hash,
            duration_ms, started_at, status, error,
            json.dumps(tags),
        ))
        self._conn.commit()

    def log_pipeline_start(self, *, pipeline_id, name, started_at):
        self._conn.execute("""
            INSERT INTO pipelines (pipeline_id, name, started_at)
            VALUES (?,?,?)
        """, (pipeline_id, name, started_at))
        self._conn.commit()

    def log_pipeline_end(self, *, pipeline_id, status, ended_at):
        self._conn.execute("""
            UPDATE pipelines SET status=?, ended_at=?
            WHERE pipeline_id=?
        """, (status, ended_at, pipeline_id))
        self._conn.commit()

    def get_steps(self, step_name: Optional[str] = None) -> List[Dict]:
        if step_name:
            rows = self._conn.execute(
                "SELECT * FROM steps WHERE step_name=? ORDER BY started_at",
                (step_name,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM steps ORDER BY started_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pipelines(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM pipelines ORDER BY started_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def clear(self):
        """Wipe all records — useful in tests and fresh demo runs."""
        self._conn.executescript("DELETE FROM steps; DELETE FROM pipelines;")
        self._conn.commit()

    def close(self):
        self._conn.close()
