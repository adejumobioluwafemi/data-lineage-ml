"""
SQLite-backed lineage store.

Three tables:
  steps     — one row per @track function call (unchanged from v0.1)
  pipelines — one row per LineageContext run (unchanged from v0.1)
  snapshots — statistical profile of data at a pipeline step (NEW v0.2)
  metrics   — safety/fairness measurements linked to a run (NEW v0.2)

Default path: ./lineage.db (current working directory).
Always pass db_path explicitly in scripts and tests to avoid writing
to an unexpected location.
"""

import sqlite3
import json
import os
from typing import Optional, List, Dict, Any


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
                                 
            CREATE TABLE IF NOT EXISTS snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                step_name       TEXT NOT NULL,
                position        TEXT NOT NULL CHECK(position IN ('before', 'after')),
                row_count       INTEGER,
                column_count    INTEGER,
                column_names    TEXT,
                null_rates      TEXT,
                numeric_stats   TEXT,
                categorical_stats TEXT,
                sensitive_stats TEXT,
                recorded_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL,
                metric_name  TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_source TEXT,
                step_name    TEXT,
                tags         TEXT,
                measured_at  TEXT
            );                                                                 
        """)
        self._conn.commit()

    def log_step(self, *, run_id: str, step_name: str, fn_module: str,
                 fn_qualname: str, input_hashes: dict, output_hash: Optional[str],
                 duration_ms: float, started_at: str, status: str,
                 error: Optional[str], tags: dict) -> None:
        """Persist a completed pipeline step."""
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

    def log_pipeline_start(self, *, pipeline_id: str, name: str,
                           started_at: str) -> None:
        """Record the start of a named pipeline run."""
        self._conn.execute("""
            INSERT INTO pipelines (pipeline_id, name, started_at)
            VALUES (?,?,?)
        """, (pipeline_id, name, started_at))
        self._conn.commit()

    def log_pipeline_end(self, *, pipeline_id: str, status: str,
                         ended_at: str) -> None:
        """Record the completion of a pipeline run."""
        self._conn.execute("""
            UPDATE pipelines SET status=?, ended_at=?
            WHERE pipeline_id=?
        """, (status, ended_at, pipeline_id))
        self._conn.commit()

    def get_steps(self, step_name: Optional[str] = None) -> List[Dict]:
        """Return logged steps, optionally filtered by step name."""
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
        """Return all logged pipeline runs."""
        rows = self._conn.execute(
            "SELECT * FROM pipelines ORDER BY started_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def log_snapshot(self, *, run_id: str, step_name: str,
                     position: str, row_count: int, column_count: int,
                     column_names: List[str], null_rates: Dict[str, float],
                     numeric_stats: Dict[str, Dict], categorical_stats: Dict[str, Dict],
                     sensitive_stats: Dict[str, Dict], recorded_at: str) -> None:
        """Persist a statistical snapshot of data at a pipeline step.

        Args:
            run_id:            UUID of the @track call this snapshot belongs to.
            step_name:         Name of the pipeline step.
            position:          'before' (input snapshot) or 'after' (output snapshot).
            row_count:         Number of rows in the DataFrame.
            column_count:      Number of columns.
            column_names:      List of column names.
            null_rates:        Dict mapping column name → fraction of null values (0–1).
            numeric_stats:     Dict mapping column name → {mean, std, min, max, p25, p75}.
            categorical_stats: Dict mapping column name → {value: count, ...} (top 10).
            sensitive_stats:   Dict mapping sensitive column name → {value: fraction, ...}.
            recorded_at:       UTC ISO timestamp.

        Example:
            store.log_snapshot(
                run_id="uuid", step_name="clean_data", position="after",
                row_count=7403, column_count=5,
                column_names=["age", "gender", "income"],
                null_rates={"age": 0.0, "gender": 0.0},
                numeric_stats={"age": {"mean": 38.2, "std": 12.1, ...}},
                categorical_stats={"gender": {"F": 1637, "M": 5766}},
                sensitive_stats={"gender": {"F": 0.221, "M": 0.779}},
                recorded_at="2026-03-18T07:00:00"
            )
        """
        assert position in ("before", "after"), \
            f"position must be 'before' or 'after', got: {position!r}"

        self._conn.execute("""
            INSERT INTO snapshots
              (run_id, step_name, position, row_count, column_count,
               column_names, null_rates, numeric_stats, categorical_stats,
               sensitive_stats, recorded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, step_name, position, row_count, column_count,
            json.dumps(column_names),
            json.dumps(null_rates),
            json.dumps(numeric_stats),
            json.dumps(categorical_stats),
            json.dumps(sensitive_stats),
            recorded_at,
        ))
        self._conn.commit()

    def get_snapshots(self, step_name: Optional[str] = None,
                      position: Optional[str] = None) -> List[Dict]:
        """Return logged snapshots with optional filters.

        Args:
            step_name: Filter to a specific step name.
            position:  Filter to 'before' or 'after' snapshots only.

        Returns:
            List of snapshot dicts. JSON fields are returned as parsed dicts.
        """
        clauses, params = [], []
        if step_name:
            clauses.append("step_name=?")
            params.append(step_name)
        if position:
            clauses.append("position=?")
            params.append(position)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM snapshots {where} ORDER BY recorded_at",
            params
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            for field in ("column_names", "null_rates", "numeric_stats",
                          "categorical_stats", "sensitive_stats"):
                if d.get(field):
                    d[field] = json.loads(d[field])
            result.append(d)
        return result

    def log_metrics(self, *, run_id: str, metrics: Dict[str, float],
                    metric_source: Optional[str] = None,
                    step_name: Optional[str] = None,
                    tags: Optional[Dict] = None,
                    measured_at: Optional[str] = None) -> None:
        """Persist one or more safety/fairness metrics linked to a run.

        Args:
            run_id:        UUID of the pipeline run these metrics describe.
            metrics:       Dict mapping metric name → numeric value.
            metric_source: Tool or method that produced the metrics (e.g. 'equitrace').
            step_name:     Pipeline step these metrics are associated with, if any.
            tags:          Arbitrary key-value metadata.
            measured_at:   UTC ISO timestamp (defaults to now).

        Example:
            store.log_metrics(
                run_id="uuid",
                metrics={"gender_bias_score": 0.34, "accuracy": 0.91},
                metric_source="equitrace",
                step_name="train_model",
                tags={"dataset_version": "v3", "model": "random_forest"}
            )
        """
        from datetime import datetime
        ts = measured_at or datetime.utcnow().isoformat()

        for name, value in metrics.items():
            self._conn.execute("""
                INSERT INTO metrics
                  (run_id, metric_name, metric_value, metric_source,
                   step_name, tags, measured_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                run_id, name, float(value),
                metric_source, step_name,
                json.dumps(tags or {}), ts,
            ))
        self._conn.commit()

    def get_metrics(self, run_id: Optional[str] = None,
                    metric_name: Optional[str] = None) -> List[Dict]:
        """Return logged metrics with optional filters.

        Args:
            run_id:      Filter to a specific pipeline run.
            metric_name: Filter to a specific metric name.

        Returns:
            List of metric dicts. The 'tags' field is returned as a parsed dict.
        """
        clauses, params = [], []
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if metric_name:
            clauses.append("metric_name=?")
            params.append(metric_name)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM metrics {where} ORDER BY measured_at",
            params
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            if d.get("tags"):
                d["tags"] = json.loads(d["tags"])
            result.append(d)
        return result
    
    def clear(self) -> None:
        """Wipe all records from all tables. Useful in tests and fresh demo runs."""
        self._conn.executescript(
            "DELETE FROM steps; DELETE FROM pipelines; "
            "DELETE FROM snapshots; DELETE FROM metrics;"
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
