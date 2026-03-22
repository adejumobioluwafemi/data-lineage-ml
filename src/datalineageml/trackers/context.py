"""
LineageContext — groups multiple tracked steps into a named pipeline run.
"""

import uuid
from datetime import datetime
from ..storage.sqlite_store import LineageStore


class LineageContext:
    """
    Context manager to group multiple @track steps under one pipeline run.

    Usage:
        with LineageContext(name="training_pipeline") as ctx:
            df = load_data(path)
            df = clean_data(df)
            model = train_model(df)
    """

    def __init__(self, name: str, store: LineageStore = None):
        self.name = name
        self.pipeline_id = str(uuid.uuid4())
        self.store = store or LineageStore()
        self.started_at = None

    def __enter__(self):
        self.started_at = datetime.utcnow().isoformat()
        self.store.log_pipeline_start(
            pipeline_id=self.pipeline_id,
            name=self.name,
            started_at=self.started_at,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "success"
        self.store.log_pipeline_end(
            pipeline_id=self.pipeline_id,
            status=status,
            ended_at=datetime.utcnow().isoformat(),
        )
        return False
