"""
LineageContext — groups multiple tracked steps into a named pipeline run.
"""

import uuid
from datetime import datetime
from typing import Optional
#from ..storage.sqlite_store import LineageStore


class LineageContext:
    """Context manager to group multiple @track steps under one pipeline run.

    Uses the global default store (set via dlm.init()) when no explicit
    store= argument is provided.

    Args:
        name:    Human-readable name for this pipeline run.
        store:   LineageStore to write to. If None, uses the global default
                 store (set via dlm.init()), or creates a new default store.

    Example:
        # With global store (after dlm.init()):
        with LineageContext(name="training_pipeline"):
            df = clean_data(raw)
            model = train(df)

        # With explicit store:
        with LineageContext(name="experiment_42", store=my_store):
            df = clean_data(raw)
            model = train(df)
    """

    def __init__(self, name: str, store=None):
        self.name = name
        self._store_arg = store
        self.pipeline_id = str(uuid.uuid4())
        self.started_at: Optional[str] = None
        self._store = None

    def _resolve_store(self):
        if self._store_arg is not None:
            return self._store_arg
        from datalineageml import get_default_store
        from datalineageml.storage.sqlite_store import LineageStore
        default = get_default_store()
        return default if default is not None else LineageStore()
    
    def __enter__(self):
        self._store = self._resolve_store()
        self.started_at = datetime.utcnow().isoformat()
        self._store.log_pipeline_start(
            pipeline_id=self.pipeline_id,
            name=self.name,
            started_at=self.started_at,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "success"
        self._store.log_pipeline_end( # type: ignore
            pipeline_id=self.pipeline_id,
            status=status,
            ended_at=datetime.utcnow().isoformat(),
        )
        return False

