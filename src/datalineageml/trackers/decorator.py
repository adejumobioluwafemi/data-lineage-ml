"""
@track decorator — wraps any function to log its inputs, outputs,
dataset hash, and execution metadata to the lineage store.
"""

import functools
import hashlib
import time
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from ..storage.sqlite_store import LineageStore


def _hash_input(obj: Any) -> str:
    """Produce a stable hash for common ML objects (DataFrames, arrays, dicts)."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return hashlib.md5(pd.util.hash_pandas_object(obj).values).hexdigest()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return hashlib.md5(obj.tobytes()).hexdigest()
    except ImportError:
        pass
    try:
        raw = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.md5(raw).hexdigest()
    except Exception:
        return hashlib.md5(str(obj).encode()).hexdigest()


def track(
    name: Optional[str] = None,
    tags: Optional[dict] = None,
    store: Optional[LineageStore] = None,
):
    """
    Decorator to track a data transformation step.

    Usage:
        @track(name="clean_data", tags={"stage": "preprocessing"})
        def clean_data(df):
            return df.dropna()

        @track()
        def my_transform(df):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        step_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _store = store or LineageStore()
            run_id = str(uuid.uuid4())
            started_at = datetime.utcnow().isoformat()
            t0 = time.perf_counter()

            input_hashes = {
                f"arg_{i}": _hash_input(a) for i, a in enumerate(args)
            }
            input_hashes.update({k: _hash_input(v) for k, v in kwargs.items()})

            result = None
            output_hash = None
            status = "success"
            error = None

            try:
                result = fn(*args, **kwargs)
                output_hash = _hash_input(result)
            except Exception as exc:
                status = "failed"
                error = str(exc)
                raise
            finally:
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                _store.log_step(
                    run_id=run_id,
                    step_name=step_name,
                    fn_module=fn.__module__,
                    fn_qualname=fn.__qualname__,
                    input_hashes=input_hashes,
                    output_hash=output_hash,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    status=status,
                    error=error,
                    tags=tags or {},
                )
            return result
        return wrapper
    return decorator
