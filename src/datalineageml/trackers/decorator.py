"""
@track decorator — wraps any function to log its inputs, outputs,
dataset hash, and execution metadata to the lineage store.

Uses the global default store (set via dlm.init()) when no explicit
store= argument is provided.
"""

import functools
import hashlib
import time
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Optional


def _hash_input(obj: Any) -> str:
    """Produce a stable hash for common ML objects (DataFrames, arrays, dicts)."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return hashlib.md5(
                pd.util.hash_pandas_object(obj, index=True).values.tobytes() # type: ignore
            ).hexdigest()
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


def _resolve_store(store):
    """Return the store to use: explicit store > global default > new default."""
    if store is not None:
        return store
    # Import here to avoid circular import at module load time
    from datalineageml import get_default_store
    from datalineageml.storage.sqlite_store import LineageStore
    default = get_default_store()
    if default is not None:
        return default
    # Fallback: create a new store at the default path.
    # This preserves v0.1 behaviour for scripts that never call dlm.init().
    return LineageStore()


def track(
    name: Optional[str] = None,
    tags: Optional[dict] = None,
    store=None,
    snapshot: bool = False,
    sensitive_cols: Optional[list] = None,
):
    """Decorator to track a data transformation step.

    Args:
        name:           Human-readable step name. Defaults to the function name.
        tags:           Arbitrary key-value metadata stored with the step record.
        store:          LineageStore to write to. If None, uses the global default
                        store (set via dlm.init()), or creates a new default store.
        snapshot:       (v0.2) If True, log a statistical snapshot of DataFrame
                        arguments before and after the function runs.
        sensitive_cols: (v0.2) Column names to track demographic distributions for.
                        Used only when snapshot=True.

    Example:
        @track(name="clean_data", tags={"stage": "preprocessing"})
        def clean_data(df):
            return df.dropna()

        # Using the global default store (after dlm.init()):
        @track(name="normalize")
        def normalize(df):
            return (df - df.mean()) / df.std()
    """
    def decorator(fn: Callable) -> Callable:
        step_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _store = _resolve_store(store)
            run_id = str(uuid.uuid4())
            started_at = datetime.utcnow().isoformat()
            t0 = time.perf_counter()

            input_hashes = {
                f"arg_{i}": _hash_input(a) for i, a in enumerate(args)
            }
            input_hashes.update({k: _hash_input(v) for k, v in kwargs.items()})

            # v0.2: snapshot before
            if snapshot:
                _log_snapshot_safe(
                    _store, run_id, step_name, "before",
                    args, kwargs, sensitive_cols or []
                )

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

                # v0.2: snapshot after (only on success)
                if snapshot and status == "success" and result is not None:
                    _log_snapshot_safe(
                        _store, run_id, step_name, "after",
                        [result], {}, sensitive_cols or []
                    )

            return result

        # Store track metadata on the wrapper so CounterfactualReplayer
        # can read it via register_tracked() without re-specifying args.
        wrapper.__track_meta__ = { # type: ignore
            "name":           step_name,
            "snapshot":       snapshot,
            "sensitive_cols": sensitive_cols,
        }
        return wrapper
    return decorator


def _log_snapshot_safe(store, run_id, step_name, position,
                        args, kwargs, sensitive_cols):
    """Attempt to log a DataFrame snapshot using DataFrameProfiler.

    Fails silently — snapshots must never break a production pipeline.
    The actual computation is delegated to DataFrameProfiler so the
    logic lives in one place and can be tested independently.
    """
    try:
        import pandas as pd
        from datalineageml.analysis.profiler import DataFrameProfiler

        # Find the first DataFrame in the arguments
        df = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, pd.DataFrame):
                df = a
                break
        if df is None:
            return  # no DataFrame argument — nothing to snapshot

        profiler = DataFrameProfiler(sensitive_cols=sensitive_cols)
        snapshot = profiler.profile(df, step_name=step_name,
                                    position=position, run_id=run_id)
        store.log_snapshot(**snapshot)

    except Exception:
        # Snapshots are best-effort — never let them break the pipeline
        pass