"""
Thin wrappers that auto-track common pandas operations.
"""

from ..trackers.decorator import track


def tracked_read_csv(path, store=None, **kwargs):
    """pd.read_csv with automatic lineage tracking."""
    import pandas as pd

    @track(name=f"read_csv:{path}", tags={"source": str(path), "type": "ingestion"}, store=store)
    def _read(p, **kw):
        return pd.read_csv(p, **kw)

    return _read(path, **kwargs)


def tracked_merge(left, right, store=None, **kwargs):
    """pd.merge with automatic lineage tracking."""
    import pandas as pd

    @track(name="merge", tags={"type": "join"}, store=store)
    def _merge(l, r, **kw):
        return pd.merge(l, r, **kw)

    return _merge(left, right, **kwargs)
