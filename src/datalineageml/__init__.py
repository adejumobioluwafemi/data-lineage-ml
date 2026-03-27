"""
DataLineageML — Causal data provenance for AI safety.

Quick start (global store — recommended):

    import datalineageml as dlm
    from datalineageml import track, LineageContext, LineageGraph

    dlm.init(db_path="pipeline.db")   # call once at top of script

    @dlm.track(name="clean_data")
    def clean_data(df):
        return df.dropna()

    with LineageContext(name="my_pipeline"):
        result = clean_data(raw_df)

    LineageGraph().show()

Quick start (explicit store — useful for tests and multi-pipeline scripts):

    from datalineageml import track, LineageContext, LineageStore, LineageGraph

    store = LineageStore(db_path="experiments/run_42.db")

    @track(name="clean_data", store=store)
    def clean_data(df):
        return df.dropna()

    with LineageContext(name="my_pipeline", store=store):
        result = clean_data(raw_df)
"""

__version__ = "0.2.0"
__author__ = "Oluwafemi Adejumobi"

from .trackers.decorator import track
from .replay.replayer import CounterfactualReplayer
from .report import generate_report
from .trackers.context import LineageContext
from .storage.sqlite_store import LineageStore
from .visualization.graph import LineageGraph

# ── global default store ──────────────────────────────────────────────────────
# Set by calling dlm.init(). When set, all @track calls and LineageContext
# instances that do not supply an explicit store= will use this automatically.

_DEFAULT_STORE: LineageStore = None  # type: ignore[assignment]


def init(db_path: str = "lineage.db") -> LineageStore:
    """Initialise the global default store.

    Call this once at the top of your script. After calling init(), every
    @track decorator and LineageContext that does not supply an explicit
    store= argument will automatically use this store.

    Args:
        db_path: Path to the SQLite lineage database.
                 Created if it does not exist.

    Returns:
        The initialised LineageStore instance.

    Example:
        import datalineageml as dlm

        dlm.init(db_path="my_pipeline.db")

        @dlm.track(name="clean")
        def clean(df):
            return df.dropna()
    """
    global _DEFAULT_STORE
    _DEFAULT_STORE = LineageStore(db_path=db_path)
    return _DEFAULT_STORE


def get_default_store() -> LineageStore:
    """Return the current global default store, or None if not initialised.

    Used internally by @track and LineageContext. Not usually called directly.
    """
    return _DEFAULT_STORE


def reset() -> None:
    """Reset the global default store to None.

    Primarily useful in tests to ensure a clean state between test runs.
    """
    global _DEFAULT_STORE
    _DEFAULT_STORE = None # type: ignore


__all__ = [
    "track",
    "CounterfactualReplayer",
    "generate_report",
    "LineageContext",
    "LineageStore",
    "LineageGraph",
    "init",
    "get_default_store",
    "reset",
]