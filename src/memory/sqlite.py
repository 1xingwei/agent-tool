from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite import AsyncSqliteStore

from core.settings import settings


def _ensure_parent_dir(db_path: str) -> None:
    """Create the directory that holds a SQLite file, if it is missing.

    The paths default to `var/`, which git does not track, so a fresh clone
    (and the container image, which only copies `src/`) starts without it.
    SQLite will not create missing parent directories on its own.
    """
    if db_path == ":memory:":
        return
    parent = Path(db_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Initialize and return a SQLite saver instance."""
    _ensure_parent_dir(settings.SQLITE_DB_PATH)
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


@asynccontextmanager
async def get_sqlite_store():
    """Initialize and return a store instance for long-term memory.

    Persisted to SQLite so long-term memory survives restarts.
    """
    _ensure_parent_dir(settings.SQLITE_STORE_PATH)
    async with AsyncSqliteStore.from_conn_string(settings.SQLITE_STORE_PATH) as store:
        await store.setup()
        yield store
