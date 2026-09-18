import logging
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite import AsyncSqliteStore

from core.embeddings import build_store_index
from core.settings import settings

logger = logging.getLogger(__name__)


def _ensure_parent_dir(db_path: str) -> None:
    """创建存放 SQLite 文件的目录（若缺失）。

    路径默认为 `var/`，git 不跟踪该目录，因此全新克隆
    （以及仅复制 `src/` 的容器镜像）启动时该目录不存在。
    SQLite 不会自行创建缺失的父目录。
    """
    if db_path == ":memory:":
        return
    parent = Path(db_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """初始化并返回一个 SQLite saver 实例。"""
    _ensure_parent_dir(settings.SQLITE_DB_PATH)
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


@asynccontextmanager
async def get_sqlite_store():
    """初始化并返回用于长期记忆的 store 实例。

    持久化到 SQLite，使长期记忆在重启后依然保留。

    `index` 决定语义检索是否开启：**不传时 LangGraph 静默关闭语义检索**
    （`search(query=...)` 仍返回结果，但按主键序且 `score is None`）。
    因此这里显式构造，embedding 不可用时退回纯 KV 并留下告警，而不是让服务起不来。
    """
    _ensure_parent_dir(settings.SQLITE_STORE_PATH)

    index = build_store_index()
    if index is None:
        logger.warning(
            "SQLite store 未启用语义检索（embedding 不可用）；记忆仍可读写，"
            "但 store.asearch(query=...) 不会返回相似度排序。"
        )

    async with AsyncSqliteStore.from_conn_string(
        settings.SQLITE_STORE_PATH,
        index=index,  # type: ignore[bad-argument-type]
    ) as store:
        await store.setup()
        yield store
