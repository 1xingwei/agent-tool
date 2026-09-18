import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.embeddings import build_store_index
from core.settings import settings

logger = logging.getLogger(__name__)


def validate_postgres_config() -> None:
    """
    校验所有必需的 PostgreSQL 配置是否齐全。
    若缺少任何必需配置则抛出 ValueError。
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )

    if settings.POSTGRES_MIN_CONNECTIONS_PER_POOL > settings.POSTGRES_MAX_CONNECTIONS_PER_POOL:
        raise ValueError(
            f"POSTGRES_MIN_CONNECTIONS_PER_POOL ({settings.POSTGRES_MIN_CONNECTIONS_PER_POOL}) must be less than or equal to POSTGRES_MAX_CONNECTIONS_PER_POOL ({settings.POSTGRES_MAX_CONNECTIONS_PER_POOL})"
        )


def get_postgres_connection_string() -> str:
    """根据 settings 构建并返回 PostgreSQL 连接字符串。"""
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


@asynccontextmanager
async def get_postgres_saver():
    """基于连接池初始化并返回一个 PostgreSQL saver 实例，以获得更健壮的连接。"""
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "saver"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # Langgraph 要求 autocommmit=true 且 row_factory 设置为 dict_row。
        # 传入 application_name 以便在 Postgres 数据库连接管理器中识别该连接。
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # 确保连接在使用前仍然有效
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            checkpointer = AsyncPostgresSaver(pool)  # type: ignore[bad-argument-type]
            await checkpointer.setup()
            yield checkpointer
        finally:
            await pool.close()


@asynccontextmanager
async def get_postgres_store():
    """
    基于连接池获取一个 PostgreSQL store 实例，以获得更健壮的连接。

    返回一个可与异步上下文管理器模式配合使用的 AsyncPostgresStore 实例。

    `index` 决定语义检索是否开启（pgvector）。**不传时 LangGraph 静默关闭语义检索**
    —— `search(query=...)` 仍返回结果，但按主键序且 `score is None`。
    这里显式构造；embedding 不可用时退回纯 KV 并留下告警。
    """
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "store"

    index = build_store_index()
    if index is None:
        logger.warning(
            "Postgres store 未启用语义检索（embedding 不可用或 pgvector 未装）；"
            "记忆仍可读写，但 store.asearch(query=...) 不会返回相似度排序。"
        )

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # Langgraph 要求 autocommmit=true 且 row_factory 设置为 dict_row
        # 传入 application_name 以便在 Postgres 数据库连接管理器中识别该连接。
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # 确保连接在使用前仍然有效
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            store = AsyncPostgresStore(pool, index=index)  # type: ignore[bad-argument-type]
            await store.setup()
            yield store
        finally:
            await pool.close()
