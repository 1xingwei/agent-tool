from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.redis.aio import AsyncRedisStore

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | MongoDBSaver | AsyncRedisSaver
]:
    """
    根据配置初始化相应的数据库 checkpointer。
    返回一个已初始化的 AsyncCheckpointer 实例。
    """
    if settings.REDIS_URL:
        return AsyncRedisSaver.from_conn_string(settings.REDIS_URL)
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    else:  # 默认使用 SQLite
        return get_sqlite_saver()


def initialize_store():
    """
    根据配置初始化相应的 store。
    返回已初始化 store 的异步上下文管理器。
    """
    if settings.REDIS_URL:
        return AsyncRedisStore.from_conn_string(settings.REDIS_URL)
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: 添加 Mongo store - https://pypi.org/project/langgraph-store-mongodb/
    else:  # 默认使用 SQLite
        return get_sqlite_store()


__all__ = ["initialize_database", "initialize_store"]
