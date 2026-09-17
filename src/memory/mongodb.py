import asyncio
import logging
import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from core.settings import settings

logger = logging.getLogger(__name__)


def _has_auth_credentials() -> bool:
    required_auth = ["MONGO_USER", "MONGO_PASSWORD", "MONGO_AUTH_SOURCE"]
    set_auth = [var for var in required_auth if getattr(settings, var, None)]
    if len(set_auth) > 0 and len(set_auth) != len(required_auth):
        raise ValueError(
            f"If any of the following environment variables are set, all must be set: {', '.join(required_auth)}."
        )
    return len(set_auth) == len(required_auth)


def validate_mongo_config() -> None:
    """
    校验所有必需的 MongoDB 配置是否齐全。
    若缺少任何必需配置则抛出 ValueError。
    """
    required_always = ["MONGO_HOST", "MONGO_PORT", "MONGO_DB"]
    missing_always = [var for var in required_always if not getattr(settings, var, None)]
    if missing_always:
        raise ValueError(
            f"Missing required MongoDB configuration: {', '.join(missing_always)}. "
            "These environment variables must be set to use MongoDB persistence."
        )

    _has_auth_credentials()


def get_mongo_connection_string() -> str:
    """根据 settings 构建并返回 MongoDB 连接字符串。"""

    tls_param = "&tls=true" if settings.MONGO_TLS else ""
    if _has_auth_credentials():
        if settings.MONGO_PASSWORD is None:  # 用于类型检查
            raise ValueError("MONGO_PASSWORD is not set")
        password = settings.MONGO_PASSWORD.get_secret_value().strip()
        password_escaped = urllib.parse.quote_plus(password)
        return (
            f"mongodb://{settings.MONGO_USER}:{password_escaped}@"
            f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
            f"?authSource={settings.MONGO_AUTH_SOURCE}{tls_param}"
        )
    else:
        tls_query = "?tls=true" if settings.MONGO_TLS else ""
        return f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/{tls_query}"


class _AsyncMongoDBSaver(AbstractAsyncContextManager[MongoDBSaver]):
    """包装 MongoDBSaver 的异步上下文管理器，截至
    langgraph-checkpoint-mongodb 0.4 它仅支持同步（内部通过线程执行器桥接到异步）。
    连接和构建 saver 的索引都是阻塞 I/O，因此两者都在事件循环线程之外运行。
    """

    def __init__(self, conn_string: str, db_name: str):
        self._conn_string = conn_string
        self._db_name = db_name
        self._saver: MongoDBSaver | None = None

    async def __aenter__(self) -> MongoDBSaver:
        def _connect() -> MongoDBSaver:
            client: MongoClient = MongoClient(self._conn_string)
            return MongoDBSaver(client, db_name=self._db_name)

        self._saver = await asyncio.to_thread(_connect)
        return self._saver

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._saver is not None:
            await asyncio.to_thread(self._saver.close)


def get_mongo_saver() -> AbstractAsyncContextManager[MongoDBSaver]:
    """初始化并返回一个 MongoDB saver 实例。"""
    validate_mongo_config()
    if settings.MONGO_DB is None:  # 用于类型检查
        raise ValueError("MONGO_DB is not set")
    return _AsyncMongoDBSaver(get_mongo_connection_string(), settings.MONGO_DB)
