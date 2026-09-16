from unittest.mock import patch

from core.settings import settings
from memory import initialize_database, initialize_store

REDIS_URL = "redis://localhost:6379"


def test_initialize_store_uses_redis_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL)
    with patch("memory.AsyncRedisStore.from_conn_string") as mock:
        initialize_store()
    mock.assert_called_once_with(REDIS_URL)


def test_initialize_database_uses_redis_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", REDIS_URL)
    with patch("memory.AsyncRedisSaver.from_conn_string") as mock:
        initialize_database()
    mock.assert_called_once_with(REDIS_URL)


def test_initialize_falls_back_without_redis(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", None)
    with patch("memory.AsyncRedisStore.from_conn_string") as mock:
        initialize_store()
    mock.assert_not_called()
