import pytest

from core.settings import DatabaseType, settings
from memory import get_sqlite_store, initialize_store

NS = ("user", "1")


def _use_tmp_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "SQLITE_STORE_PATH", str(tmp_path / "memory_store.db"))


@pytest.mark.asyncio
async def test_sqlite_store_persists_across_reopen(tmp_path, monkeypatch):
    _use_tmp_path(tmp_path, monkeypatch)

    async with get_sqlite_store() as store:
        await store.aput(NS, "k", {"v": 1})

    async with get_sqlite_store() as store:
        item = await store.aget(NS, "k")
        assert item is not None
        assert item.value["v"] == 1


@pytest.mark.asyncio
async def test_initialize_store_wires_sqlite(tmp_path, monkeypatch):
    _use_tmp_path(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.SQLITE)

    async with initialize_store() as store:
        assert hasattr(store, "setup")
        await store.aput(NS, "k2", {"v": 2})

    async with initialize_store() as store:
        item = await store.aget(NS, "k2")
        assert item is not None
        assert item.value["v"] == 2
