import logging
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from schema import AgentInfo


@pytest.mark.asyncio
async def test_lifespan(monkeypatch, caplog) -> None:
    """测试 lifespan 设置数据库和 store、加载 agent 并记录错误。"""
    from service import service

    fake_saver_setup = False
    fake_store_setup = False

    class FakeSaver:
        async def setup(self) -> None:
            nonlocal fake_saver_setup
            fake_saver_setup = True

    class FakeStore:
        async def setup(self) -> None:
            nonlocal fake_store_setup
            fake_store_setup = True

    fake_saver = FakeSaver()
    fake_store = FakeStore()

    @asynccontextmanager
    async def fake_initialize_database():
        yield fake_saver

    @asynccontextmanager
    async def fake_initialize_store():
        yield fake_store

    agents = {
        "good": type("Agent", (), {"checkpointer": None, "store": None})(),
        "bad": type("Agent", (), {"checkpointer": None, "store": None})(),
    }

    async def fake_load_agent(agent_key: str) -> None:
        if agent_key == "bad":
            raise RuntimeError("boom")

    def fake_get_agent(agent_key: str):
        return agents[agent_key]

    monkeypatch.setattr(service, "initialize_database", fake_initialize_database)
    monkeypatch.setattr(service, "initialize_store", fake_initialize_store)
    monkeypatch.setattr(service, "load_agent", fake_load_agent)
    monkeypatch.setattr(service, "get_agent", fake_get_agent)
    monkeypatch.setattr(
        service,
        "get_all_agent_info",
        lambda: [
            AgentInfo(key="good", description=""),
            AgentInfo(key="bad", description=""),
        ],
    )

    caplog.set_level(logging.INFO, logger=service.logger.name)

    async with service.lifespan(FastAPI()):
        pass

    assert fake_saver_setup
    assert fake_store_setup
    assert agents["good"].checkpointer is fake_saver
    assert agents["good"].store is fake_store
    # F4（docs/20）：load 失败的 agent 被跳过，不再拿到记忆组件
    assert agents["bad"].checkpointer is None

    assert "Agent loaded: good" in caplog.text
    assert "Failed to load agent bad: boom" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_survives_get_agent_after_load_failure(monkeypatch) -> None:
    """load 失败后紧接着的 get_agent 若抛 RuntimeError，不得拖垮整个 lifespan（docs/20 F4）。

    反例注入：把 get_agent(a.key) 挪回 try 之外，本用例必须变红。
    """
    from service import service

    async def _noop_setup(self) -> None:
        pass

    @asynccontextmanager
    async def fake_initialize_database():
        yield type("Saver", (), {"setup": _noop_setup, "checkpointer": None})()

    @asynccontextmanager
    async def fake_initialize_store():
        yield type("Store", (), {"setup": _noop_setup, "store": None})()

    async def fake_load_agent(agent_key: str) -> None:
        if agent_key == "bad":
            raise RuntimeError("boom")

    def fake_get_agent(agent_key: str):
        if agent_key == "bad":
            raise RuntimeError(f"Agent {agent_key} not loaded. Call load() first.")
        raise KeyError(agent_key)  # 只会在 bad 上被调用

    monkeypatch.setattr(service, "initialize_database", fake_initialize_database)
    monkeypatch.setattr(service, "initialize_store", fake_initialize_store)
    monkeypatch.setattr(service, "load_agent", fake_load_agent)
    monkeypatch.setattr(service, "get_agent", fake_get_agent)
    monkeypatch.setattr(
        service,
        "get_all_agent_info",
        lambda: [AgentInfo(key="bad", description="")],
    )

    async with service.lifespan(FastAPI()):
        pass  # 不应抛出


@pytest.mark.asyncio
async def test_lifespan_does_not_touch_var_sqlite(monkeypatch) -> None:
    """真实 lifespan 打开 saver/store 时不得落在生产 `var/` 下的 SQLite 库（docs/20 V1）。

    决定性实验：删掉 `tests/conftest.py` `_isolate_stateful_backends` 里的两条
    SQLITE setattr，本用例必须变红——lifespan 会就地重建 `var/checkpoints.db`。
    """
    import time
    from pathlib import Path

    from fastapi import FastAPI

    from service import service

    var = Path(__file__).resolve().parents[2] / "var"

    def snapshot():
        snap = {}
        for name in ("checkpoints.db", "memory_store.db"):
            p = var / name
            snap[name] = (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
        return snap

    before = snapshot()
    monkeypatch.setattr(service, "get_all_agent_info", lambda: [])
    async with service.lifespan(FastAPI()):
        pass  # 真实 initialize_database/initialize_store，走被隔离的 tmp 路径
    time.sleep(0.05)
    assert snapshot() == before, "lifespan 触碰了生产 var/ 下的 SQLite 库"
