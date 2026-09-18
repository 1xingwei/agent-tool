from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport
from langchain_core.messages import AIMessage
from langgraph.types import StateSnapshot

from core import settings
from service import app


@pytest.fixture(autouse=True)
def _disable_auth_by_default(monkeypatch):
    """保持此测试套件独立于开发者本地的 .env。

    Settings 会从 .env 加载 AUTH_SECRET，因此若开发者配置过密钥，这些测试里
    每个未认证请求都会返回 401。认证行为本身由 test_auth.py 覆盖，
    其中显式设置了密钥。
    """
    monkeypatch.setattr(settings, "AUTH_SECRET", None)


@pytest.fixture
def test_client():
    """用于创建 FastAPI 测试客户端的 fixture。"""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """用于创建可针对不同测试场景配置的 mock agent 的 fixture。"""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
    )
    agent_mock.aget_state = AsyncMock(
        return_value=StateSnapshot(
            values={},
            next=(),
            config={},
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )
    )
    # 涉及 checkpointer 读取的测试会显式设置此项；否则 AsyncMock 会
    # 自动创建一个看起来像可用 checkpointer 的属性。
    agent_mock.checkpointer = None
    with patch("service.service.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_settings(mock_env):
    """用于确保每个测试的 settings 都是干净的 fixture。"""
    with patch("service.service.settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_httpx():
    """将 httpx.stream 和 httpx.get 打补丁以使用我们的测试客户端。

    sync `stream`（`AgentClient`）并轨后委托给 `astream`，走 `httpx.AsyncClient`，
    所以这里也必须把 AsyncClient 指向同一个 app（ASGI 传输层）——
    否则 `client.stream(...)` 会变成对 `http://0.0.0.0` 的真实请求。
    """

    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # 去掉 base URL，因为 TestClient 只期望路径
            path = url.replace("http://0.0.0.0", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # 去掉 base URL，因为 TestClient 只期望路径
            path = url.replace("http://0.0.0.0", "")
            return client.get(path, **kwargs)

        @asynccontextmanager
        async def mock_astream(self, method: str, url: str, **kwargs):
            # ASGITransport 直接把完整 URL 交给 app；请求体一次装载完整，
            # 避免半途 break 时残留未关闭的流式生成器。
            kwargs.pop("timeout", None)
            request = httpx.Request(method, url, **kwargs)
            response = await ASGITransport(app=app).handle_async_request(request)
            response.request = request
            yield response

        with patch("httpx.stream", mock_stream):
            with patch("httpx.get", mock_get):
                with patch("httpx.AsyncClient.stream", mock_astream):
                    yield
