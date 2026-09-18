import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Request, Response

from client import AgentClient, AgentClientError
from schema import AgentInfo, ChatHistory, ChatMessage, ServiceMetadata, UserThreads
from schema.models import OpenAIModelName


def test_init(mock_env):
    """测试使用不同参数初始化客户端。"""
    # 测试默认值
    client = AgentClient(get_info=False)
    assert client.base_url == "http://0.0.0.0"
    assert client.timeout is None

    # 测试自定义值
    client = AgentClient(
        base_url="http://test",
        timeout=30.0,
        get_info=False,
    )
    assert client.base_url == "http://test"
    assert client.timeout == 30.0
    client.update_agent("test-agent", verify=False)
    assert client.agent == "test-agent"


def test_headers(mock_env):
    """测试带认证和不带认证时的请求头生成。"""
    # 测试不带认证
    client = AgentClient(get_info=False)
    assert client._headers == {}

    # 测试带认证
    with patch.dict(os.environ, {"AUTH_SECRET": "test-secret"}, clear=True):
        client = AgentClient(get_info=False)
        assert client._headers == {"Authorization": "Bearer test-secret"}


def test_invoke(agent_client):
    """并轨后同步方法只是薄委托：断言转发，不重复测 HTTP。"""
    with patch.object(agent_client, "ainvoke", new=AsyncMock(return_value="R")) as m:
        assert agent_client.invoke("q", model="m", thread_id="t", user_id="u") == "R"
    m.assert_awaited_once_with("q", model="m", thread_id="t", user_id="u", agent_config=None)


@pytest.mark.asyncio
async def test_ainvoke(agent_client):
    """测试异步调用。"""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # 测试成功响应
    mock_request = Request("POST", "http://test/invoke")
    mock_response = Response(200, json={"type": "ai", "content": ANSWER}, request=mock_request)
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        response = await agent_client.ainvoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # 测试带 model 和 thread_id
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        response = await agent_client.ainvoke(
            QUESTION,
            model="gpt-5-nano",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER
        # 验证请求
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-5-nano"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # 测试错误响应
    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch("httpx.AsyncClient.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await agent_client.ainvoke(QUESTION)
        assert "500 Internal Server Error" in str(exc.value)


def test_stream(agent_client):
    """并轨后同步流只是薄委托：断言逐项转发到 astream。"""

    async def fake_astream(*args, **kwargs):
        yield "tok1"
        yield "tok2"

    with patch.object(agent_client, "astream", new=fake_astream):
        assert list(agent_client.stream("q")) == ["tok1", "tok2"]


def test_sync_bridge_propagates_error(agent_client):
    """同步桥必须原样透传异步层抛出的 AgentClientError。

    asyncio.run 分支：_run_sync 把 coro 的异常原样复抛给调用方。
    """
    with patch.object(agent_client, "ainvoke", new=AsyncMock(side_effect=AgentClientError("boom"))):
        with pytest.raises(AgentClientError) as exc:
            agent_client.invoke("q")
    assert str(exc.value) == "boom"


def test_sync_stream_propagates_error(agent_client):
    """同步流桥接也必须原样透传 astream 的 AgentClientError。"""

    async def boom_astream(*args, **kwargs):
        raise AgentClientError("boom")
        yield

    with patch.object(agent_client, "astream", new=boom_astream):
        with pytest.raises(AgentClientError) as exc:
            list(agent_client.stream("q"))
    assert str(exc.value) == "boom"


@pytest.mark.asyncio
async def test_sync_call_inside_running_loop(agent_client):
    """守卫 streamlit_app.py 的嵌套场景。

    反例：把 _run_sync 的 ThreadPoolExecutor 分支去掉，本用例必须变红
    （asyncio.run 在运行中的 loop 内必抛 RuntimeError）。
    """
    with patch.object(agent_client, "aget_user_threads", new=AsyncMock(return_value="T")):
        assert agent_client.get_user_threads("u") == "T"


@pytest.mark.asyncio
async def test_sync_error_inside_running_loop(agent_client):
    """线程分支的错误透传：loop 内同步调用遇到异步异常同样原样复抛。"""
    with patch.object(
        agent_client, "aget_user_threads", new=AsyncMock(side_effect=AgentClientError("boom"))
    ):
        with pytest.raises(AgentClientError) as exc:
            agent_client.get_user_threads("u")
    assert str(exc.value) == "boom"


@pytest.mark.asyncio
async def test_sync_stream_inside_running_loop(agent_client):
    """守卫同步流桥在运行中 loop 内可用（docs/18 N2）。

    反例：去掉 _iterate_sync 的工作线程分支，本用例必须变红
    （run_until_complete 在运行中的 loop 内必抛 RuntimeError）。
    """

    async def fake_astream(*args, **kwargs):
        yield "tok"

    with patch.object(agent_client, "astream", new=fake_astream):
        assert list(agent_client.stream("q")) == ["tok"]


@pytest.mark.asyncio
async def test_astream(agent_client):
    """测试异步流式。"""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # 创建带流式事件的模拟响应
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"
        ]
        + ["data: [DONE]"]
    )

    # 为事件创建异步迭代器
    async def async_events():
        for event in events:
            yield event

    # 模拟流式响应
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.raise_for_status = Mock()
    mock_response.aiter_lines = Mock(return_value=async_events())
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = Mock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # 收集所有流式响应
        responses = []
        async for response in agent_client.astream(QUESTION):
            responses.append(response)

        # 验证 token 已被流式输出
        assert len(responses) == len(TOKENS) + 1  # token + 最终消息
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # 验证最终消息
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # 测试错误响应
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/stream")
    )
    error_response_mock = AsyncMock()
    error_response_mock.__aenter__ = AsyncMock(return_value=error_response)

    mock_client.stream.return_value = error_response_mock

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(AgentClientError) as exc:
            async for _ in agent_client.astream(QUESTION):
                pass
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_acreate_feedback(agent_client):
    """测试异步反馈创建。"""
    RUN_ID = "test-run"
    KEY = "test-key"
    SCORE = 0.8
    KWARGS = {"comment": "Great response!"}

    # 测试成功响应
    mock_response = Response(200, json={}, request=Request("POST", "http://test/feedback"))
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        await agent_client.acreate_feedback(RUN_ID, KEY, SCORE, KWARGS)
        # 验证请求
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["run_id"] == RUN_ID
        assert kwargs["json"]["key"] == KEY
        assert kwargs["json"]["score"] == SCORE
        assert kwargs["json"]["kwargs"] == KWARGS

    # 测试错误响应
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/feedback")
    )
    with patch("httpx.AsyncClient.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await agent_client.acreate_feedback(RUN_ID, KEY, SCORE)
        assert "500 Internal Server Error" in str(exc.value)


def test_get_history(agent_client):
    """测试聊天历史检索。"""
    THREAD_ID = "test-thread"
    HISTORY = {
        "messages": [
            {"type": "human", "content": "What is the weather?"},
            {"type": "ai", "content": "The weather is sunny."},
        ]
    }

    # 模拟成功响应——默认使用客户端选中的 agent
    mock_response = Response(200, json=HISTORY, request=Request("POST", "http://test/history"))
    with patch("httpx.post", return_value=mock_response) as mock_post:
        history = agent_client.get_history(THREAD_ID)
        assert isinstance(history, ChatHistory)
        assert len(history.messages) == 2
        assert history.messages[0].type == "human"
        assert history.messages[1].type == "ai"
        # 客户端选中的 agent 用于限定历史请求的范围
        assert mock_post.call_args.args[0] == "http://test/test-agent/history"

    # 显式指定的 agent 会覆盖客户端选中的 agent
    with patch("httpx.post", return_value=mock_response) as mock_post:
        agent_client.get_history(THREAD_ID, agent="chatbot")
        assert mock_post.call_args.args[0] == "http://test/chatbot/history"

    # 测试错误响应
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/history")
    )
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            agent_client.get_history(THREAD_ID)
        assert "500 Internal Server Error" in str(exc.value)


def test_info(agent_client):
    assert agent_client.info is None
    assert agent_client.agent == "test-agent"

    # 模拟 info 响应
    test_info = ServiceMetadata(
        default_agent="custom-agent",
        agents=[AgentInfo(key="custom-agent", description="Custom agent")],
        default_model=OpenAIModelName.GPT_5_NANO,
        models=[OpenAIModelName.GPT_5_NANO, OpenAIModelName.GPT_5_MINI],
    )
    test_response = Response(
        200, json=test_info.model_dump(), request=Request("GET", "http://test/info")
    )

    # 用 info 更新已有客户端
    with patch("httpx.get", return_value=test_response):
        agent_client.retrieve_info()

    assert agent_client.info == test_info
    assert agent_client.agent == "custom-agent"

    # 测试无效的 update_agent
    with pytest.raises(AgentClientError) as exc:
        agent_client.update_agent("unknown-agent")
    assert "Agent unknown-agent not found in available agents: custom-agent" in str(exc.value)

    # 测试带 info 的全新客户端
    with patch("httpx.get", return_value=test_response):
        agent_client = AgentClient(base_url="http://test")
    assert agent_client.info == test_info
    assert agent_client.agent == "custom-agent"

    # 测试未设置 agent 时调用报错
    agent_client = AgentClient(base_url="http://test", get_info=False)
    with pytest.raises(AgentClientError) as exc:
        agent_client.invoke("test")
    assert "No agent selected. Use update_agent() to select an agent." in str(exc.value)


def test_get_user_threads(agent_client):
    """并轨后同步方法只是薄委托：断言转发到 aget_user_threads。"""
    with patch.object(agent_client, "aget_user_threads", new=AsyncMock(return_value="T")) as m:
        assert agent_client.get_user_threads("u", agent="custom-agent", limit=50) == "T"
    m.assert_awaited_once_with("u", agent="custom-agent", limit=50)


@pytest.mark.asyncio
async def test_aget_user_threads(agent_client):
    """测试异步用户 thread 检索。"""
    USER_ID = "user-123"

    mock_request = Request("GET", "http://test/test-agent/threads")
    mock_response = Response(
        200,
        json={
            "threads": [
                {
                    "thread_id": "thread-1",
                    "agent_id": "test-agent",
                    "updated_at": datetime.now(UTC).isoformat(),
                    "title": "First Chat",
                }
            ]
        },
        request=mock_request,
    )

    with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
        result = await agent_client.aget_user_threads(USER_ID, agent="custom-agent", limit=50)

        assert isinstance(result, UserThreads)
        assert result.threads[0].thread_id == "thread-1"

        args, kwargs = mock_get.call_args
        assert args[0] == "http://test/custom-agent/threads"
        assert kwargs["params"] == {"user_id": USER_ID, "limit": 50}

    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch("httpx.AsyncClient.get", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await agent_client.aget_user_threads(USER_ID)
        assert "500 Internal Server Error" in str(exc.value)
