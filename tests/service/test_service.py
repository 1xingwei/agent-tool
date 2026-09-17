import json
from unittest.mock import AsyncMock, patch

import langsmith
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Interrupt, StateSnapshot

from agents.agents import Agent
from schema import ChatHistory, ChatMessage, ServiceMetadata
from schema.models import AnthropicModelName, OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """测试 /invoke 能配合自定义 agent_id 路径参数工作。"""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # 为默认 agent 创建一个单独的 mock
    default_mock = AsyncMock()
    default_mock.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})
    ]

    # 配置我们的自定义 mock agent
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})]

    # patch get_agent，使其根据提供的 agent_id 返回正确的 agent
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # 验证自定义 agent 被调用，而默认 agent 未被调用
        mock_agent.ainvoke.assert_awaited_once()
        default_mock.ainvoke.assert_not_awaited()

        input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # 验证我们得到了自定义 agent 的响应


def test_invoke_model_param(test_client, mock_agent) -> None:
    """测试如果指定了 model 参数，它会被正确地传递给 agent。"""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = OpenAIModelName.GPT_5_MINI
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # 验证 model 在 config 中被正确传递
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # 验证响应仍然正确
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # 验证配置的允许列表之外的有效枚举值返回 400。
    unavailable_model = AnthropicModelName.SONNET_45
    response = test_client.post("/invoke", json={"message": QUESTION, "model": unavailable_model})
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]

    # 验证格式错误的 model 字符串仍然无法通过请求校验。
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_no_model_param_uses_none_default(test_client, mock_agent) -> None:
    """测试当未指定 model 时，UserInput 默认为 None，且不会被传递给 runnable config（而不是硬编码为 gpt-5-nano）。"""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    # 不在请求中指定 model
    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert "model" not in config["configurable"]  # 为 None 时不应存在

    # 验证响应仍然正确
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """测试 agent_config 参数被正确地传递给 agent。"""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG}
    )
    assert response.status_code == 200

    # 验证 agent_config 在 config 中被正确传递
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # 验证响应仍然正确
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # 验证 agent_config 中的保留键会抛出校验错误
    INVALID_CONFIG = {"model": "gpt-5-nano"}
    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG}
    )
    assert response.status_code == 422


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=ANSWER)]}),
        ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
    ]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client, test_client, monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_feedback_without_langsmith_key(test_client) -> None:
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post(
        "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
    )
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


def test_history_for_unknown_thread_returns_empty(test_client, mock_agent) -> None:
    """从未写入过的 thread 会返回 `values={}`；读取它不能返回 500。

    `mock_agent` fixture 已经从 `aget_state` 返回 `values={}`，这正是过去在
    handler 内部抛出 KeyError('messages') 的那种形状。
    """
    response = test_client.post("/history", json={"thread_id": "never-written-thread"})

    assert response.status_code == 200
    assert response.json() == {"messages": []}


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/no-such-agent/invoke", {"json": {"message": "hi"}}),
        ("post", "/no-such-agent/stream", {"json": {"message": "hi"}}),
        ("post", "/no-such-agent/history", {"json": {"thread_id": "t"}}),
        # UserThreadsInput 要求 user_id，因此裸 GET 会在 agent id 被解析之前就返回 422
        # ——这会让这个用例因为错误的原因而通过。
        ("get", "/no-such-agent/threads", {"params": {"user_id": "u"}}),
    ],
)
def test_unknown_agent_returns_404(test_client, method, path, kwargs) -> None:
    """未注册的 agent id 必须是 404，而不是裸 500，也不是静默的 200。

    `/stream` 是这里采用参数化而非拆分的原因：它过去会返回
    200 和空 body，因此必须与其他用例由同一断言覆盖。

    有意不使用 `mock_agent` fixture——它会 patch
    `service.service.get_agent`，而这个用例需要真实的查找来抛出 KeyError。
    """
    response = getattr(test_client, method)(path, **kwargs)

    assert response.status_code == 404
    assert "no-such-agent" in response.json()["detail"]


def test_history_custom_agent(test_client) -> None:
    """测试 /{agent_id}/history 通过所请求 agent 的图读取 thread。"""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."

    custom_snapshot = StateSnapshot(
        values={"messages": [HumanMessage(content=QUESTION), AIMessage(content=ANSWER)]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
    # 默认 agent 的图不知道这个 thread，因此不返回任何消息。
    default_snapshot = StateSnapshot(
        values={"messages": []},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    custom_mock = AsyncMock()
    custom_mock.aget_state.return_value = custom_snapshot
    custom_mock.checkpointer = None
    default_mock = AsyncMock()
    default_mock.aget_state.return_value = default_snapshot
    default_mock.checkpointer = None

    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return custom_mock
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(
            f"/{CUSTOM_AGENT}/history",
            json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"},
        )
        assert response.status_code == 200

        # 使用的是自定义 agent 的图，而不是默认的。
        custom_mock.aget_state.assert_awaited_once()
        default_mock.aget_state.assert_not_awaited()

        output = ChatHistory.model_validate(response.json())
        assert output.messages[0].type == "human"
        assert output.messages[0].content == QUESTION
        assert output.messages[1].type == "ai"
        assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """测试流式 token 和消息。"""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # 配置 mock 以使用我们的异步迭代器函数
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # 发起带流式的请求
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": True}
    ) as response:
        assert response.status_code == 200

        # 收集所有 SSE 消息
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # 跳过 [DONE] 消息
                messages.append(json.loads(line.lstrip("data: ")))

        # 验证流式 token
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # 验证最终消息
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """测试不带 token 的流式。"""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # 配置 mock 以使用我们的异步迭代器函数
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # 发起禁用流式的请求
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # 收集所有 SSE 消息
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # 跳过 [DONE] 消息
                messages.append(json.loads(line.lstrip("data: ")))

        # 验证没有 token 消息
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # 验证最终消息
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"]["content"] == FINAL_ANSWER
        assert messages[0]["content"]["type"] == "ai"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # 配置 mock 以使用我们的异步迭代器函数
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # 发起禁用流式的请求
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # 收集所有 SSE 消息
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # 跳过 [DONE] 消息
                messages.append(json.loads(line.lstrip("data: ")))

        # 验证中断消息
        assert len(messages) == 1
        assert messages[0]["content"]["content"] == INTERRUPT
        assert messages[0]["content"]["type"] == "ai"


def test_info(test_client, mock_settings) -> None:
    """测试 /info 返回正确的服务元数据。"""

    base_agent = Agent(description="A base agent.", graph_like=None)
    mock_settings.AUTH_SECRET = None
    mock_settings.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
    mock_settings.AVAILABLE_MODELS = {OpenAIModelName.GPT_5_NANO, OpenAIModelName.GPT_5_MINI}
    with patch.dict("agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "research-assistant"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_5_NANO
    assert output.models == [OpenAIModelName.GPT_5_MINI, OpenAIModelName.GPT_5_NANO]
