# AppTest 默认的 3s 超时在整套测试负载下过于紧张：进程中
# 最先运行 AppTest 的测试还要承担 Streamlit 一次性的组件发现开销。
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClientError
from schema import ChatHistory, ChatMessage, ThreadSummary, UserThreads
from schema.models import OpenAIModelName


def test_app_simple_non_streaming(mock_agent_client):
    """测试完整应用——正常路径"""
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    WELCOME_START = "你好！我是一个 AI Agent，有什么想问的尽管说！"
    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[0].markdown[0].value.startswith(WELCOME_START)

    at.sidebar.toggle[0].set_value(False)  # 使用 Streaming = False
    at.chat_input[0].set_value(PROMPT).run()
    print(at)
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE
    assert not at.exception


def test_app_settings(mock_agent_client):
    """测试完整应用——正常路径"""
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["user_id"] = "1234"
    at.run()

    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    at.sidebar.toggle[0].set_value(False)  # 使用 Streaming = False
    assert at.sidebar.selectbox[0].value == "gpt-5-nano"
    assert mock_agent_client.agent == "test-agent"
    at.sidebar.selectbox[0].set_value("gpt-5-mini")
    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    # 基本检查
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE

    # 检查参数是否与设置匹配
    assert mock_agent_client.agent == "chatbot"
    mock_agent_client.ainvoke.assert_called_with(
        message=PROMPT,
        model=OpenAIModelName.GPT_5_MINI,
        thread_id=at.session_state.thread_id,
        user_id="1234",
    )
    assert not at.exception


def test_app_thread_id_history(mock_agent_client):
    """测试 thread_id 是否生成"""

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    # 重置并设置 thread_id
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["thread_id"] = "1234"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    # URL 中没有 agent，因此历史记录通过客户端选定的 agent 读取。
    mock_agent_client.get_history.assert_called_with(thread_id="1234", agent="test-agent")
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_resume_with_agent_param(mock_agent_client):
    """?agent= URL 参数将恢复的历史记录限定到该 agent 的图。"""

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["thread_id"] = "1234"
    at.query_params["agent"] = "chatbot"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    # 历史记录通过 URL 中指定的 agent 获取，而非默认 agent。
    mock_agent_client.get_history.assert_called_with(thread_id="1234", agent="chatbot")
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_feedback(mock_agent_client):
    """TODO：无法弄清楚如何与 st.feedback 交互"""

    pass


@pytest.mark.asyncio
async def test_app_streaming(mock_agent_client):
    """测试启用流式传输的应用——包括工具消息"""
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    # 设置模拟流式响应
    PROMPT = "What is 6 * 7?"
    ai_with_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "calculator", "id": "test_call_id", "args": {"expression": "6 * 7"}}],
    )
    tool_message = ChatMessage(type="tool", content="42", tool_call_id="test_call_id")
    final_ai_message = ChatMessage(type="ai", content="The answer is 42")

    messages = [ai_with_tool, tool_message, final_ai_message]

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # 使用 Streaming = True
    at.toggle[2].set_value(True)  # 显示工具调用（默认隐藏）
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    response = at.chat_message[1]
    tool_status = response.status[0]
    assert response.avatar == "assistant"
    assert tool_status.label == "🛠️ 工具调用：calculator"
    assert tool_status.icon == ":material/check:"
    assert tool_status.markdown[0].value == "输入："
    assert tool_status.json[0].value == '{"expression": "6 * 7"}'
    assert tool_status.markdown[1].value == "输出："
    assert tool_status.markdown[2].value == "42"
    assert response.markdown[-1].value == "The answer is 42"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_hides_tool_calls_by_default(mock_agent_client):
    """工具调用属于实现细节，不会出现在对话记录中。"""
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    PROMPT = "What is 6 * 7?"
    ai_with_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "calculator", "id": "test_call_id", "args": {"expression": "6 * 7"}}],
    )
    tool_message = ChatMessage(type="tool", content="42", tool_call_id="test_call_id")
    final_ai_message = ChatMessage(type="ai", content="The answer is 42")

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in [ai_with_tool, tool_message, final_ai_message]:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # 使用 Streaming = True，保持工具调用隐藏
    at.chat_input[0].set_value(PROMPT).run()

    response = at.chat_message[1]
    assert len(response.status) == 0, "Tool calls must not be rendered by default"
    assert response.markdown[-1].value == "The answer is 42"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_aggregates_multiple_tool_calls(mock_agent_client):
    """一轮中的多次调用会折叠为单个容器，而非各自一个。"""
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    PROMPT = "Compare two things"
    ai_with_tools = ChatMessage(
        type="ai",
        content="",
        tool_calls=[
            {"name": "web_search", "id": "call-1", "args": {"query": "a"}},
            {"name": "web_search", "id": "call-2", "args": {"query": "b"}},
        ],
    )
    first_result = ChatMessage(type="tool", content="result 1", tool_call_id="call-1")
    second_result = ChatMessage(type="tool", content="result 2", tool_call_id="call-2")
    final_ai_message = ChatMessage(type="ai", content="Here is the comparison")

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in [ai_with_tools, first_result, second_result, final_ai_message]:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # 使用 Streaming = True
    at.toggle[2].set_value(True)  # 显示工具调用
    at.chat_input[0].set_value(PROMPT).run()

    response = at.chat_message[1]
    assert len(response.status) == 1, "Tool calls in one turn should share a container"
    assert response.status[0].label == "🛠️ 工具调用（2）"
    assert response.markdown[-1].value == "Here is the comparison"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_hides_sub_agent_tools_by_default(mock_agent_client):
    """子 agent 转移及其内部的工具调用同样会被隐藏。

    这些消息通过 handle_sub_agent_msgs() 到达，是与顶层工具调用不同的渲染路径，
    因此需要单独覆盖。
    """
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    PROMPT = "Research this for me"
    transfer = ChatMessage(
        type="ai",
        content="",
        tool_calls=[
            {
                "name": "transfer_to_research_expert",
                "id": "transfer-1",
                "args": {"task": "research"},
            }
        ],
    )
    transfer_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_research_expert",
        tool_call_id="transfer-1",
    )
    sub_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "web_search", "id": "search-1", "args": {"query": "secret"}}],
    )
    sub_tool_result = ChatMessage(type="tool", content="search results", tool_call_id="search-1")
    transfer_back = ChatMessage(
        type="ai",
        content="",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-1",
                "args": {"result": "done"},
            }
        ],
    )
    transfer_back_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-1",
    )
    final = ChatMessage(type="ai", content="Here is the research summary")

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in [
            transfer,
            transfer_success,
            sub_tool,
            sub_tool_result,
            transfer_back,
            transfer_back_success,
            final,
        ]:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # 使用 Streaming = True，保持工具调用隐藏
    at.chat_input[0].set_value(PROMPT).run()

    response = at.chat_message[1]
    assert len(response.status) == 0, "Sub-agent transfers must not be rendered by default"
    assert response.markdown[-1].value == "Here is the research summary"

    transcript = " ".join(m.value for m in response.markdown)
    assert "transfer_to" not in transcript, "Sub-agent transfers must not leak into the transcript"
    assert "web_search" not in transcript, "Tool names must not leak into the transcript"
    assert "search results" not in transcript, "Tool output must not leak into the transcript"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_init_error(mock_agent_client):
    """测试 agent 初始化出错时的应用"""
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    # 设置模拟流式响应
    PROMPT = "What is 6 * 7?"
    mock_agent_client.astream.side_effect = AgentClientError("Error connecting to agent")

    at.toggle[0].set_value(True)  # 使用 Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[1].avatar == "user"
    assert at.chat_message[1].markdown[0].value == PROMPT
    assert at.error[0].value == "生成回复时出错：Error connecting to agent"
    assert not at.exception


def test_app_new_chat_btn(mock_agent_client):
    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()
    thread_id_a = at.session_state.thread_id

    at.sidebar.button[0].click().run()

    assert at.session_state.thread_id != thread_id_a
    assert not at.exception


@pytest.fixture
def multi_agent_messages():
    """为多 agent 测试提供可复用消息的 fixture"""
    from schema import ChatMessage

    # 工具 1
    tool_1 = ChatMessage(
        type="ai",
        content="Starting tool 1...",
        tool_calls=[{"name": "do_work_1", "id": "tool-1", "args": {"my-arg": "value"}}],
    )
    tool_1_result = ChatMessage(type="tool", content="Tool 1 complete", tool_call_id="tool-1")

    # 工具 2
    tool_2 = ChatMessage(
        type="ai",
        content="Starting tool 2...",
        tool_calls=[{"name": "do_work_2", "id": "tool-2", "args": {"my-arg-2": "value"}}],
    )
    tool_2_result = ChatMessage(type="tool", content="Tool 2 complete", tool_call_id="tool-2")

    # 转移到 agent A
    transfer_a = ChatMessage(
        type="ai",
        content="Transferring to agent A...",
        tool_calls=[
            {"name": "transfer_to_agent_a", "id": "transfer-a", "args": {"task": "task_1"}}
        ],
    )
    transfer_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_a",
        tool_call_id="transfer-a",
    )

    # Agent A 转移到 agent B（子 agent）
    transfer_b_from_a = ChatMessage(
        type="ai",
        content="Agent A delegating to agent B...",
        tool_calls=[
            {"name": "transfer_to_agent_b", "id": "transfer-a-b", "args": {"sub_task": "task_2"}}
        ],
    )
    transfer_b_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_b",
        tool_call_id="transfer-a-b",
    )

    # Agent B 转移回 A
    transfer_back_b = ChatMessage(
        type="ai",
        content="Agent B finished.",
        tool_calls=[
            {"name": "transfer_back_to_agent_a", "id": "back-b-a", "args": {"result": "result_2"}}
        ],
    )
    transfer_back_b_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_agent_a",
        tool_call_id="back-b-a",
    )

    # Agent A 转移回 supervisor
    transfer_back_a = ChatMessage(
        type="ai",
        content="Agent A finished.",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-a-super",
                "args": {"result": "result_1"},
            }
        ],
    )
    transfer_back_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-a-super",
    )

    # Supervisor 继续并转移到 agent C（A 的兄弟节点）
    supervisor_continues = ChatMessage(
        type="ai",
        content="Now transferring to agent C...",
        tool_calls=[
            {"name": "transfer_to_agent_c", "id": "transfer-c", "args": {"task": "task_3"}}
        ],
    )
    transfer_c_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_c",
        tool_call_id="transfer-c",
    )

    # Agent C 转移回来
    transfer_back_c = ChatMessage(
        type="ai",
        content="Agent C finished.",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-c-super",
                "args": {"result": "result_3"},
            }
        ],
    )
    transfer_back_c_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-c-super",
    )

    # 最终响应
    supervisor_final = ChatMessage(
        type="ai", content="All agents have completed their tasks successfully."
    )

    return {
        "tool_1": tool_1,
        "tool_1_result": tool_1_result,
        "tool_2": tool_2,
        "tool_2_result": tool_2_result,
        "transfer_a": transfer_a,
        "transfer_a_success": transfer_a_success,
        "transfer_b_from_a": transfer_b_from_a,
        "transfer_b_success": transfer_b_success,
        "transfer_back_b": transfer_back_b,
        "transfer_back_b_success": transfer_back_b_success,
        "transfer_back_a": transfer_back_a,
        "transfer_back_a_success": transfer_back_a_success,
        "supervisor_continues": supervisor_continues,
        "transfer_c_success": transfer_c_success,
        "transfer_back_c": transfer_back_c,
        "transfer_back_c_success": transfer_back_c_success,
        "supervisor_final": supervisor_final,
    }


@pytest.mark.asyncio
async def test_app_streaming_single_sub_agent(mock_agent_client, multi_agent_messages):
    """测试单个子 agent 的多次工具调用，以验证弹出框功能"""

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    PROMPT = "Test single sub-agent with multiple tools"

    # 使用 fixture 并包含多个工作工具以测试多个弹出框
    # Supervisor -> Agent A（使用 tool_1 和 tool_2）-> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.toggle[2].set_value(True)  # 显示工具调用（默认隐藏）
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message"
    )

    status_agent = ai_message.status[0]
    assert status_agent == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_agent.label

    assert status_agent.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )

    popover_1 = status_agent.children[1]
    assert hasattr(popover_1, "type") and popover_1.type == "popover", (
        "Second child of status should be a popover for the first tool call"
    )
    assert popover_1.proto.popover.label == "do_work_1"
    assert popover_1.proto.popover.icon == "🛠️"
    assert popover_1.markdown[0].value == "**工具：**do_work_1"
    assert popover_1.markdown[1].value == "**输入：**"
    assert '"my-arg": "value"' in popover_1.json[0].value
    assert popover_1.markdown[2].value == "**输出：**"
    assert popover_1.markdown[3].value == "Tool 1 complete"

    assert status_agent.children[2].value == "Starting tool 2...", (
        "Third child of status should be tool 2 message"
    )

    popover_2 = status_agent.children[3]
    assert hasattr(popover_2, "type") and popover_2.type == "popover", (
        "Fourth child of the status should be a popover for the second tool call"
    )
    assert popover_2.proto.popover.label == "do_work_2"

    assert not at.exception


@pytest.mark.asyncio
async def test_app_streaming_sequential_sub_agents(mock_agent_client, multi_agent_messages):
    """测试 supervisor agent 转移到子 agent A，再返回 supervisor，然后转移到子 agent C，再返回的情况"""

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    PROMPT = "Test multiple transfer back patterns"

    # 为顺序 agent 创建消息流：Supervisor -> Agent A（使用 tool 1）-> Supervisor -> Agent C（使用 tool 2）-> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_continues"],
            messages["transfer_c_success"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_c"],
            messages["transfer_back_c_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.toggle[2].set_value(True)  # 显示工具调用（默认隐藏）
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message to agent A"
    )

    status_a = ai_message.status[0]
    assert status_a == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_a.label

    assert status_a.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )
    # status 的第二个子元素应为第一次工具调用的弹出框
    popover_a = status_a.children[1]
    assert popover_a.type == "popover"
    assert popover_a.proto.popover.label == "do_work_1"
    assert popover_a.proto.popover.icon == "🛠️"
    assert popover_a.markdown[0].value == "**工具：**do_work_1"
    assert popover_a.markdown[1].value == "**输入：**"
    assert popover_a.json[0].value == '{"my-arg": "value"}'
    assert popover_a.markdown[2].value == "**输出：**"
    assert popover_a.markdown[3].value == "Tool 1 complete"

    assert ai_message.children[2].value == "Now transferring to agent C...", (
        "Third child should be transfer message to agent C"
    )

    status_c = ai_message.status[1]
    assert status_c == ai_message.children[3], "Fourth child should be the second status"
    assert "transfer_to_agent_c" in status_c.label

    assert status_c.children[0].value == "Starting tool 2...", (
        "First child of next status should be tool 2 message"
    )
    popover_c = status_c.children[1]
    assert popover_c.type == "popover"
    assert popover_c.proto.popover.label == "do_work_2"
    assert popover_c.proto.popover.icon == "🛠️"
    assert popover_c.markdown[0].value == "**工具：**do_work_2"
    assert popover_c.markdown[1].value == "**输入：**"
    assert popover_c.json[0].value == '{"my-arg-2": "value"}'
    assert popover_c.markdown[2].value == "**输出：**"
    assert popover_c.markdown[3].value == "Tool 2 complete"

    assert ai_message.children[4].value == "All agents have completed their tasks successfully.", (
        "Fifth child should be final supervisor message"
    )

    assert len(ai_message.children) == 6, (
        f"Should have 6 children: transfer to a, status for a, transfer to c, status for c, final message, feedback stars - got {len(ai_message.children)}"
    )

    assert not at.exception


@pytest.mark.asyncio
async def test_app_streaming_nested_sub_agents(mock_agent_client, multi_agent_messages):
    """测试嵌套子 agent，其中 agent B 是 agent A 的子 agent"""

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10).run()

    PROMPT = "Test nested sub-agents"

    # 为嵌套子 agent 创建消息流：Supervisor -> Agent A（使用 tool 1）-> Agent B（使用 tool 2）-> Agent A -> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["transfer_b_from_a"],
            messages["transfer_b_success"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_b"],
            messages["transfer_back_b_success"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.toggle[2].set_value(True)  # 显示工具调用（默认隐藏）
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message to agent A"
    )

    status_a = ai_message.status[0]
    assert status_a == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_a.label

    assert status_a.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )
    # status 的第二个子元素应为第一次工具调用的弹出框
    popover_a = status_a.children[1]
    assert popover_a.type == "popover"
    assert popover_a.proto.popover.label == "do_work_1"
    assert popover_a.proto.popover.icon == "🛠️"
    assert popover_a.markdown[0].value == "**工具：**do_work_1"
    assert popover_a.markdown[1].value == "**输入：**"
    assert popover_a.json[0].value == '{"my-arg": "value"}'
    assert popover_a.markdown[2].value == "**输出：**"
    assert popover_a.markdown[3].value == "Tool 1 complete"

    assert status_a.children[2].value == "Agent A delegating to agent B...", (
        "Third child of status should be transfer message to agent B"
    )

    # status 的第四个子元素应为 Agent B 的嵌套 status
    nested_status_b = status_a.children[3]
    assert "transfer_to_agent_b" in nested_status_b.label

    assert nested_status_b.children[0].value == "Starting tool 2...", (
        "First child of nested status should be tool 2 message"
    )
    # 嵌套 status 的第二个子元素应为 task 2 工具调用的弹出框
    popover_b = nested_status_b.children[1]
    assert popover_b.type == "popover"
    assert popover_b.proto.popover.label == "do_work_2"
    assert popover_b.proto.popover.icon == "🛠️"
    assert popover_b.markdown[0].value == "**工具：**do_work_2"
    assert popover_b.markdown[1].value == "**输入：**"
    assert popover_b.json[0].value == '{"my-arg-2": "value"}'
    assert popover_b.markdown[2].value == "**输出：**"
    assert popover_b.markdown[3].value == "Tool 2 complete"

    assert ai_message.children[2].value == "All agents have completed their tasks successfully.", (
        "Third child should be final supervisor message"
    )

    assert len(ai_message.children) == 4, (
        f"Should have 4 children: transfer to a, status for a (with nested b), final message, feedback stars - got {len(ai_message.children)}"
    )

    assert not at.exception


@pytest.fixture
def mock_threads_data():
    """为缓存测试提供虚拟 thread 数据的 fixture。"""
    return UserThreads(
        threads=[
            ThreadSummary(
                thread_id="thread-1111-2222",
                agent_id="test-agent",
                updated_at="2026-07-31T20:14:19.804150+00:00",
                title="What is Python?",
            )
        ]
    )


def test_app_thread_caching_sidebar(mock_agent_client, mock_threads_data):
    """验证 thread 列表通过 get_user_threads 获取并渲染在侧边栏历史记录中。"""
    mock_agent_client.get_user_threads = Mock(return_value=mock_threads_data)

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["user_id"] = "user-123"
    at.run()

    mock_agent_client.get_user_threads.assert_called_with(
        user_id="user-123", agent="test-agent", limit=20
    )

    sidebar_buttons = [b.label for b in at.sidebar.button]
    assert "What is Python?" in sidebar_buttons
    assert not at.exception


def test_app_thread_click_loads_history(mock_agent_client, mock_threads_data):
    """验证点击侧边栏 thread 会将该对话加载到聊天中。"""
    mock_agent_client.get_user_threads = Mock(return_value=mock_threads_data)
    mock_agent_client.get_history = Mock(
        return_value=ChatHistory(
            messages=[
                ChatMessage(type="human", content="What is Python?"),
                ChatMessage(type="ai", content="A programming language."),
            ]
        )
    )

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["user_id"] = "user-123"
    at.run()

    at.button(key="thread_thread-1111-2222").click().run()

    mock_agent_client.get_history.assert_called_with(
        thread_id="thread-1111-2222", agent="test-agent"
    )
    assert at.session_state.thread_id == "thread-1111-2222"
    assert [m.content for m in at.session_state.messages] == [
        "What is Python?",
        "A programming language.",
    ]
    assert not at.exception


def test_app_thread_click_history_error(mock_agent_client, mock_threads_data):
    """验证历史记录获取失败时会显示错误，且不影响当前聊天。"""
    mock_agent_client.get_user_threads = Mock(return_value=mock_threads_data)
    mock_agent_client.get_history = Mock(side_effect=AgentClientError("service down"))

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["user_id"] = "user-123"
    at.run()
    original_thread_id = at.session_state.thread_id

    at.button(key="thread_thread-1111-2222").click().run()

    assert any("无法加载该对话。" in error.value for error in at.error)
    assert at.session_state.thread_id == original_thread_id
    assert not at.exception


def test_app_thread_fetch_error_shows_caption(mock_agent_client):
    """验证 threads 端点失败时侧边栏能优雅降级。"""
    mock_agent_client.get_user_threads = Mock(side_effect=AgentClientError("service down"))

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=10)
    at.query_params["user_id"] = "user-123"
    at.run()

    assert any("无法加载对话历史" in caption.value for caption in at.caption)
    assert not at.exception
