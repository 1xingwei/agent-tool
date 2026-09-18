"""ReAct 循环 agent 的测试。

`test_loop_agent_compiles` 断言图具有正确的节点，而这正是
supervisor 交接 bug 逃过的那类断言：图编译正常，
只有在真正运行一条消息时才会失败。下面的脚本化模型测试
驱动真实图——真实条件边、真实 ToolNode、真实 Calculator——因此当
model -> tools -> model 循环停止工作时它会失败。
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

from agents.loop_agent import loop_agent, tools


def test_loop_agent_compiles() -> None:
    nodes = set(loop_agent.get_graph().nodes.keys())
    assert {"model", "tools"} <= nodes


def test_loop_agent_has_react_tools() -> None:
    names = [t.name for t in tools]
    assert "WebSearch" in names
    assert "Calculator" in names
    assert "fetch_url" in names


def test_loop_agent_instructions_include_stop_condition() -> None:
    """P1 守卫：止损条款必须已进 instructions（防止再次漏改）。"""
    from agents.loop_agent import instructions

    assert "stop early" in instructions


def test_search_budget_removes_websearch_when_exhausted() -> None:
    """P3 守卫：WebSearch 次数超过阈值后，bind_tools 里不再有 WebSearch。"""
    from agents.loop_agent import SEARCH_BUDGET, _search_calls_in, wrap_model
    from core.llm import FakeToolModel

    history = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="WebSearch", args={"query": "x"}, id=f"w{i}")
                for i in range(SEARCH_BUDGET)
            ],
        )
    ]
    assert _search_calls_in(history) >= SEARCH_BUDGET, "前置失败：计数应达到预算上限"

    captured: dict = {}

    class SpyModel(FakeToolModel):
        def bind_tools(self, tools, **kwargs):
            captured["names"] = [t.name for t in tools]
            return self

    wrap_model(SpyModel(responses=["ok"]), search_budget_exhausted=True)  # type: ignore[arg-type]
    assert "WebSearch" not in captured["names"], "P3 回归：预算耗尽后 WebSearch 仍在绑定里"
    assert "Calculator" in captured["names"]
    assert "fetch_url" in captured["names"]


@pytest.mark.asyncio
async def test_loop_agent_calls_a_tool_then_answers(fake_tool_model) -> None:
    """驱动一次完整的 model -> tools -> model 循环并断言最终答案。"""
    responses = [
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="Calculator", args={"expression": "2+3"}, id="call-1")],
        ),
        AIMessage(content="The answer is 5."),
    ]

    with patch("agents.loop_agent.get_model", return_value=fake_tool_model(responses)):
        result = await loop_agent.ainvoke(
            {"messages": [HumanMessage(content="What is 2+3?")]},
            {"configurable": {}},
        )

    messages = result["messages"]

    # tools 边被走到，真实 Calculator 确实运行了。
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "5"

    # 循环在模型的第二次响应处结束，而不是在工具之后停止。
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "The answer is 5."
    assert not messages[-1].tool_calls
