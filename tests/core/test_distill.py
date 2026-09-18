"""会话蒸馏（P0-7）的回归用例。

分两层：

- **纯函数层**（`plan_distillation` / `apply_distillation` / `build_summary_prompt`）
  完全离线，用假模型驱动，断言的是判定与改写的不变量。
- **协议层**（`distill_history` 真实调模型、`_distill_input` 真实改写）
  必须真跑，因为被守护的缺陷都住在「消息 id 对不对得上」「模型回的
  content 是不是 list」这类只有真调用才暴露的地方。

分层依据见 docs/10 的测试分层法则：bug 住在静态结构里用 mock，
住在 API 协议里必须真实调用。
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import StateSnapshot

from core import settings
from core.distill import (
    apply_distillation,
    build_summary_prompt,
    distill_history,
    plan_distillation,
)


def _history(turns: int, *, with_ids: bool = True) -> list:
    """造 `turns` 轮对话，每轮「人类发言 + 助手回复」。"""
    messages: list = []
    for i in range(turns):
        human = HumanMessage(content=f"问题 {i}")
        reply = AIMessage(content=f"回答 {i}")
        if with_ids:
            human.id = f"h{i}"
            reply.id = f"a{i}"
        messages.extend([human, reply])
    return messages


class _RecordingModel:
    """记下收到的 prompt 并回一个固定摘要。"""

    def __init__(self, reply: str = "这是一段摘要。") -> None:
        self.reply = reply
        self.calls: list[Any] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.reply)


@pytest.fixture
def _distill_on(monkeypatch):
    """打开蒸馏并收紧阈值，让用例不必造几十条消息。"""
    monkeypatch.setattr(settings, "DISTILL_ENABLED", True)
    monkeypatch.setattr(settings, "DISTILL_TRIGGER_MESSAGES", 10)
    monkeypatch.setattr(settings, "DISTILL_MIN_NEW_MESSAGES", 4)
    monkeypatch.setattr(settings, "DISTILL_KEEP_MESSAGES", 4)
    monkeypatch.setattr(settings, "DISTILL_KEEP_TURNS", 2)
    monkeypatch.setattr(settings, "DISTILL_SNIPPET_CHARS", 40)


# ---------------------------------------------------------------------------
# 判定：纯函数
# ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    """默认不蒸馏——这是「零外部依赖、零额外花费」的默认值。"""
    monkeypatch.setattr(settings, "DISTILL_ENABLED", False)
    decision = plan_distillation(_history(30))
    assert decision.should_distill is False
    assert decision.reason == "disabled"


def test_below_threshold_is_not_distilled(_distill_on):
    decision = plan_distillation(_history(3))  # 6 条 < 10
    assert decision.should_distill is False
    assert "below_threshold" in decision.reason


def test_above_threshold_is_distilled(_distill_on):
    decision = plan_distillation(_history(10))  # 20 条
    assert decision.should_distill is True
    assert decision.to_summarize


def test_messages_without_id_are_still_summarized(_distill_on):
    """消息有没有 id 不影响蒸馏。

    曾经这里要求「必须有 id」——那是为 `RemoveMessage` 服务的。现在写入侧
    一条消息都不删，这个门槛就变成了纯粹的功能缺陷：图节点自己产生的消息
    通常没有 id，于是摘要永远不触发（探针实测摘要长度 0，一次都没生效）。
    """
    decision = plan_distillation(_history(10, with_ids=False))
    assert decision.should_distill is True
    assert decision.to_summarize


def test_keep_turns_window_is_respected(_distill_on):
    """最近 DISTILL_KEEP_TURNS 轮必须落在摘要范围之外。"""
    messages = _history(10)
    decision = plan_distillation(messages)
    # 保留 2 轮 + KEEP_MESSAGES=4 的保守修正 → 切点取更靠后者
    kept_ids = {m.id for m in messages if m.id not in {s.id for s in decision.to_summarize}}
    assert "h9" in kept_ids and "a9" in kept_ids
    assert "h8" in kept_ids and "a8" in kept_ids


def test_min_new_messages_prevents_repeated_summarization(_distill_on):
    """超阈值但新增不足时不重摘——摘要调用是有成本的。"""
    messages = _history(10)
    decision = plan_distillation(messages)
    # 拿第一次的摘要范围回填，模拟下一轮只剩极少新增
    already = {m.id for m in decision.to_summarize}
    fresh = [m for m in messages if m.id not in already]
    assert plan_distillation(fresh, "已有摘要").should_distill is False


# ---------------------------------------------------------------------------
# 改写：纯函数
# ---------------------------------------------------------------------------


def test_apply_without_summary_is_identity(_distill_on):
    messages = _history(20)
    assert apply_distillation(messages, "") == messages


def test_apply_zero_keep_is_identity(_distill_on):
    messages = _history(20)
    assert apply_distillation(messages, "摘要", keep_messages=0) == messages


def test_apply_below_keep_is_identity(_distill_on):
    messages = _history(2)  # 4 条
    assert apply_distillation(messages, "摘要", keep_messages=4) == messages


def test_apply_prefixes_summary_and_keeps_tail(_distill_on):
    messages = _history(10)
    view = apply_distillation(messages, "这是摘要", keep_messages=4)
    assert len(view) == 5
    assert isinstance(view[0], HumanMessage)
    assert "这是摘要" in view[0].content
    # 尾部原文按原顺序保留
    assert [m.id for m in view[1:]] == [m.id for m in messages[-4:]]


def test_apply_never_starts_with_orphan_tool_message(_distill_on):
    """切点必须落回人类发言上。

    若视图以 ToolMessage 开头，部分 provider 会直接 400——孤立的工具结果
    没有对应的工具调用请求。这条用例造出「切点刚好落在工具消息上」的形状。
    """
    messages: list = []
    for i in range(6):
        human = HumanMessage(content=f"问题 {i}")
        human.id = f"h{i}"
        call = AIMessage(content="", tool_calls=[{"id": f"t{i}", "name": "x", "args": {}}])
        call.id = f"a{i}"
        result = ToolMessage(content="结果", tool_call_id=f"t{i}")
        result.id = f"t{i}"
        messages.extend([human, call, result])

    view = apply_distillation(messages, "摘要", keep_messages=2)
    assert isinstance(view[1], HumanMessage), "视图第二条必须是人类发言，不能是孤立工具结果"


def test_apply_does_not_mutate_input(_distill_on):
    messages = _history(10)
    before = list(messages)
    apply_distillation(messages, "摘要", keep_messages=4)
    assert messages == before


def test_build_summary_prompt_merges_previous_summary():
    prompt = build_summary_prompt(_history(2), "旧摘要内容")
    assert "旧摘要内容" in prompt
    assert "已有摘要" in prompt
    assert "新增对话" in prompt
    assert "用户: 问题 0" in prompt
    assert "助手: 回答 0" in prompt


def test_build_summary_prompt_truncates_long_tool_output(_distill_on, monkeypatch):
    monkeypatch.setattr(settings, "DISTILL_SNIPPET_CHARS", 10)
    tool = ToolMessage(content="x" * 500, tool_call_id="t1")
    tool.name = "read_file"
    prompt = build_summary_prompt([tool])
    assert "xxxxxxx…" in prompt
    assert "x" * 100 not in prompt


# ---------------------------------------------------------------------------
# 写入路径：真实（协议层）
# ---------------------------------------------------------------------------


def test_distill_history_returns_messages_key(_distill_on):
    """`AgentState` 继承 `MessagesState`，缺 `messages` 键会被类型检查拦下。"""
    with patch("core.distill.get_distill_model", return_value=_RecordingModel()):
        out = distill_history({"messages": _history(10), "distilled_summary": ""})
    assert "messages" in out


def test_distill_history_never_deletes_original_messages(_distill_on):
    """写入侧一条消息都不删——这是本方案与「有损改写」的分界线。

    实测过反直觉的点：返回 `RemoveMessage` 时，reducer 会**立刻**把它执行掉，
    写进 checkpoint 的就是裁剪后的短列表（探针里 12 轮对话只剩 6 条），
    `/history` 再也拿不到原文。所以这条用例断言 `messages` 必须是空列表。
    """
    with patch("core.distill.get_distill_model", return_value=_RecordingModel()):
        out = distill_history({"messages": _history(10), "distilled_summary": ""})
    assert out["messages"] == [], "不得返回任何删除指令，原文必须留在 checkpoint"
    assert out["distilled_summary"] == "这是一段摘要。"


def test_distill_history_survives_model_failure(_distill_on):
    """蒸馏失败必须退回「不蒸馏」，绝不让流式请求跟着挂掉。"""

    class _Boom:
        def invoke(self, messages: list) -> AIMessage:
            raise RuntimeError("模型不可用")

    with patch("core.distill.get_distill_model", return_value=_Boom()):
        out = distill_history({"messages": _history(10), "distilled_summary": ""})
    assert out == {"messages": []}


def test_distill_history_discards_empty_summary(_distill_on):
    with patch("core.distill.get_distill_model", return_value=_RecordingModel(reply="   ")):
        out = distill_history({"messages": _history(10), "distilled_summary": ""})
    assert "distilled_summary" not in out
    assert out["messages"] == []


def test_distill_history_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "DISTILL_ENABLED", False)
    model = _RecordingModel()
    with patch("core.distill.get_distill_model", return_value=model):
        out = distill_history({"messages": _history(30), "distilled_summary": ""})
    assert out == {"messages": []}
    assert model.calls == [], "关闭时不应调用模型"


def test_distill_history_merges_previous_summary(_distill_on):
    model = _RecordingModel()
    with patch("core.distill.get_distill_model", return_value=model):
        distill_history({"messages": _history(10), "distilled_summary": "上一轮摘要"})
    sent = model.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert "上一轮摘要" in sent[1].content


def test_get_distill_model_falls_back_to_default(monkeypatch):
    """`DISTILL_MODEL` 留空时用 `DEFAULT_MODEL`，且复用 `@cache` 的实例。"""
    from core.distill import get_distill_model

    monkeypatch.setattr(settings, "DISTILL_MODEL", None)
    sentinel = Mock()
    with patch("core.distill.get_model", return_value=sentinel) as fake:
        assert get_distill_model() is sentinel
    fake.assert_called_once_with(settings.DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# 读取路径：service 层集成
# ---------------------------------------------------------------------------


def _state_with(messages: list, summary: str = "", tasks: tuple = ()) -> StateSnapshot:
    return StateSnapshot(
        values={"messages": messages, "distilled_summary": summary},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=tasks,
        interrupts=(),
    )


def test_distill_input_prepends_summary_view(_distill_on):
    from service.service import _distill_input

    messages = _history(10)
    payload = {"messages": [HumanMessage(content="最新的问题")]}
    out = _distill_input(payload, _state_with(messages, "历史摘要"))

    assert len(out["messages"]) < len(messages) + 1
    assert isinstance(out["messages"][0], HumanMessage)
    assert "历史摘要" in out["messages"][0].content
    # 本轮输入必须仍在最后
    assert out["messages"][-1].content == "最新的问题"


def test_distill_input_is_noop_without_summary(_distill_on):
    """没有摘要时不改写。

    这里**必须**同时把 `apply_distillation` 打桩成「一定会改写」，否则用例是
    假的：真实实现自己也对空摘要短路，即使 service 层的守卫被删掉，
    调用链最终仍然返回原样，用例照样绿。守卫被拿掉的缺陷就从这个缝里漏过去。

    打桩后语义变成：「若 service 层把空摘要当有效摘要传下去，会不会改写？」
    —— 会，所以用例真的在测 service 层自己那道闸。
    """
    from service.service import _distill_input

    payload = {"messages": [HumanMessage(content="hi")]}
    marker = HumanMessage(content="被改写过的历史")
    with patch("service.service.apply_distillation", return_value=[marker]):
        out = _distill_input(payload, _state_with(_history(10), ""))
    assert out["messages"] == payload["messages"], "没有摘要时不应把历史拼进 input"


def test_distill_input_is_noop_when_disabled(monkeypatch):
    from service.service import _distill_input

    monkeypatch.setattr(settings, "DISTILL_ENABLED", False)
    payload = {"messages": [HumanMessage(content="hi")]}
    assert _distill_input(payload, _state_with(_history(10), "摘要")) is payload


def test_distill_input_is_noop_for_empty_thread(_distill_on):
    from service.service import _distill_input

    payload = {"messages": [HumanMessage(content="hi")]}
    out = _distill_input(payload, _state_with([], "摘要"))
    assert out["messages"] == payload["messages"]


@pytest.mark.asyncio
async def test_handle_input_does_not_distill_on_interrupt_resume(_distill_on):
    """恢复 interrupt 时不能改写——`Command` 是控制指令，不是消息。"""
    from langgraph.types import Command, Interrupt

    from service.service import _handle_input

    task = Mock()
    task.interrupts = (Interrupt(value="请确认", id="i1"),)
    state = _state_with(_history(10), "历史摘要", tasks=(task,))

    agent = AsyncMock()
    agent.aget_state = AsyncMock(return_value=state)

    kwargs, _ = await _handle_input(
        Mock(message="确认", thread_id="t1", user_id="u1", model=None, agent_config=None),
        agent,
        "a1",
    )

    assert isinstance(kwargs["input"], Command)
    # 未被换成 dict，说明读取侧没有介入
    assert not isinstance(kwargs["input"], dict)
