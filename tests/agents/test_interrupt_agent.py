"""真实 `interrupt-agent` 图的测试。

`service.py` 中的恢复流程已由
`tests/service/test_service_real_graphs.py` 使用合成的中断图覆盖。它无法覆盖的是
此 agent 自身对 provider 特定结构化输出的依赖：
`determine_birthdate` 调用 `with_structured_output`，而 DeepSeek 会拒绝默认的
（provider 原生 json_schema），报错 "This response_format type is unavailable now"。
因此每个请求都在第一个业务节点失败——此 agent 存在所要演示的中断从未被触达。

所以这里有三层，与 bug 所在位置对应：

* 针对确切契约的离线防护（`method="json_mode"` + 提及
  json 的提示词），这是 CI 运行的部分；
* 一个离线驱动真实编译图的「中断→恢复」用例，用来守住
  `determine_birthdate` 的提取循环（恢复之后必须重新提取）；
* 一个实时的两轮中断/恢复测试，因为真实故障是 provider
  协议拒绝，任何 mock 都无法复现。
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agents.interrupt_agent import (
    BirthdateExtraction,
    birthdate_extraction_prompt,
    determine_birthdate,
)
from schema.models import DeepseekModelName

# 实时测试固定此项，使其不依赖于 DEFAULT_MODEL 在测试进程中恰好如何解析
# （见测试内部的注释）。
LIVE_MODEL = DeepseekModelName.DEEPSEEK_V4_FLASH.value


class RecordingStructuredModel:
    """替代聊天模型，并记录结构化输出是如何被请求的。"""

    def __init__(self, payload: BirthdateExtraction) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append({"schema": schema, **kwargs})
        payload = self.payload
        return RunnableLambda(lambda _model_input: payload)


class ScriptedAgentModel(Runnable[Any, Any]):
    """离线替身：既能当普通聊天模型用，又提供结构化输出。

    图里的三个节点都调用 `get_model`，所以同一个对象必须同时满足两种用法：
    `background` / `generate_response` 走 `wrap_model`，要求它是可 `|` 拼接的
    Runnable；`determine_birthdate` 则要求它有 `with_structured_output`。

    结构化结果**由它看到的输入消息推导**，而不是按固定队列回放。原因是
    LangGraph 恢复中断时会**从头重新执行**该节点，所以「提取被调用了几次」
    属于实现细节。看输入能让断言绑定行为，而不是绑定调用次数。
    """

    def __init__(self) -> None:
        self.chat_inputs: list[Any] = []
        self.structured_calls: list[dict[str, Any]] = []
        self.structured_inputs: list[Any] = []
        self._chat_replies = 0

    # --- 聊天路径 -------------------------------------------------------
    # 参数名必须是 `input`：它覆盖 `Runnable.invoke` / `Runnable.ainvoke`，
    # pyrefly 会按 `bad-override-param-name` 检查名字是否与父类一致。
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # noqa: A002
        return self._reply(input)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # noqa: A002
        return self._reply(input)

    def _reply(self, model_input: Any) -> AIMessage:
        self.chat_inputs.append(model_input)
        self._chat_replies += 1
        return AIMessage(content=f"offline reply #{self._chat_replies}")

    # --- 结构化路径 -----------------------------------------------------
    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        self.structured_calls.append({"schema": schema, **kwargs})
        return RunnableLambda(self._answer_structured)

    def _answer_structured(self, model_input: Any) -> BirthdateExtraction:
        self.structured_inputs.append(model_input)
        text = "\n".join(
            str(getattr(message, "content", ""))
            for message in model_input
            if getattr(message, "content", "")
        )
        if "1990" in text:
            return BirthdateExtraction(birthdate="1990-03-03", reasoning="user gave a date")
        return BirthdateExtraction(birthdate=None, reasoning="no birthdate found")


def test_birthdate_prompt_mentions_json() -> None:
    """`json_mode` 实现为 response_format={"type": "json_object"}。

    OpenAI 兼容的 provider 会拒绝该格式，除非提示词本身提及 json，因此
    从提示词中去掉这个词会让 agent 以 400 失败，即使
    Python 侧看起来仍然正确。
    """
    text = birthdate_extraction_prompt.format().content

    assert isinstance(text, str)
    assert "json" in text.lower()


@pytest.mark.asyncio
async def test_determine_birthdate_requests_json_mode() -> None:
    """结构化输出调用必须请求 json_mode，而非 provider 默认值。"""
    model = RecordingStructuredModel(
        BirthdateExtraction(birthdate="1990-03-03", reasoning="user said so")
    )

    with patch("agents.interrupt_agent.get_model", return_value=model):
        # config 中没有 user_id，因此完全跳过 store 路径。
        result = await determine_birthdate(
            {"messages": [HumanMessage(content="I was born on March 3, 1990.")]},
            {"configurable": {}},
            AsyncMock(),
        )

    assert result["birthdate"] == datetime(1990, 3, 3)
    assert len(model.calls) == 1
    assert model.calls[0]["schema"] is BirthdateExtraction
    assert model.calls[0]["method"] == "json_mode"


@pytest.mark.asyncio
async def test_extraction_loop_reasks_then_succeeds_on_resume() -> None:
    """离线覆盖 `determine_birthdate` 的提取循环：第 1 轮中断 → 第 2 轮恢复。

    `..._with_live_model` 是唯一走「中断→恢复」的用例，但它带 `network` 标记，
    默认 CI 不跑。把递归改成 `while True` 恰恰只在恢复那一轮生效，所以这条
    路径必须有一个离线守护，否则改动等于没被测到。

    这里用脚本化假模型驱动**真实的编译图**（`MemorySaver` + `InMemoryStore`），
    断言的重点是循环语义：恢复后必须把中断返回的回答追加进消息、**再跑一次提取**。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command

    from agents.interrupt_agent import agent as interrupt_graph

    store = InMemoryStore()
    compiled = interrupt_graph.compile(checkpointer=MemorySaver(), store=store)
    config = {
        "configurable": {
            "thread_id": "offline-interrupt-thread",
            "user_id": "offline-user",
        }
    }
    fake = ScriptedAgentModel()

    with patch("agents.interrupt_agent.get_model", return_value=fake):
        first = await compiled.ainvoke({"messages": [HumanMessage(content="Hello there")]}, config)
        assert "__interrupt__" in first, f"expected an interrupt, got {list(first)}"

        resumed = await compiled.ainvoke(Command(resume="March 3, 1990"), config)

    assert not resumed.get("__interrupt__")
    assert resumed["birthdate"] == datetime(1990, 3, 3)

    # 循环确实在恢复之后又跑了一次提取：最后一次提取的输入里带着中断返回的回答。
    # 若循环被写成「恢复后不再重新提取」，这条断言立刻变红。
    assert fake.structured_inputs, "the extraction model was never invoked"
    last_texts = [str(getattr(message, "content", "")) for message in fake.structured_inputs[-1]]
    assert any("1990" in text for text in last_texts), last_texts

    # 提取成功后应写回 store，后续轮次可直接命中而无需再询问。
    # 写入的是 `datetime.isoformat()`，即带时分秒的 `1990-03-03T00:00:00`。
    saved = await store.aget(("offline-user",), "birthdate")
    assert saved is not None
    assert saved.value["birthdate"].startswith("1990-03-03"), saved.value

    # 最终回复由 `generate_response` 产生，非空。
    reply = resumed["messages"][-1].content
    assert isinstance(reply, str) and reply.strip()


@pytest.mark.network
@pytest.mark.asyncio
async def test_interrupt_agent_two_turn_resume_with_live_model() -> None:
    """驱动真实图在实时模型上完成一次完整的 interrupt/resume 循环。

    第 1 轮不提供出生日期，因此提取必须返回 None，节点必须
    中断。第 2 轮用真实日期恢复该中断。这正是该
    agent 旨在演示的路径，在 json_mode 修复之前它一直无法到达。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command

    from agents.interrupt_agent import agent as interrupt_graph
    from core import settings

    if not settings.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not configured; the live path needs a real model")

    compiled = interrupt_graph.compile(checkpointer=MemorySaver(), store=InMemoryStore())
    config = {
        "configurable": {
            "thread_id": "live-interrupt-thread",
            "user_id": "live-user",
            # 显式固定模型。pytest 通过 pyproject 的 [tool.pytest_env] 注入占位符
            # OPENAI_API_KEY，这会使 DEFAULT_MODEL 解析为
            # gpt-5-nano，并会带着假 key 把该请求发往 OpenAI（401）。
            "model": LIVE_MODEL,
        }
    }

    first = await compiled.ainvoke({"messages": [HumanMessage(content="Hello there")]}, config)
    assert "__interrupt__" in first, f"expected an interrupt, got {list(first)}"

    second = await compiled.ainvoke(Command(resume="March 3, 1990"), config)
    assert not second.get("__interrupt__")
    assert second["birthdate"] == datetime(1990, 3, 3)

    reply = second["messages"][-1].content
    assert isinstance(reply, str) and reply.strip()
