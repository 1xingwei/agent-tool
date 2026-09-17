"""真实 `interrupt-agent` 图的测试。

`service.py` 中的恢复流程已由
`tests/service/test_service_real_graphs.py` 使用合成的中断图覆盖。它无法覆盖的是
此 agent 自身对 provider 特定结构化输出的依赖：
`determine_birthdate` 调用 `with_structured_output`，而 DeepSeek 会拒绝默认的
（provider 原生 json_schema），报错 "This response_format type is unavailable now"。
因此每个请求都在第一个业务节点失败——此 agent 存在所要演示的中断从未被触达。

所以这里有两层，与 bug 所在位置对应：

* 针对确切契约的离线防护（`method="json_mode"` + 提及
  json 的提示词），这是 CI 运行的部分；
* 一个实时的两轮中断/恢复测试，因为真实故障是 provider
  协议拒绝，任何 mock 都无法复现。
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

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
