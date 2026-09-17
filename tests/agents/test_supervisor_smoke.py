"""supervisor agent 及其规避的 DeepSeek 思考模式约束的测试。

`test_supervisor_agent_compiles` 只能证明对象存在，交接
bug 正是这样存活的：图编译完美，却在第一条真实消息上就崩溃。下面的测试
改为覆盖该机制。

分两层，因为该 bug 存在于 API 协议而非静态结构中：

* 对 `get_supervisor_model` 的离线防护，这是修复交接的杠杆
  （以及它不得干扰的缓存模型），适合 CI；
* 一次实时交接，因为「DeepSeek 拒绝没有
  reasoning_content 的纯内容 assistant 消息」无法用任何 mock 复现。
"""

import pytest
from langchain_openai import ChatOpenAI

from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.tools import web_search
from core import get_model, get_supervisor_model
from schema.models import DeepseekModelName, OpenAIModelName


def test_supervisor_agent_compiles():
    assert hasattr(langgraph_supervisor_agent, "ainvoke")


def test_supervisor_model_disables_deepseek_thinking() -> None:
    """交接会把子 agent 的答案作为纯内容 assistant 消息拼接进去。

    DeepSeek 的思考模式会拒绝这种做法，除非回传 `reasoning_content`，而
    `ChatOpenAI` 从不捕获它，因此 supervisor 模型必须在关闭思考的情况下运行。
    """
    model = get_supervisor_model(DeepseekModelName.DEEPSEEK_V4_FLASH)

    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_supervisor_model_leaves_other_providers_alone() -> None:
    """只有 DeepSeek 理解该字段；发送到其他地方会导致请求错误。"""
    model = get_supervisor_model(OpenAIModelName.GPT_5_NANO)

    assert "thinking" not in (model.extra_body or {})


def test_supervisor_model_does_not_pollute_the_shared_cached_model() -> None:
    """`get_model` 被缓存并与其他 agent 共享，因此必须保持不被改动。

    如果该适配器就地修改了缓存实例，其他每个 agent 都会
    静默失去思考模式。
    """
    plain = get_model(DeepseekModelName.DEEPSEEK_V4_FLASH)
    supervisor = get_supervisor_model(DeepseekModelName.DEEPSEEK_V4_FLASH)

    assert supervisor is not plain
    assert "thinking" not in (plain.extra_body or {})


@pytest.mark.network
@pytest.mark.asyncio
async def test_web_search_returns_real_results():
    # web_search 将每个命中格式化为 "n. <title>\n<href>\n<body>"（agents/tools.py）。
    # 断言 "snippet" 一直是错的——DuckDuckGo 的 body 键是 `body`——
    # 因此该套件唯一的网络测试永远无法通过。
    result = await web_search.ainvoke({"query": "langgraph agent framework", "max_results": 3})
    assert result.startswith("1. "), result[:300]
    assert "http" in result


@pytest.mark.network
@pytest.mark.asyncio
async def test_supervisor_handoff_reaches_a_sub_agent() -> None:
    """强制一次真实交接并断言子 agent 的答案返回。

    子 agent 的工具返回一个 supervisor 不可能知道的代码，因此
    正确答案*必须*依赖成功的委派并返回；猜测不是
    选项。这很重要，因为静默从未发生的交接否则
    会让该测试因错误的原因通过。

    这里重新构建图，而不是导入模块中已编译的 agent，后者
    在导入时绑定 `settings.DEFAULT_MODEL`——在 pytest 下它会解析为
    位于 pyproject `[tool.pytest_env]` 占位符 key 之后的 OpenAI 模型。
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langgraph_supervisor import create_supervisor

    from core import settings

    if not settings.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not configured; the handoff path needs a real model")

    def lookup_team_code() -> str:
        """查询该团队的私有代码。"""
        return "ZQ-4417-KX"

    def build():
        model = get_supervisor_model(DeepseekModelName.DEEPSEEK_V4_FLASH)
        expert = create_agent(
            model=model,
            tools=[lookup_team_code],
            name="sub-agent-code_keeper",
            system_prompt="You hold the team code. Always use your tool to look it up.",
        )
        workflow = create_supervisor(
            [expert],
            model=model,
            prompt="You manage a code keeper. Delegate any question about the team code to it.",
            add_handoff_back_messages=True,
            output_mode="full_history",
        )
        return workflow.compile()

    result = await build().ainvoke(
        {"messages": [HumanMessage(content="What is the team's private code?")]}
    )

    transcript = "\n".join(str(message.content) for message in result["messages"])
    assert "ZQ-4417-KX" in transcript, transcript[:600]
