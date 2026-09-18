"""工具类多智能体图的共享装配工厂（docs/16 第 2 项）。

`build_tool_agent_graph` 按「guard_input → model → tools」+ 安检骨架装配图，
`_wrap_model` 是**走工厂的 agent** 的视图构造（含会话蒸馏折叠）唯一实现。

注意「唯一」只对工厂成立：另有 4 处手写图自带视图构造、不经过这里，
因此**不带**蒸馏折叠 —— `loop_agent`（最典型的长对话 ReAct agent）、
`knowledge_base_agent`、`interrupt_agent`、`bg_task_agent`（docs/19 R3）。
不 import `agents.agents`（其 `AgentGraph` 定义在那里，会成环）。
"""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from core import get_model, settings
from core.distill import apply_distillation


class AgentState(MessagesState, total=False):
    """共享状态；`total=False` 对没有相应节点的 agent 无害。"""

    safety: SafeguardOutput
    remaining_steps: RemainingSteps
    recalled_reviews: str
    distilled_summary: str


def _wrap_model(
    model: BaseChatModel,
    *,
    tools: Sequence[BaseTool],
    instructions: str,
    system_suffix: Callable[[AgentState], str] | None = None,
) -> RunnableSerializable[AgentState, AIMessage]:
    """构造「视图 → 绑定工具的模型」这条链。

    视图构造（含蒸馏折叠）在**工厂内**只有这一个实现；走工厂的 agent 想改视图
    只能通过 `system_suffix` 钩子，改不了折叠本身。全仓另有 4 处手写图的视图
    构造不经此处（见模块 docstring），它们没有蒸馏折叠。
    """

    def build_messages(state: AgentState) -> list:
        system = instructions
        if system_suffix:
            suffix = system_suffix(state)
            if suffix:
                system += suffix
        return [SystemMessage(content=system)] + apply_distillation(
            list(state["messages"]), state.get("distilled_summary") or ""
        )

    preprocessor = RunnableLambda(build_messages, name="StateModifier")
    return preprocessor | model.bind_tools(tools)  # type: ignore[return-value]


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


def build_tool_agent_graph(
    *,
    tools: Sequence[BaseTool],
    instructions: str,
    system_suffix: Callable[[AgentState], str] | None = None,
    extra_nodes: Mapping[str, Callable[..., Any]] | None = None,
    extra_edges: Sequence[tuple[str, str]] = (),
    guard_next: str = "model",
    done_node: str = END,
) -> CompiledStateGraph:
    """按骨架装配图：guard_input → (safe: guard_next) → model → tools 循环，
    模型完成后的条件边落到 `done_node`（默认 END；传普通节点时自动补该节点 → END）。
    """
    model_wrapper = partial(
        _wrap_model, tools=tools, instructions=instructions, system_suffix=system_suffix
    )

    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        model_runnable = model_wrapper(m)
        response = await model_runnable.ainvoke(state, config)

        if state["remaining_steps"] < 2 and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # 返回列表，因为它会被追加到现有列表
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("guard_input", safeguard_input)
    builder.add_node("model", acall_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("block_unsafe_content", block_unsafe_content)
    for name, node in (extra_nodes or {}).items():
        builder.add_node(name, node)
    builder.set_entry_point("guard_input")

    builder.add_conditional_edges(
        "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": guard_next}
    )
    builder.add_edge("block_unsafe_content", END)
    for source, target in extra_edges:
        builder.add_edge(source, target)
    builder.add_edge("tools", "model")
    builder.add_conditional_edges(
        "model", pending_tool_calls, {"tools": "tools", "done": done_node}
    )
    if done_node != END:
        builder.add_edge(done_node, END)
    return builder.compile()
