import logging
from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from agents.code.tools import file_search, git_diff, git_log, read_file
from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from core import get_model, settings

logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    safety: SafeguardOutput
    remaining_steps: RemainingSteps


tools = [git_log, git_diff, file_search, read_file]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
You are a code repository reviewer. You analyze a git repository with read-only tools:
- git_log and git_diff read commit history and changes.
- file_search and read_file locate and read source files.

Rules:
- Only use information returned by the tools. Never invent commit hashes, file contents, or diffs.
- For change review: list what changed, note risks, and give concrete improvement suggestions.
- For code questions: trace through real code, cite file paths.
- After a review, summarize the conclusions so they can be remembered.
Chat content in the user's language.
Today's date is {current_date}.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
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
    # 返回列表，因为这会追加到现有列表
    return {"messages": [response]}


async def remember_review(
    state: AgentState, config: RunnableConfig, store: BaseStore | None
) -> AgentState:
    """将最终审查结论持久化到 store，按仓库为键。"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {"messages": []}
    if last_message.tool_calls:
        return {"messages": []}
    if store is None:
        # 仅当图在服务的 lifespan 下运行时才会注入 store。
        # 独立调用它（`langgraph dev`、run_agent.py、单元测试）会传入 None，
        # 而裸调用 `store.aput` 会引发 AttributeError。
        logger.warning("No store injected; the review conclusion was not persisted.")
        return {"messages": []}
    user_id = config["configurable"].get("user_id", "anonymous")
    namespace = ("code-reviewer", user_id)
    key = f"review-{datetime.now().strftime('%Y%m%d')}"
    await store.aput(namespace, key, {"conclusion": last_message.content})
    return {"messages": []}


async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# 定义图
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", safeguard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("remember_review", remember_review)
agent.set_entry_point("guard_input")


# 检查不安全输入，若发现则阻止后续处理
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# 阻止不安全内容后始终 END
agent.add_edge("block_unsafe_content", END)

# 始终在 "tools" 之后运行 "model"
agent.add_edge("tools", "model")


# 在 "model" 之后，若有工具调用则运行 "tools"。否则 remember + END。
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges(
    "model", pending_tool_calls, {"tools": "tools", "done": "remember_review"}
)

agent.add_edge("remember_review", END)

code_reviewer = agent.compile()
code_reviewer.name = "code-reviewer"
