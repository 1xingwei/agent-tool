import logging
from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from agents.code.tools import file_search, git_diff, git_log, read_file
from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from core import get_model, settings
from core.distill import distill_history

logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    safety: SafeguardOutput
    remaining_steps: RemainingSteps
    # 从长期记忆中召回的历史审查结论（已格式化为可直接注入提示词的文本）
    recalled_reviews: str
    # 会话蒸馏摘要（见 core/distill.py）。存在 state 里的独立字段而
    # 不是塞进 messages，是为了让 /history 仍能返回全量原文。
    distilled_summary: str


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
    def build_messages(state: AgentState) -> list:
        system = instructions
        recalled = state.get("recalled_reviews") or ""
        if recalled:
            system += (
                "\n\n以下是**以往对同一用户的审查结论**（来自长期记忆，按相关度排序）。"
                "若与本次问题相关，请参考它们保持一致，并明确指出哪些是历史结论、"
                "哪些是本次新发现：\n\n" + recalled
            )
        return [SystemMessage(content=system)] + state["messages"]

    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(build_messages, name="StateModifier")
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


def _recalled(value: str) -> AgentState:
    """构造 `recall_reviews` 的返回值。

    `messages` 是 `MessagesState` 的必填键，即便本节点不改动消息也必须显式带上
    （与 `safeguard_input` / `block_unsafe_content` 的既有写法一致），
    否则类型检查会报 `Missing required key 'messages' for TypedDict`。
    """
    return {"recalled_reviews": value, "messages": []}


async def recall_reviews(
    state: AgentState, config: RunnableConfig, store: BaseStore | None
) -> AgentState:
    """在回答前从长期记忆中召回同一用户的历史审查结论。

    这是记忆层的**读路径**。此前 `remember_review` 只写不读，导致 store 变成
    「只进不出的仓库」—— 结论写进去了但从未被使用。

    检索用当前用户问题作为 query，依赖 store 的语义检索（`index` 配置）。
    若 store 未开启语义检索，`asearch` 仍会返回结果但 `score is None`
    且是主键序 —— 那种情况下召回结果不可信，因此**按 score 是否有效来判定**，
    而不是「结果是否为空」。
    """
    if store is None:
        return _recalled("")

    # 取最后一条人类消息作为检索 query
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return _recalled("")

    query = str(human_messages[-1].content)
    if not query.strip():
        return _recalled("")

    user_id = config["configurable"].get("user_id", "anonymous")
    namespace = ("code-reviewer", user_id)

    try:
        items = await store.asearch(namespace, query=query, limit=3)
    except Exception as e:
        # 记忆读失败不应影响本次审查
        logger.warning("Failed to recall past reviews: %s: %s", type(e).__name__, e)
        return _recalled("")

    # `score is None` 表示 store 未开启语义检索，此时排序是主键序、不具参考性。
    # 直接跳过，避免把不相关的内容当成「相关历史」注入提示词。
    scored = [it for it in items if getattr(it, "score", None) is not None]
    if not scored:
        return _recalled("")

    formatted = "\n\n".join(
        f"- ({it.key}, 相似度 {it.score:.3f}) {it.value.get('conclusion', '')}".strip()
        for it in scored
    )
    return _recalled(formatted)


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
agent.add_node("recall_reviews", recall_reviews)
agent.add_node("remember_review", remember_review)
# 蒸馏是同步节点（`distill_history` 内部走同步 `model.invoke`），
# 在进入模型节点前折叠旧消息。关闭时它只做一次阈值判断并返回空 part。
agent.add_node("distill_history", distill_history)
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
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "recall_reviews"}
)

# 阻止不安全内容后始终 END
agent.add_edge("block_unsafe_content", END)

# 召回历史结论后才进入模型节点，使召回内容能进入提示词
agent.add_edge("recall_reviews", "distill_history")

# 蒸馏之后才进模型：这样 model 看到的是一份已经折叠过旧消息的 messages
agent.add_edge("distill_history", "model")

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
