import logging
from datetime import datetime
from functools import partial

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agents.code.tools import file_search, git_diff, git_log, read_file
from agents.graph_factory import AgentState, _wrap_model, build_tool_agent_graph
from core.distill import distill_history

logger = logging.getLogger(__name__)


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


def _recall_suffix(state: AgentState) -> str:
    recalled = state.get("recalled_reviews") or ""
    if not recalled:
        return ""
    return (
        "\n\n以下是**以往对同一用户的审查结论**（来自长期记忆，按相关度排序）。"
        "若与本次问题相关，请参考它们保持一致，并明确指出哪些是历史结论、"
        "哪些是本次新发现：\n\n" + recalled
    )


# 接缝：`agents.code_reviewer.wrap_model` 被 tests/agents/test_memory_read_path.py
# 直接 import 并调用，必须继续从这里导出。
wrap_model = partial(
    _wrap_model, tools=tools, instructions=instructions, system_suffix=_recall_suffix
)


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
    """将最终审查结论持久化到 store，按用户 + 日期为键（无仓库维度，见 docs/09 §9）。"""
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
    try:
        await store.aput(namespace, key, {"conclusion": last_message.content})
    except Exception as e:
        # 记忆写失败不应影响已经生成的审查回答（docs/20 F7）
        logger.warning("Failed to remember review: %s: %s", type(e).__name__, e)
    return {"messages": []}


code_reviewer = build_tool_agent_graph(
    tools=tools,
    instructions=instructions,
    system_suffix=_recall_suffix,
    extra_nodes={
        "recall_reviews": recall_reviews,
        "remember_review": remember_review,
        # 蒸馏是同步节点（`distill_history` 内部走同步 `model.invoke`），
        # 在进入模型节点前折叠旧消息。关闭时它只做一次阈值判断并返回空 part。
        "distill_history": distill_history,
    },
    extra_edges=[
        ("recall_reviews", "distill_history"),
        ("distill_history", "model"),
    ],
    guard_next="recall_reviews",
    done_node="remember_review",
)
code_reviewer.name = "code-reviewer"
