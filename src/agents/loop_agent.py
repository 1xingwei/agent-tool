from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.instructions import SEARCH_STOP_CONDITION
from agents.tools import calculator, fetch_url, web_search
from core import get_model, settings

SEARCH_BUDGET = 4  # 超过此后摘掉 WebSearch，模型失去空转条件


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps
    search_calls: int


tools = [calculator, web_search, fetch_url]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
You are a ReAct loop agent that iterates between "think", "act using a tool",
and "observe the result" until the user's request is fully answered.
Today's date is {current_date}.

Loop until done:
1. Plan the next step in your thinking.
2. Call a tool if more information is needed (e.g. WebSearch or Calculator).
3. Read the tool result and continue the loop.
4. Stop calling tools and answer directly once you have everything.
{SEARCH_STOP_CONDITION}

Tool order: first use WebSearch to find candidate pages; when a snippet is too
short to contain the answer, open the page body with fetch_url; stop once you have it.
Remember the final answer must be a single, complete response.
"""


def _search_calls_in(messages) -> int:
    """统计历史里已发生的 WebSearch 工具调用次数。"""
    calls = 0
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []) or []:
            if tc.get("name") == "WebSearch" or getattr(tc, "name", None) == "WebSearch":
                calls += 1
    return calls


def wrap_model(
    model: BaseChatModel, search_budget_exhausted: bool = False
) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(
        [t for t in tools if not (search_budget_exhausted and t.name == "WebSearch")]
    )
    budget_msg = SystemMessage(
        content=(
            "You have run out of WebSearch budget. Answer from what you already know; "
            "do not attempt further web searches."
        )
    )

    def build_messages(state: AgentState) -> list:
        system = [SystemMessage(content=instructions)]
        if search_budget_exhausted:
            system.append(budget_msg)
        return system + state["messages"]

    preprocessor = RunnableLambda(build_messages, name="StateModifier")
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    search_calls = _search_calls_in(state["messages"])
    budget_exhausted = search_calls >= SEARCH_BUDGET
    response = await wrap_model(m, budget_exhausted).ainvoke(state, config)

    if state["remaining_steps"] <= 1 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I ran out of steps before finishing. Please ask me a shorter question.",
                )
            ]
        }
    return {"messages": [response], "search_calls": search_calls}


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


# 定义图
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")

loop_agent = agent.compile()
