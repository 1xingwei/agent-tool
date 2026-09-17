from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.tools import calculator, web_search
from core import get_model, settings


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps


tools = [calculator, web_search]

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

Remember the final answer must be a single, complete response.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await wrap_model(m).ainvoke(state, config)

    if state["remaining_steps"] <= 1 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I ran out of steps before finishing. Please ask me a shorter question.",
                )
            ]
        }
    return {"messages": [response]}


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")

loop_agent = agent.compile()
