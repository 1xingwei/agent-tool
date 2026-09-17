import random
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command


class AgentState(MessagesState, total=False):
    """`total=False` 是 PEP589 规范。

    文档：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# 定义节点


def node_a(state: AgentState) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["a", "b"])
    goto: Literal["node_b", "node_c"]
    # 这是条件边函数的替代方案
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    # 注意 Command 如何让你既能更新图状态又能路由到下一个节点
    return Command(
        # 这是状态更新
        update={"messages": [AIMessage(content=f"Hello {value}")]},
        # 这是边的替代方案
        goto=goto,
    )


def node_b(state: AgentState):
    print("Called B")
    return {"messages": [AIMessage(content="Hello B")]}


def node_c(state: AgentState):
    print("Called C")
    return {"messages": [AIMessage(content="Hello C")]}


builder = StateGraph(AgentState)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# NOTE: 节点 A、B 和 C 之间没有边！

command_agent = builder.compile()
