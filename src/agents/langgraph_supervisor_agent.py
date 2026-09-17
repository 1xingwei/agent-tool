from typing import Any

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from agents.tools import web_search
from core import get_supervisor_model, settings

# Handoffs 会将子 agent 的回答拼接进此图的历史记录，而 DeepSeek 的
# thinking 模式会拒绝这种内容——get_supervisor_model 的文档说明了原因。
model = get_supervisor_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    """将两个数相加。"""
    return a + b


def multiply(a: float, b: float) -> float:
    """将两个数相乘。"""
    return a * b


math_agent: Any = create_agent(
    model=model,
    tools=[add, multiply],
    name="sub-agent-math_expert",
    system_prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

research_agent: Any = create_agent(
    model=model,
    tools=[web_search],
    name="sub-agent-research_expert",
    system_prompt="You are a world class researcher with access to web search. Do not do any math.",
).with_config(tags=["skip_stream"])


# 创建 supervisor 工作流
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
    add_handoff_back_messages=True,
    # UI 现在期望此值为 True，这样我们无需猜测何时发生 handoff 返回
    output_mode="full_history",  # 否则在重新加载对话时，子 agent 的消息不会被包含
)

langgraph_supervisor_agent = workflow.compile()
