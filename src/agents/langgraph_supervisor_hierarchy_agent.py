from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from agents.langgraph_supervisor_agent import add, multiply
from agents.tools import web_search
from core import get_supervisor_model, settings

# Handoffs 会将子 agent 的回答拼接进此图的历史记录，而 DeepSeek 的
# thinking 模式会拒绝这种内容——get_supervisor_model 的文档说明了原因。
model = get_supervisor_model(settings.DEFAULT_MODEL)


def workflow(chosen_model):
    math_agent = create_agent(
        model=chosen_model,
        tools=[add, multiply],
        name="sub-agent-math_expert",  # 将该图节点标识为子 agent
        system_prompt="You are a math expert. Always use one tool at a time.",
    ).with_config(tags=["skip_stream"])

    research_agent = (
        create_supervisor(
            [math_agent],
            model=chosen_model,
            tools=[web_search],
            prompt="You are a world class researcher with access to web search. Do not do any math, you have a math expert for that. ",
            supervisor_name="supervisor-research_expert",  # 将该图节点标识为 math agent 的 supervisor
        )
        .compile(name="sub-agent-research_expert")  # 将该图节点标识为主 supervisor 的子 agent
        .with_config(tags=["skip_stream"])
    )  # 在 UI 中，子 agent 的流式 token 会被忽略

    # 创建 supervisor 工作流
    return create_supervisor(
        [research_agent],
        model=chosen_model,
        prompt=(
            "You are a team supervisor managing a research expert with math capabilities. "
            "For current events, use research_agent. "
        ),
        add_handoff_back_messages=True,
        # UI 现在期望此值为 True，这样我们无需猜测何时发生 handoff 返回
        output_mode="full_history",  # 否则在重新加载对话时，子 agent 的消息不会被包含
    )  # supervisor 的默认名称为 "supervisor"。


langgraph_supervisor_hierarchy_agent = workflow(model).compile()
