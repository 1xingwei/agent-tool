from agents.loop_agent import loop_agent, tools


def test_loop_agent_compiles() -> None:
    nodes = set(loop_agent.get_graph().nodes.keys())
    assert {"model", "tools"} <= nodes


def test_loop_agent_has_react_tools() -> None:
    names = [t.name for t in tools]
    assert "WebSearch" in names
    assert "Calculator" in names