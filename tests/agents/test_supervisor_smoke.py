import pytest

from agents.langgraph_supervisor_agent import langgraph_supervisor_agent, web_search


def test_supervisor_agent_compiles():
    assert hasattr(langgraph_supervisor_agent, "ainvoke")


@pytest.mark.network
@pytest.mark.asyncio
async def test_web_search_returns_real_results():
    result = await web_search.ainvoke("langgraph agent framework")
    assert "snippet" in result
