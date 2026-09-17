import pytest

from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.tools import web_search


def test_supervisor_agent_compiles():
    assert hasattr(langgraph_supervisor_agent, "ainvoke")


@pytest.mark.network
@pytest.mark.asyncio
async def test_web_search_returns_real_results():
    # web_search formats each hit as "n. <title>\n<href>\n<body>" (agents/tools.py).
    # Asserting "snippet" was always wrong -- DuckDuckGo's body key is `body` --
    # so the suite's only network test could never pass.
    result = await web_search.ainvoke({"query": "langgraph agent framework", "max_results": 3})
    assert result.startswith("1. "), result[:300]
    assert "http" in result
