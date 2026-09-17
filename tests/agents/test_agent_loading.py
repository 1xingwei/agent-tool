"""agent 加载功能的测试。"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.agents import agents, get_agent, load_agent
from agents.lazy_agent import LazyLoadingAgent


class TestAgentLoading:
    """测试 agent 加载功能。"""

    @pytest.mark.asyncio
    async def test_load_agent_static_agent(self):
        """测试加载静态 agent（空操作）。"""
        # 静态 agent 无需加载
        await load_agent("chatbot")
        # 不应抛出任何异常

    @pytest.mark.asyncio
    async def test_load_agent_lazy_agent(self):
        """测试加载惰性 agent。"""
        # 模拟 GitHub MCP agent
        mock_agent = Mock(spec=LazyLoadingAgent)
        mock_agent.load = AsyncMock()

        with patch.dict(agents, {"test-lazy-agent": Mock(graph_like=mock_agent)}):
            await load_agent("test-lazy-agent")

        mock_agent.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_agent_nonexistent(self):
        """测试加载不存在的 agent。"""
        with pytest.raises(KeyError):
            await load_agent("nonexistent-agent")

    def test_get_agent_static_agent(self):
        """测试获取静态 agent。"""
        agent = get_agent("chatbot")
        assert agent is not None

    def test_get_agent_lazy_agent_not_loaded(self):
        """测试获取尚未加载的惰性 agent。"""
        mock_agent = Mock(spec=LazyLoadingAgent)
        mock_agent._loaded = False

        with patch.dict(agents, {"test-lazy-agent": Mock(graph_like=mock_agent)}):
            with pytest.raises(
                RuntimeError, match="Agent test-lazy-agent not loaded. Call load\\(\\) first."
            ):
                get_agent("test-lazy-agent")

    def test_get_agent_lazy_agent_loaded(self):
        """测试获取已加载的惰性 agent。"""
        mock_agent = Mock(spec=LazyLoadingAgent)
        mock_agent._loaded = True
        mock_graph = Mock()
        mock_agent.get_graph.return_value = mock_graph

        with patch.dict(agents, {"test-lazy-agent": Mock(graph_like=mock_agent)}):
            result = get_agent("test-lazy-agent")

        assert result == mock_graph
        mock_agent.get_graph.assert_called_once()

    def test_get_agent_nonexistent(self):
        """测试获取不存在的 agent。"""
        with pytest.raises(KeyError):
            get_agent("nonexistent-agent")
