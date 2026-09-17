"""GitHub MCP Agent 的测试。"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.github_mcp_agent.github_mcp_agent import GitHubMCPAgent, prompt
from core.settings import settings


class TestGitHubMCPAgent:
    """测试 GitHub MCP Agent 功能。"""

    def test_initialization(self):
        """测试 agent 是否正确初始化。"""
        agent = GitHubMCPAgent()
        assert not agent._loaded
        assert agent._mcp_tools == []
        assert agent._mcp_client is None

    @pytest.mark.asyncio
    async def test_load_without_github_pat(self):
        """测试未设置 GITHUB_PAT 时的加载。"""
        agent = GitHubMCPAgent()

        with patch.object(settings, "GITHUB_PAT", None):
            await agent.load()

        assert agent._loaded
        assert agent._mcp_tools == []
        assert agent._mcp_client is None
        assert agent._graph is not None

    @pytest.mark.asyncio
    async def test_load_with_github_pat(self):
        """测试已设置并配置 GITHUB_PAT 时的加载。"""
        agent = GitHubMCPAgent()
        mock_client = Mock()

        # 创建正确的工具实例
        from langchain_core.tools import Tool

        mock_tool1 = Tool(name="test_tool_1", description="Test tool 1", func=lambda x: x)
        mock_tool2 = Tool(name="test_tool_2", description="Test tool 2", func=lambda x: x)
        mock_tools = [mock_tool1, mock_tool2]

        with (
            patch.object(
                settings, "GITHUB_PAT", Mock(get_secret_value=Mock(return_value="test_token"))
            ),
            patch.object(settings, "MCP_GITHUB_SERVER_URL", "https://api.githubcopilot.com/mcp/"),
            patch(
                "agents.github_mcp_agent.github_mcp_agent.MultiServerMCPClient"
            ) as mock_client_class,
            patch("agents.github_mcp_agent.github_mcp_agent.StreamableHttpConnection"),
            patch("agents.github_mcp_agent.github_mcp_agent.get_model") as mock_get_model,
        ):
            mock_client_class.return_value = mock_client
            mock_client.get_tools = AsyncMock(return_value=mock_tools)
            mock_get_model.return_value = Mock()

            await agent.load()

        assert agent._loaded
        assert agent._mcp_tools == mock_tools
        assert agent._mcp_client == mock_client
        assert agent._graph is not None

    @pytest.mark.asyncio
    async def test_load_with_mcp_error(self):
        """测试 MCP 客户端创建失败时的加载。"""
        agent = GitHubMCPAgent()

        with (
            patch.object(
                settings, "GITHUB_PAT", Mock(get_secret_value=Mock(return_value="test_token"))
            ),
            patch.object(settings, "MCP_GITHUB_SERVER_URL", "https://api.githubcopilot.com/mcp/"),
            patch(
                "agents.github_mcp_agent.github_mcp_agent.MultiServerMCPClient",
                side_effect=Exception("Connection failed"),
            ),
            patch("agents.github_mcp_agent.github_mcp_agent.get_model") as mock_get_model,
        ):
            mock_get_model.return_value = Mock()

            await agent.load()

        assert agent._loaded
        assert agent._mcp_tools == []
        assert agent._mcp_client is None
        assert agent._graph is not None

    def test_create_graph(self):
        """测试图创建。"""
        agent = GitHubMCPAgent()
        agent._mcp_tools = [Mock(), Mock()]

        with (
            patch("agents.github_mcp_agent.github_mcp_agent.get_model") as mock_get_model,
            patch("agents.github_mcp_agent.github_mcp_agent.create_agent") as mock_create_agent,
        ):
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_graph = Mock()
            mock_create_agent.return_value = mock_graph

            graph = agent._create_graph()

            assert graph == mock_graph
            mock_create_agent.assert_called_once_with(
                model=mock_model,
                tools=agent._mcp_tools,
                name="github-mcp-agent",
                system_prompt=prompt,
            )

    def test_get_graph_not_loaded(self):
        """测试 agent 未加载时的 get_graph。"""
        agent = GitHubMCPAgent()

        with pytest.raises(RuntimeError, match="Agent not loaded. Call load\\(\\) first."):
            agent.get_graph()

    def test_get_graph_loaded(self):
        """测试 agent 已加载时的 get_graph。"""
        agent = GitHubMCPAgent()
        agent._loaded = True
        agent._graph = Mock()

        graph = agent.get_graph()
        assert graph == agent._graph
