import pytest

from client import AgentClient


@pytest.fixture
def agent_client(mock_env):
    """用于创建具有干净环境的测试客户端的 fixture。"""
    ac = AgentClient(base_url="http://test", get_info=False)
    ac.update_agent("test-agent", verify=False)
    return ac
