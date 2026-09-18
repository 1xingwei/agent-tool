import os
from unittest.mock import patch

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage

from core.settings import settings


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )
    parser.addoption(
        "--run-network", action="store_true", default=False, help="run network-dependent tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")
    config.addinivalue_line("markers", "network: mark test as requiring live external network")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="need --run-network option to run")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


@pytest.fixture(autouse=True)
def _isolate_fts_sidecar(tmp_path, monkeypatch):
    """全局兜底：默认把 FTS 侧车库指到临时目录，任何测试都别碰生产索引。

    `settings.CHROMA_FTS_DB` 的默认值是**相对 CWD 的 `./var/chroma_fts.sqlite`**，
    也就是生产 RAG 索引；而 `create_chroma_db` 建库时会先 `DROP TABLE` 再重建。
    因此只要有一个测试忘了显式隔离，跑一次测试就会把生产索引覆盖成测试语料 ——
    这在 `tests/test_ingest.py` 与 `tests/agents/test_hybrid_retrieval.py` 上
    各实测发生过一次（docs/19 R1）。

    这里做系统性兜底而不是逐点打补丁：需要断言「默认取 settings」这条分支的用例
    （如 `test_ingest_builds_fts_sidecar`）可以在测试体内再 monkeypatch 覆盖回来。
    """
    monkeypatch.setattr(settings, "CHROMA_FTS_DB", str(tmp_path / "chroma_fts.sqlite"))


@pytest.fixture
def mock_env():
    """fixture，确保每个测试的环境干净。

    home 目录相关变量会保留，以便 streamlit AppTest 的 expanduser() 仍能工作；
    清空 USERPROFILE/HOME 会导致首次脚本运行崩溃，报错
    「Could not determine home directory.」
    """
    kept = {k: v for k, v in os.environ.items() if k in {"HOME", "USERPROFILE", "TMP", "TEMP"}}
    with patch.dict(os.environ, kept, clear=True):
        yield


class FakeToolModel(FakeMessagesListChatModel):
    """回放脚本化的消息，包括携带工具调用的 `AIMessage`。

    `bind_tools` 是空操作，因此图可以绑定其真实工具，同时仍能驱动这个
    替身模型。注意 `src/core/llm.py` 中有一个同名类用于
    USE_FAKE_MODEL provider，但那个类接收 `list[str]` 且无法产生工具
    调用，因此无法驱动工具循环。
    """

    def __init__(self, responses: list[BaseMessage]):
        super().__init__(responses=responses)

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def fake_tool_model():
    """返回类而非实例，这样每个测试可以脚本化自己的响应。

    放在这个 conftest 中是因为 tests/ 不是包：tests/ 下任何位置都没有 `__init__.py`，
    且 pyproject.toml 中的 `pythonpath` 只列出了 `src` 和
    `scripts`，因此某个子目录中的测试模块无法从
    同级子目录导入辅助模块。
    """
    return FakeToolModel
