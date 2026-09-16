import os
from unittest.mock import patch

import pytest


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


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test.

    Home-dir vars survive so streamlit AppTest's expanduser() still works;
    clearing USERPROFILE/HOME  crashes first script run with
    "Could not determine home directory."
    """
    kept = {k: v for k, v in os.environ.items() if k in {"HOME", "USERPROFILE", "TMP", "TEMP"}}
    with patch.dict(os.environ, kept, clear=True):
        yield
