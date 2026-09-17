import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClient


@pytest.mark.docker
def test_service_with_fake_model():
    """使用 fake 模型测试服务。

    此测试要求服务容器以 USE_FAKE_MODEL=true 运行
    """
    client = AgentClient("http://0.0.0.0", agent="chatbot")
    response = client.invoke("Tell me a joke?", model="fake")
    assert response.type == "ai"
    assert response.content == "This is a test response from the fake model."


@pytest.mark.docker
def test_service_with_app():
    """使用 app 测试服务。

    此测试要求服务容器以 USE_FAKE_MODEL=true 运行
    """
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    assert at.chat_message[0].avatar == "assistant"
    welcome = at.chat_message[0].markdown[0].value
    assert welcome.startswith("Hello! I'm an AI-powered research assistant")
    assert not at.exception

    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value("What is the weather in Tokyo?").run()
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather in Tokyo?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "This is a test response from the fake model."
    assert not at.exception
