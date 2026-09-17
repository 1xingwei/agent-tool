"""voice 模块测试的共享 fixture。"""

import io
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_openai_client():
    """用于 TTS/STT 测试的 mock OpenAI 客户端。"""
    client = Mock()

    # Mock TTS 响应（返回带 .content 属性的对象）
    mock_audio_response = Mock()
    mock_audio_response.content = b"fake audio data"
    client.audio.speech.create.return_value = mock_audio_response

    # Mock STT 响应（response_format="text" 时直接返回字符串）
    client.audio.transcriptions.create.return_value = "transcribed text"

    return client


@pytest.fixture
def mock_audio_file():
    """用于 STT 测试的 BytesIO mock 音频文件。"""
    return io.BytesIO(b"fake audio bytes")
