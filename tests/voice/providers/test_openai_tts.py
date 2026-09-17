"""OpenAI TTS provider 的测试。"""

from unittest.mock import patch

import pytest

from voice.providers.openai_tts import OpenAITTS


def test_init_with_valid_params(mock_openai_client):
    """测试使用有效参数创建 OpenAITTS。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key", voice="nova", model="tts-1")
        # 如果 client 初始化无错误则通过
        assert tts.client == mock_openai_client


def test_init_with_invalid_voice():
    """测试无效 voice 会抛出 ValueError。"""
    # 如果无效 voice 抛出 ValueError 则通过
    with pytest.raises(ValueError, match="Invalid voice"):
        OpenAITTS(api_key="test-key", voice="invalid", model="tts-1")


def test_init_with_invalid_model():
    """测试无效 model 会抛出 ValueError。"""
    # 如果无效 model 抛出 ValueError 则通过
    with pytest.raises(ValueError, match="Invalid model"):
        OpenAITTS(api_key="test-key", voice="nova", model="invalid")


def test_validate_text_too_short(mock_openai_client):
    """测试短于 MIN_LENGTH 的文本返回 None。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts._validate_and_prepare_text("ab")  # 2 个字符 < MIN_LENGTH (3)
        # 如果过短文本返回 None 则通过
        assert result is None


def test_validate_text_too_long(mock_openai_client):
    """测试长于 MAX_LENGTH 的文本会被截断。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        long_text = "a" * 5000  # 超过 MAX_LENGTH (4096)
        result = tts._validate_and_prepare_text(long_text)
        # 如果文本被截断到 MAX_LENGTH 则通过
        assert result is not None
        assert len(result) == 4096


def test_generate_success(mock_openai_client):
    """测试成功生成音频。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts.generate("Hello world")
        # 如果返回音频字节则通过
        assert result == b"fake audio data"


def test_generate_api_error(mock_openai_client):
    """测试 API 错误被优雅处理。"""
    # 让 mock 抛出异常
    mock_openai_client.audio.speech.create.side_effect = Exception("API Error")
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts.generate("Hello world")
        # 如果返回 None 而不是崩溃则通过
        assert result is None
