"""TextToSpeech 工厂类的测试。"""

import os
from unittest.mock import patch

import pytest

from voice.tts import TextToSpeech


def test_init_with_openai_provider(mock_openai_client):
    """测试使用 openai provider 和显式 API key 创建 TTS。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = TextToSpeech(provider="openai", api_key="test-key")
        # 如果 provider 属性返回 "openai" 则通过
        assert tts.provider == "openai"


def test_init_with_invalid_provider():
    """测试无效 provider 会抛出 ValueError。"""
    # 如果抛出带有预期消息的 ValueError 则通过
    with pytest.raises(ValueError, match="Unknown TTS provider: invalid"):
        TextToSpeech(provider="invalid", api_key="test-key")


def test_init_with_unimplemented_provider():
    """测试未实现的 provider 会抛出 NotImplementedError。"""
    # 若抛出 NotImplementedError 则通过（elevenlabs 尚未实现）
    with pytest.raises(NotImplementedError, match="ElevenLabs TTS provider not yet implemented"):
        TextToSpeech(provider="elevenlabs", api_key="test-key")


def test_from_env_provider_not_set():
    """测试当 VOICE_TTS_PROVIDER 未设置时 from_env 返回 None。"""
    with patch.dict(os.environ, {}, clear=True):
        result = TextToSpeech.from_env()
        # 若环境变量未设置时返回 None 则通过
        assert result is None


def test_from_env_valid_provider(mock_openai_client):
    """测试 from_env 使用有效 provider 创建 TTS 实例。"""
    with patch.dict(os.environ, {"VOICE_TTS_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
        with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
            tts = TextToSpeech.from_env()
            # 若使用正确的 provider 创建 TextToSpeech 实例则通过
            assert tts is not None
            assert tts.provider == "openai"


def test_from_env_invalid_provider_returns_none():
    """测试 from_env 对无效 provider 返回 None（并记录错误日志）。"""
    with patch.dict(os.environ, {"VOICE_TTS_PROVIDER": "invalid"}):
        result = TextToSpeech.from_env()
        # 若返回 None 而非崩溃则通过
        assert result is None


def test_get_api_key_from_param(mock_openai_client):
    """测试显式 api_key 参数优先于环境变量。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        with patch(
            "voice.providers.openai_tts.OpenAI", return_value=mock_openai_client
        ) as mock_openai:
            TextToSpeech(provider="openai", api_key="param-key")
            # 若 OpenAI 客户端使用参数 key（而非环境 key）初始化则通过
            mock_openai.assert_called_once_with(api_key="param-key")


def test_get_api_key_from_env(mock_openai_client):
    """测试未提供时从环境加载 API key。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        with patch(
            "voice.providers.openai_tts.OpenAI", return_value=mock_openai_client
        ) as mock_openai:
            TextToSpeech(provider="openai")
            # 若 OpenAI 客户端使用环境 key 初始化则通过
            mock_openai.assert_called_once_with(api_key="env-key")
