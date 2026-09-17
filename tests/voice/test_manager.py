"""VoiceManager 核心逻辑的测试（不含 Streamlit UI 测试）。"""

from unittest.mock import Mock, patch

from voice.manager import VoiceManager


def test_init_with_both_stt_and_tts():
    """测试同时使用 STT 和 TTS 创建 VoiceManager。"""
    mock_stt = Mock()
    mock_tts = Mock()
    manager = VoiceManager(stt=mock_stt, tts=mock_tts)
    # 如果 STT 和 TTS 都被正确赋值则通过
    assert manager.stt == mock_stt
    assert manager.tts == mock_tts


def test_init_with_only_tts():
    """测试仅使用 TTS（STT=None）创建 VoiceManager。"""
    mock_tts = Mock()
    manager = VoiceManager(stt=None, tts=mock_tts)
    # 如果部分语音功能可用（仅 TTS）则通过
    assert manager.stt is None
    assert manager.tts == mock_tts


def test_from_env_both_configured():
    """测试同时配置 STT 和 TTS 时的 from_env。"""
    mock_stt = Mock()
    mock_tts = Mock()
    with patch("voice.manager.SpeechToText.from_env", return_value=mock_stt):
        with patch("voice.manager.TextToSpeech.from_env", return_value=mock_tts):
            manager = VoiceManager.from_env()
            # 如果 VoiceManager 同时使用 STT 和 TTS 创建则通过
            assert manager is not None
            assert manager.stt == mock_stt
            assert manager.tts == mock_tts


def test_from_env_only_tts_configured():
    """测试仅配置 TTS 时的 from_env。"""
    mock_tts = Mock()
    with patch("voice.manager.SpeechToText.from_env", return_value=None):
        with patch("voice.manager.TextToSpeech.from_env", return_value=mock_tts):
            manager = VoiceManager.from_env()
            # 如果 VoiceManager 仅使用 TTS 创建则通过（STT=None 可接受）
            assert manager is not None
            assert manager.stt is None
            assert manager.tts == mock_tts


def test_from_env_neither_configured():
    """测试既未配置 STT 也未配置 TTS 时的 from_env。"""
    with patch("voice.manager.SpeechToText.from_env", return_value=None):
        with patch("voice.manager.TextToSpeech.from_env", return_value=None):
            manager = VoiceManager.from_env()
            # 如果未配置语音功能时返回 None 则通过
            assert manager is None


def test_transcribe_audio_stt_not_configured():
    """测试未配置 STT 时 _transcribe_audio 返回 None。"""
    manager = VoiceManager(stt=None, tts=Mock())
    mock_audio = Mock()
    # mock Streamlit 的 st.error 以避免实际 UI 调用
    with patch("voice.manager.st.error"):
        result = manager._transcribe_audio(mock_audio)
        # 如果返回 None 则通过（未配置 STT 时的防御性检查）
        assert result is None
