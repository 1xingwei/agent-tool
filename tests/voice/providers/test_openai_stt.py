"""OpenAI STT provider 的测试。"""

from unittest.mock import patch

from voice.providers.openai_stt import OpenAISTT


def test_init_with_api_key(mock_openai_client):
    """测试使用 API key 创建 OpenAISTT。"""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        # 若 client 初始化无错误则通过
        assert stt.client == mock_openai_client


def test_transcribe_success(mock_openai_client, mock_audio_file):
    """测试音频转写成功。"""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # 若返回（去除空白后的）转写文本则通过
        assert result == "transcribed text"


def test_transcribe_seeks_file_to_beginning(mock_openai_client, mock_audio_file):
    """测试 transcribe 在读取前会将文件 seek 到开头。"""
    # 移动文件指针以模拟已读取过的文件
    mock_audio_file.seek(100)
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        stt.transcribe(mock_audio_file)
        # 若转写前文件位置被重置为 0 则通过
        assert mock_audio_file.tell() == 0


def test_transcribe_strips_whitespace(mock_openai_client, mock_audio_file):
    """测试转写结果会去除空白。"""
    # Mock API 返回带首尾空白的文本
    mock_openai_client.audio.transcriptions.create.return_value = "  text with spaces  "
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # 若结果中的空白被去除则通过
        assert result == "text with spaces"


def test_transcribe_api_error(mock_openai_client, mock_audio_file):
    """测试 API 错误能被优雅处理。"""
    # 让 mock 抛出异常
    mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # 如果返回空字符串而不是崩溃则通过
        assert result == ""
