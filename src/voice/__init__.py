"""语音输入/输出模块。

提供语音转文本和文本转语音能力，支持
多种 provider。

模块：
    - SpeechToText：STT 处理器（可独立使用）
    - TextToSpeech：TTS 处理器（可独立使用）
    - VoiceManager：Streamlit 便捷封装

快速开始：
    >>> from voice import VoiceManager
    >>>
    >>> # 简单方式：从环境创建
    >>> voice = VoiceManager.from_env()
    >>>
    >>> # 在 Streamlit 中使用
    >>> if voice:
    ...     user_input = voice.get_chat_input()
    ...     # ... 处理输入 ...
    ...     with st.chat_message("ai"):
    ...         voice.render_message(response)

高级用法：
    >>> from voice import SpeechToText, TextToSpeech, VoiceManager
    >>>
    >>> # 混合 provider：OpenAI STT + 自定义 TTS
    >>> stt = SpeechToText(provider="openai")
    >>> tts = TextToSpeech(provider="openai", voice="nova")
    >>> voice = VoiceManager(stt=stt, tts=tts)
"""

from voice.manager import VoiceManager
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

__all__ = ["VoiceManager", "SpeechToText", "TextToSpeech"]
