"""语音提供方实现。"""

from voice.providers.openai_stt import OpenAISTT
from voice.providers.openai_tts import OpenAITTS

# 未来可在此导入其他提供方：
# from voice.providers.deepgram_stt import DeepgramSTT
# from voice.providers.elevenlabs_tts import ElevenLabsTTS

__all__ = ["OpenAISTT", "OpenAITTS"]
