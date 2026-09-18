"""文本转语音工厂。

本模块提供一个工厂类，根据配置加载对应的 TTS provider。
共享骨架在 `voice.factory`，这里只声明 TTS 的差异数据与委派方法。
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

from voice.factory import ProviderFactory


class _TTSProvider(Protocol):
    """`TextToSpeech` 所委派对象的结构约定。"""

    def generate(self, text: str) -> bytes | None: ...

    def get_format(self) -> str: ...


class TextToSpeech(ProviderFactory[_TTSProvider]):
    """文本转语音工厂。

    加载并委托给具体的 TTS provider 实现。

    专属配置通过关键字传入，由 `_build` 抽取并套用默认值：

    - OpenAI：`voice="alloy"`、`model="tts-1"`
    - ElevenLabs：`voice_id="..."`、`model_id="..."`

    Example:
        >>> tts = TextToSpeech(provider="openai", voice="nova")
        >>> audio = tts.generate("Hello world")
        >>>
        >>> # 或从环境变量创建
        >>> tts = TextToSpeech.from_env()
        >>> if tts:
        ...     audio = tts.generate("Hello world")
    """

    label: ClassVar[str] = "TTS"
    provider_env_var: ClassVar[str] = "VOICE_TTS_PROVIDER"
    api_key_env_vars: ClassVar[Mapping[str, str]] = {
        "openai": "OPENAI_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
    }
    available: ClassVar[str] = "openai"

    # 未来扩展示例：要添加 ElevenLabs 支持，实现 ElevenLabsTTS provider 并取消注释：
    # from voice.providers.elevenlabs_tts import ElevenLabsTTS
    # voice_id = config.get("voice_id")
    # model_id = config.get("model_id", "eleven_monolingual_v1")
    # return ElevenLabsTTS(api_key=api_key, voice_id=voice_id, model_id=model_id)
    # 然后把它从下面的 unimplemented 删掉（`_build` 里按 provider 分发）。
    unimplemented: ClassVar[Mapping[str, str]] = {
        "elevenlabs": "ElevenLabs TTS provider not yet implemented"
    }

    def _build(self, provider: str, api_key: str | None, config: dict[str, Any]) -> _TTSProvider:
        """构造 OpenAI TTS provider，并套用 OpenAI 专属配置的默认值。"""
        from voice.providers.openai_tts import OpenAITTS

        # 提取 OpenAI 专属配置并应用默认值
        voice = config.get("voice", "alloy")
        model = config.get("model", "tts-1")

        return OpenAITTS(api_key=api_key, voice=voice, model=model)

    def generate(self, text: str) -> bytes | None:
        """从文本生成语音。

        委托给底层 provider 实现。

        Args:
            text: 要转换为语音的文本

        Returns:
            音频字节（格式取决于 provider），失败时为 None
        """
        return self._provider.generate(text)

    def get_format(self) -> str:
        """获取此 provider 的音频格式（MIME 类型）。

        Returns:
            MIME 类型字符串（例如 "audio/mp3"）
        """
        return self._provider.get_format()
