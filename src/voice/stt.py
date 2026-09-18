"""语音转文本工厂。

本模块提供一个工厂类，根据配置加载相应的 STT 提供方。
共享骨架在 `voice.factory`，这里只声明 STT 的差异数据与委派方法。
"""

from collections.abc import Mapping
from typing import Any, BinaryIO, ClassVar, Protocol

from voice.factory import ProviderFactory


class _STTProvider(Protocol):
    """`SpeechToText` 所委派对象的结构约定。"""

    def transcribe(self, audio_file: BinaryIO) -> str: ...


class SpeechToText(ProviderFactory[_STTProvider]):
    """语音转文本工厂。

    加载并委托给具体的 STT 提供方实现。

    Example:
        >>> stt = SpeechToText(provider="openai")
        >>> text = stt.transcribe(audio_file)
        >>>
        >>> # 或从环境变量
        >>> stt = SpeechToText.from_env()
        >>> if stt:
        ...     text = stt.transcribe(audio_file)
    """

    label: ClassVar[str] = "STT"
    provider_env_var: ClassVar[str] = "VOICE_STT_PROVIDER"
    api_key_env_vars: ClassVar[Mapping[str, str]] = {
        "openai": "OPENAI_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
    }
    available: ClassVar[str] = "openai"

    # 未来扩展示例：要添加 Deepgram 支持，实现 DeepgramSTT provider 并取消注释：
    # from voice.providers.deepgram_stt import DeepgramSTT
    # return DeepgramSTT(api_key=api_key, **config)
    # 然后把它从下面的 unimplemented 删掉（`_build` 里按 provider 分发）。
    unimplemented: ClassVar[Mapping[str, str]] = {
        "deepgram": "Deepgram STT provider not yet implemented"
    }

    def _build(self, provider: str, api_key: str | None, config: dict[str, Any]) -> _STTProvider:
        """构造 OpenAI STT provider。"""
        from voice.providers.openai_stt import OpenAISTT

        return OpenAISTT(api_key=api_key, **config)

    def transcribe(self, audio_file: BinaryIO) -> str:
        """将音频转写为文本。

        委托给底层 provider 实现。

        Args:
            audio_file: 二进制音频文件

        Returns:
            转写后的文本（失败时为空字符串）
        """
        return self._provider.transcribe(audio_file)
