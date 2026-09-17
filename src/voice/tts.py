"""文本转语音工厂。

本模块提供一个工厂类，根据配置加载对应的 TTS provider。
"""

import logging
import os
from typing import Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "elevenlabs"]


class TextToSpeech:
    """文本转语音工厂。

    加载并委托给具体的 TTS provider 实现。

    Example:
        >>> tts = TextToSpeech(provider="openai", voice="nova")
        >>> audio = tts.generate("Hello world")
        >>>
        >>> # 或从环境变量创建
        >>> tts = TextToSpeech.from_env()
        >>> if tts:
        ...     audio = tts.generate("Hello world")
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """使用指定 provider 初始化 TTS。

        Args:
            provider: provider 名称（"openai"、"elevenlabs" 等）
            api_key: API key（未提供时使用环境变量）
            **config: provider 专属配置
                OpenAI: voice="alloy", model="tts-1"
                ElevenLabs: voice_id="...", model_id="..."

        Raises:
            ValueError: 如果 provider 未知
        """
        self._provider_name = provider

        # 从参数或环境变量解析 API key
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置 provider
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"TextToSpeech created with provider={provider}")

    def _get_api_key(self, provider: Provider, api_key: str | None) -> str | None:
        """从参数或环境变量获取 API key。

        Args:
            provider: provider 名称
            api_key: 来自参数的 API key（优先）

        Returns:
            解析后的 API key 或 None
        """
        # 如果显式提供了 API key，则使用它
        if api_key:
            return api_key

        # 否则，根据 provider 从环境变量获取
        match provider:
            case "openai":
                return os.getenv("OPENAI_API_KEY")
            case "elevenlabs":
                return os.getenv("ELEVENLABS_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """加载对应的 TTS provider 实现。

        Args:
            provider: provider 名称
            api_key: 解析后的 API key
            config: provider 专属配置

        Returns:
            provider 实例

        Raises:
            ValueError: 如果 provider 未知
            NotImplementedError: 如果 provider 尚未实现
        """
        match provider:
            case "openai":
                from voice.providers.openai_tts import OpenAITTS

                # 提取 OpenAI 专属配置并应用默认值
                voice = config.get("voice", "alloy")
                model = config.get("model", "tts-1")

                return OpenAITTS(api_key=api_key, voice=voice, model=model)

            case "elevenlabs":
                # 未来扩展示例：要添加 ElevenLabs 支持，实现 ElevenLabsTTS provider 并取消注释：
                # from voice.providers.elevenlabs_tts import ElevenLabsTTS
                # voice_id = config.get("voice_id")
                # model_id = config.get("model_id", "eleven_monolingual_v1")
                # return ElevenLabsTTS(api_key=api_key, voice_id=voice_id, model_id=model_id)
                raise NotImplementedError("ElevenLabs TTS provider not yet implemented")

            case _:
                # 兜底处理未知 provider
                raise ValueError(f"Unknown TTS provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """获取 provider 名称。

        Returns:
            provider 名称字符串
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "TextToSpeech | None":
        """从环境变量创建 TTS。

        读取 VOICE_TTS_PROVIDER 环境变量以确定使用哪个 provider。
        未配置时返回 None。

        Returns:
            TextToSpeech 实例或 None

        Example:
            >>> # 在 .env 中：VOICE_TTS_PROVIDER=openai
            >>> tts = TextToSpeech.from_env()
            >>> if tts:
            ...     audio = tts.generate("Hello world")
        """
        provider = os.getenv("VOICE_TTS_PROVIDER")

        # 若未设置 provider，则语音功能被禁用
        if not provider:
            logger.debug("VOICE_TTS_PROVIDER not set, TTS disabled")
            return None

        try:
            # 使用环境变量中的 provider 创建实例
            # 校验 provider，若无效则抛出 ValueError
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不崩溃——允许应用在没有语音的情况下继续运行
            logger.error(f"Failed to create TTS provider: {e}", exc_info=True)
            return None

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
