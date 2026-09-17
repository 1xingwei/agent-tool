"""语音转文本工厂。

本模块提供一个工厂类，根据配置加载相应的 STT 提供方。
"""

import logging
import os
from typing import BinaryIO, Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "deepgram"]


class SpeechToText:
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

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """使用指定 provider 初始化 STT。

        Args:
            provider: provider 名称（"openai"、"deepgram" 等）
            api_key: API key（未提供时使用环境变量）
            **config: provider 专属配置

        Raises:
            ValueError: 如果 provider 未知
        """
        self._provider_name = provider

        # 从参数或环境变量解析 API key
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置 provider
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"SpeechToText created with provider={provider}")

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
            case "deepgram":
                return os.getenv("DEEPGRAM_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """加载对应的 STT provider 实现。

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
                from voice.providers.openai_stt import OpenAISTT

                return OpenAISTT(api_key=api_key, **config)

            case "deepgram":
                # 未来扩展示例：要添加 Deepgram 支持，实现 DeepgramSTT provider 并取消注释：
                # from voice.providers.deepgram_stt import DeepgramSTT
                # return DeepgramSTT(api_key=api_key, **config)
                raise NotImplementedError("Deepgram STT provider not yet implemented")

            case _:
                # 兜底处理未知 provider
                raise ValueError(f"Unknown STT provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """获取 provider 名称。

        Returns:
            provider 名称字符串
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "SpeechToText | None":
        """从环境变量创建 STT。

        读取 VOICE_STT_PROVIDER 环境变量以确定使用哪个 provider。
        未配置时返回 None。

        Returns:
            SpeechToText 实例或 None

        Example:
            >>> # 在 .env 中：VOICE_STT_PROVIDER=openai
            >>> stt = SpeechToText.from_env()
            >>> if stt:
            ...     text = stt.transcribe(audio_file)
        """
        provider = os.getenv("VOICE_STT_PROVIDER")

        # 如果未设置 provider，语音功能被禁用
        if not provider:
            logger.debug("VOICE_STT_PROVIDER not set, STT disabled")
            return None

        try:
            # 使用环境变量中的 provider 创建实例
            # 校验 provider，若无效则抛出 ValueError
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不崩溃——允许应用在无语音功能的情况下继续运行
            logger.error(f"Failed to create STT provider: {e}", exc_info=True)
            return None

    def transcribe(self, audio_file: BinaryIO) -> str:
        """将音频转写为文本。

        委托给底层 provider 实现。

        Args:
            audio_file: 二进制音频文件

        Returns:
            转写后的文本（失败时为空字符串）
        """
        return self._provider.transcribe(audio_file)
