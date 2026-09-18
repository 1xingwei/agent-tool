"""语音 provider 工厂的共享骨架。

`stt.py` 与 `tts.py` 原本各自复制了同一套骨架：
`__init__` → `_get_api_key` → `_load_provider`（分发 + 未实现分支 + 未知兜底）
→ `provider` property → `from_env`（未设即关闭、异常吞掉返回 None）。

两者的实际差异只有四处**数据**：决定启用的环境变量名、provider → API key
环境变量的映射、错误串里的功能标签（STT / TTS）、以及未实现 provider 的提示。
故把这些差异抽成类属性，骨架收敛到本模块这一个维护点。

子类只需声明若干类属性，并实现：

- `_build`：真正构造 provider 实例的那一步（唯一含 provider 专属配置的地方）；
- 自己的委派方法（STT 的 `transcribe`、TTS 的 `generate` / `get_format`）。

新增一个 provider 的完整改法见各子类 `unimplemented` 上方的注释。
"""

import logging
import os
from collections.abc import Mapping
from typing import Any, ClassVar, Self

logger = logging.getLogger(__name__)


class ProviderFactory[P]:
    """语音 provider 工厂的共享骨架。

    Class attributes:
        label: 错误串里的功能标签（"STT" / "TTS"）。
        provider_env_var: 决定启用哪个 provider 的环境变量名。
        api_key_env_vars: provider 名 → 该 provider 的 API key 环境变量名。
        available: 已实现的 provider 名，用于拼 "Available providers: ..." 的后半句。
        unimplemented: provider 名 → 抛出的 `NotImplementedError` 消息。
    """

    label: ClassVar[str]
    provider_env_var: ClassVar[str]
    api_key_env_vars: ClassVar[Mapping[str, str]]
    available: ClassVar[str]
    unimplemented: ClassVar[Mapping[str, str]] = {}

    def __init__(self, provider: str = "openai", api_key: str | None = None, **config: Any) -> None:
        """使用指定 provider 初始化。

        Args:
            provider: provider 名称（"openai" 等，取值见子类的 `available` 与
                `unimplemented`）
            api_key: API key（未提供时使用环境变量）
            **config: provider 专属配置，原样透传给子类的 `_build`

        Raises:
            ValueError: 如果 provider 未知
            NotImplementedError: 如果 provider 尚未实现
        """
        self._provider_name = provider

        # 从参数或环境变量解析 API key
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置 provider
        self._provider: P = self._load_provider(provider, resolved_api_key, config)

        # 类名进日志，保持与拆分前逐字一致（SpeechToText / TextToSpeech）
        logger.info(f"{type(self).__name__} created with provider={provider}")

    def _get_api_key(self, provider: str, api_key: str | None) -> str | None:
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

        # 否则，根据 provider 从环境变量获取；未知 provider 返回 None
        env_var = self.api_key_env_vars.get(provider)
        return os.getenv(env_var) if env_var else None

    def _load_provider(self, provider: str, api_key: str | None, config: dict[str, Any]) -> P:
        """加载对应的 provider 实现。

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
        if provider == self.available:
            return self._build(provider, api_key, config)

        # 已规划但尚未实现的 provider：给出明确的提示而不是当成「未知」
        message = self.unimplemented.get(provider)
        if message:
            raise NotImplementedError(message)

        # 兜底处理未知 provider
        raise ValueError(
            f"Unknown {self.label} provider: {provider}. Available providers: {self.available}"
        )

    def _build(self, provider: str, api_key: str | None, config: dict[str, Any]) -> P:
        """构造 provider 实例。由子类实现。

        Args:
            provider: provider 名称（此时必然等于 `available`）
            api_key: 解析后的 API key
            config: provider 专属配置

        Returns:
            provider 实例
        """
        raise NotImplementedError

    @property
    def provider(self) -> str:
        """获取 provider 名称。

        Returns:
            provider 名称字符串
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> Self | None:
        """从环境变量创建。

        读取 `provider_env_var` 指定的环境变量以确定使用哪个 provider。
        未配置时返回 None。

        Returns:
            工厂实例或 None

        Example:
            >>> # 在 .env 中：VOICE_STT_PROVIDER=openai
            >>> stt = SpeechToText.from_env()
            >>> if stt:
            ...     text = stt.transcribe(audio_file)
        """
        provider = os.getenv(cls.provider_env_var)

        # 如果未设置 provider，语音功能被禁用
        if not provider:
            logger.debug(f"{cls.provider_env_var} not set, {cls.label} disabled")
            return None

        try:
            # 使用环境变量中的 provider 创建实例
            # 校验 provider，若无效则抛出 ValueError
            return cls(provider=provider)
        except Exception as e:
            # 记录错误但不崩溃——允许应用在无语音功能的情况下继续运行
            logger.error(f"Failed to create {cls.label} provider: {e}", exc_info=True)
            return None
