"""OpenAI 文本转语音实现。"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAITTS:
    """OpenAI TTS 提供方。"""

    # API 约束
    MAX_TEXT_LENGTH = 4096
    MIN_TEXT_LENGTH = 3

    # 可用配置选项
    VALID_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    VALID_MODELS = ["tts-1", "tts-1-hd"]

    def __init__(self, api_key: str | None = None, voice: str = "alloy", model: str = "tts-1"):
        """初始化 OpenAI TTS。

        Args:
            api_key: OpenAI API key（未提供时使用环境变量）
            voice: 语音名称（alloy、echo、fable、onyx、nova、shimmer）
            model: 模型名称（tts-1 或 tts-1-hd）

        Raises:
            ValueError: 若 voice 或 model 无效
            Exception: 若 OpenAI 客户端初始化失败
        """
        # 校验 voice 参数
        if voice not in self.VALID_VOICES:
            raise ValueError(f"Invalid voice '{voice}'. Must be one of {self.VALID_VOICES}")

        # 校验 model 参数
        if model not in self.VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Must be one of {self.VALID_MODELS}")

        # 使用提供的 key 或从环境变量创建 OpenAI 客户端
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.voice = voice
        self.model = model

        logger.info(f"OpenAI TTS initialized: voice={voice}, model={model}")

    def _validate_and_prepare_text(self, text: str) -> str | None:
        """校验并准备用于 TTS 生成的文本。

        Args:
            text: 原始文本输入

        Returns:
            准备好用于 TTS 的文本，若文本过短则为 None

        Note:
            - 去除空白字符
            - 若文本低于最小长度则返回 None
            - 若文本超过最大长度则截断
        """
        # 去除首尾空白字符
        text = text.strip()

        # 跳过过短文本（不值得调用 API）
        if len(text) < self.MIN_TEXT_LENGTH:
            logger.debug(f"OpenAI TTS: skipping short text ({len(text)} chars)")
            return None

        # 若需要则截断至 API 限制
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(
                f"OpenAI TTS: truncating from {len(text)} to {self.MAX_TEXT_LENGTH} chars"
            )
            text = text[: self.MAX_TEXT_LENGTH]

        return text

    def generate(self, text: str) -> bytes | None:
        """从文本生成语音。

        Args:
            text: 要转换为语音的文本

        Returns:
            MP3 音频字节，若文本过短或生成失败则为 None

        Note:
            - 文本短于 3 个字符返回 None
            - 文本长于 4096 个字符会被截断
            - 错误会被记录但不会抛出——而是返回 None
        """
        # 校验并准备文本
        prepared_text = self._validate_and_prepare_text(text)
        if not prepared_text:
            return None

        try:
            # 调用 OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=prepared_text,
                response_format="mp3",
            )

            # 从响应中提取音频字节
            audio_bytes = response.content
            logger.info(f"OpenAI TTS: generated {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            # 记录错误及完整 traceback 以便调试
            logger.error(f"OpenAI TTS failed: {e}", exc_info=True)
            # 返回 None 以允许优雅降级
            return None

    def get_format(self) -> str:
        """获取音频格式（MIME 类型）。

        Returns:
            生成音频的 MIME 类型字符串
        """
        return "audio/mp3"
