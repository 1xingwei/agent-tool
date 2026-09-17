"""OpenAI Whisper 语音转文本实现。"""

import logging
from typing import BinaryIO

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAISTT:
    """OpenAI Whisper STT 提供方。"""

    def __init__(self, api_key: str | None = None):
        """初始化 OpenAI STT。

        Args:
            api_key: OpenAI API key（未提供时使用环境变量）

        Raises:
            Exception: 若 OpenAI 客户端初始化失败
        """
        # 使用提供的 key 或从环境变量创建 OpenAI 客户端
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info("OpenAI STT initialized")

    def transcribe(self, audio_file: BinaryIO) -> str:
        """使用 OpenAI Whisper 转录音频。

        Args:
            audio_file: 二进制音频文件

        Returns:
            转录文本（失败时为空字符串）

        Note:
            错误会被记录但不会抛出——而是返回空字符串。
            这允许在面向用户的应用程序中优雅降级。
        """
        try:
            # 将文件指针重置到开头（可能已在别处被读取）
            audio_file.seek(0)

            # 调用 OpenAI Whisper API 进行转录
            result = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

            # 清理结果中的空白字符
            transcribed = result.strip()
            logger.info(f"OpenAI STT: transcribed {len(transcribed)} chars")
            return transcribed

        except Exception as e:
            # 记录错误及完整 traceback 以便调试
            logger.error(f"OpenAI STT failed: {e}", exc_info=True)
            # 返回空字符串以允许优雅降级
            return ""
