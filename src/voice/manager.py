"""VoiceManager - Streamlit 集成层。

该模块为语音功能提供 Streamlit 专属的 UI 集成。
所有 Streamlit 依赖都隔离在此处。
"""

import logging
from typing import Optional

import streamlit as st

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """语音功能的 Streamlit 便捷层。

    该类为语音输入/输出提供 Streamlit 专属方法。
    它处理 UI 反馈（spinner、错误），同时将实际的
    语音处理委托给 STT 和 TTS 模块。

    示例：
        >>> voice = VoiceManager.from_env()
        >>>
        >>> if voice:
        ...     user_input = voice.get_chat_input()
        ...     if user_input:
        ...         with st.chat_message("ai"):
        ...             voice.render_message("Hello!")
    """

    def __init__(self, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        """初始化 VoiceManager。

        Args:
            stt: SpeechToText 实例（None 表示禁用 STT）
            tts: TextToSpeech 实例（None 表示禁用 TTS）
        """
        self.stt = stt
        self.tts = tts

        logger.info(
            f"VoiceManager: STT={'enabled' if stt else 'disabled'}, "
            f"TTS={'enabled' if tts else 'disabled'}"
        )

    @classmethod
    def from_env(cls) -> Optional["VoiceManager"]:
        """从环境变量创建 VoiceManager。

        读取 VOICE_STT_PROVIDER 和 VOICE_TTS_PROVIDER 以配置
        语音转文本和文本转语音 provider。

        Returns:
            若 STT 或 TTS 任一已配置则返回 VoiceManager，否则返回 None

        示例：
            >>> # 在 .env 中：
            >>> # VOICE_STT_PROVIDER=openai
            >>> # VOICE_TTS_PROVIDER=openai
            >>>
            >>> voice = VoiceManager.from_env()
            >>> # 返回已配置的 VoiceManager，若禁用则返回 None
        """
        # 从环境创建 STT 和 TTS
        stt = SpeechToText.from_env()
        tts = TextToSpeech.from_env()

        # 若两者均禁用，返回 None（无语音功能）
        if not stt and not tts:
            logger.debug("Voice features not configured")
            return None

        return cls(stt=stt, tts=tts)

    def _transcribe_audio(self, audio) -> str | None:
        """转写音频并给出 UI 反馈。

        转写期间显示 spinner，失败时显示错误信息。

        Args:
            audio: 来自 Streamlit 聊天输入的音频文件对象

        Returns:
            转写后的文本，若转写失败则返回 None
        """
        # 防御性检查（若调用正确则不应发生）
        if not self.stt:
            st.error("⚠️ Speech-to-text not configured.")
            return None

        # 转写期间显示 spinner
        with st.spinner("🎤 Transcribing audio..."):
            transcribed = self.stt.transcribe(audio)

        # 检查转写是否成功
        if not transcribed:
            st.error("⚠️ Transcription failed. Please try again or type your message.")
            return None

        return transcribed

    def get_chat_input(self, placeholder: str = "Your message") -> str | None:
        """获取聊天输入，可选语音转写。

        处理 Streamlit UI，包括音频输入组件和转写
        反馈（spinner、错误）。

        Args:
            placeholder: 输入框的占位文本

        Returns:
            用户消息（若为音频则转写，否则为文本），若无输入则返回 None
        """
        # 无 STT - 使用常规文本输入
        if not self.stt:
            return st.chat_input(placeholder)

        # 已启用 STT - 使用支持音频的输入
        chat_value = st.chat_input(placeholder, accept_audio=True)

        if not chat_value:
            return None

        # 处理字符串返回值（仅文本输入）
        if isinstance(chat_value, str):
            return chat_value

        # 处理对象/dict 返回值（支持音频的输入）
        # 提取文本 - 同时支持属性和 dict 访问
        text_content = None
        if hasattr(chat_value, "text"):
            text_content = chat_value.text
        elif isinstance(chat_value, dict):
            text_content = chat_value.get("text", "")

        # 提取音频 - 同时支持属性和 dict 访问
        audio_content = None
        if hasattr(chat_value, "audio"):
            audio_content = chat_value.audio
        elif isinstance(chat_value, dict):
            audio_content = chat_value.get("audio")

        # 若提供了音频，则进行转写
        if audio_content:
            return self._transcribe_audio(audio_content)

        # 若无音频，返回文本内容
        if text_content:
            return text_content

        # 未提供文本或音频
        return None

    def render_message(self, content: str, container=None, audio_only: bool = False) -> None:
        """渲染消息，可选 TTS 音频。

        处理 Streamlit UI，包括文本显示和音频播放器。
        将生成的音频保存到 session state，使其在重运行间持久化。

        Args:
            content: 要显示的消息内容
            container: Streamlit 容器（默认为当前上下文）
            audio_only: 若为 True，仅渲染音频（文本已显示）
        """
        if container is None:
            container = st

        # 除非处于 audio_only 模式（用于文本已显示的流式场景），否则显示文本
        if not audio_only:
            container.write(content)

        # 若启用 TTS 且内容非空，则添加音频
        if self.tts and content.strip():
            # 生成音频时显示占位符
            placeholder = container.empty()
            with placeholder:
                st.caption("🎙️ Generating audio...")

            # 生成 TTS 音频
            audio = self.tts.generate(content)

            # 将音频保存到 session state 中，用于最后一条 AI 消息
            # 这样它就能在 st.rerun() 调用间持久化
            if audio:
                st.session_state.last_audio = {"data": audio, "format": self.tts.get_format()}

            # 用音频播放器或错误消息替换占位符
            if audio:
                placeholder.audio(audio, format=self.tts.get_format())
            else:
                placeholder.caption("🔇 Audio generation unavailable")
