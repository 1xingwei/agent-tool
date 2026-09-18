import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage, UserThreads
from schema.task_data import TaskData
from voice import VoiceManager

# 一个 Streamlit 应用，通过简单的聊天界面与 langgraph agent 交互。
# 该应用有三个主要函数，均以异步方式运行：

# - main() - 设置 streamlit 应用及高层结构
# - draw_messages() - 绘制一组聊天消息——要么重放已有消息，
#   要么流式渲染新消息。
# - handle_feedback() - 绘制反馈组件并记录用户反馈。

# 该应用大量使用 AgentClient 与 agent 的 FastAPI 端点交互。


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


class TaskDataStatus:
    """把 bg-task-agent 的自定义数据渲染成 Streamlit 状态栏（UI 层类，归属调用方）。"""

    def __init__(self) -> None:
        self.status = st.status("")
        self.current_task_data: dict[str, TaskData] = {}

    def add_and_draw_task_data(self, task_data: TaskData) -> None:
        status = self.status
        status_str = f"Task **{task_data.name}** "
        match task_data.state:
            case "new":
                status_str += "has :blue[started]. Input:"
            case "running":
                status_str += "wrote:"
            case "complete":
                if task_data.result == "success":
                    status_str += ":green[completed successfully]. Output:"
                else:
                    status_str += ":red[ended with error]. Output:"
        status.write(status_str)
        status.write(task_data.data)
        status.write("---")
        if task_data.run_id not in self.current_task_data:
            # 状态标签始终显示最近新启动的任务
            status.update(label=f"""Task: {task_data.name}""")
        self.current_task_data[task_data.run_id] = task_data
        if all(entry.completed() for entry in self.current_task_data.values()):
            # 若有任何任务出错，状态为 "error"
            if any(entry.completed_with_error() for entry in self.current_task_data.values()):
                state = "error"
            # 若所有任务均成功完成，状态为 "complete"
            else:
                state = "complete"
        # 在所有任务完成前，状态为 "running"
        else:
            state = "running"
        status.update(state=state)  # type: ignore[arg-type]


def tool_calls_visible() -> bool:
    """是否应在聊天中渲染工具调用。

    工具调用属于实现细节：用户提出问题，想要的是答案，
    而不是背后的机制。因此默认隐藏，仅在侧边栏开关启用时才绘制
    （用于调试和演示）。

    子 agent 转移遵循同样的开关。它们同样以工具调用的形式到达，而像
    ``transfer_to_research_expert`` 这样的原始名称对用户而言，并不比
    任何其他内部函数名更有意义。
    """
    return bool(st.session_state.get("show_tool_calls", False))


def get_or_create_user_id() -> str:
    """从 session state 或 URL 参数中获取用户 ID，若不存在则创建一个新的。"""
    # 检查 session state 中是否存在 user_id
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # 尝试使用新的 st.query_params 从 URL 参数中获取
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # 若未找到则生成新的 user_id
    user_id = str(uuid.uuid4())

    # 存入 session state 供本次会话使用
    st.session_state[USER_ID_COOKIE] = user_id

    # 同时添加到 URL 参数，以便收藏或分享
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


@st.cache_data(ttl=600, show_spinner=False)
def fetch_user_threads_cached(
    base_url: str, user_id: str, agent_id: str | None = None, limit: int = 20
) -> UserThreads:
    """
    使用新的同步方法 get_user_threads 获取并缓存用户 thread。
    """
    client = AgentClient(base_url=base_url, get_info=False)
    return client.get_user_threads(user_id=user_id, agent=agent_id, limit=limit)


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # 隐藏 streamlit 右上角的界面元素
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # 获取或创建用户 ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            # HOST 是绑定地址：0.0.0.0 表示「所有接口」，并非可连接的目标
            # （Windows 会以 WinError 10049 失败），因此在用于构建客户端 URL 之前，
            # 通配地址必须转换为回环地址。
            if host in {"0.0.0.0", "::"}:
                host = "127.0.0.1"
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("正在连接 Agent 服务..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"连接 Agent 服务失败（{agent_url}）：{e}")
            st.markdown("服务可能还在启动中，请几秒后再试。")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # 初始化语音管理器（每个会话一次）
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            # 从 URL 读取 agent，以便通过创建该 thread 的图来获取历史记录。
            resume_agent = st.query_params.get("agent") or agent_client.agent
            try:
                messages: ChatHistory = agent_client.get_history(
                    thread_id=thread_id, agent=resume_agent
                ).messages
            except AgentClientError:
                st.error("未找到该会话 ID 对应的历史消息。")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # 将 thread_id 保留在 URL 中，使地址栏可直接分享。
    st.query_params["thread_id"] = st.session_state.thread_id

    # 配置选项
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "基于 LangGraph、FastAPI、Streamlit 构建的 AI Agent 服务全功能工具包"
        ""

        if st.button(":material/chat: 新对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # 开始新聊天时清除已保存的音频
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.expander(":material/history: 历史对话", expanded=False):
            try:
                url_agent = st.query_params.get("agent")
                if url_agent in [a.key for a in agent_client.info.agents]:
                    agent_client.agent = url_agent
                else:
                    agent_client.agent = agent_client.info.default_agent
                user_threads = fetch_user_threads_cached(
                    base_url=agent_client.base_url,
                    user_id=user_id,
                    agent_id=agent_client.agent,
                    limit=20,
                )
                thread_list = user_threads.threads
            except Exception as e:
                st.caption(f"无法加载对话历史：{e}")
                thread_list = []

            for t in thread_list:
                label = t.title or f"对话 {t.thread_id[:8]}"
                if st.button(label, key=f"thread_{t.thread_id}", use_container_width=True):
                    try:
                        history: ChatHistory = agent_client.get_history(
                            thread_id=t.thread_id, agent=t.agent_id
                        )
                    except AgentClientError:
                        st.error("无法加载该对话。")
                        continue
                    st.session_state.messages = history.messages
                    st.session_state.thread_id = t.thread_id
                    st.query_params["thread_id"] = t.thread_id
                    if "last_audio" in st.session_state:
                        del st.session_state.last_audio
                    st.rerun()

        with st.popover(":material/settings: 设置", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("使用的模型", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            # 将选择同步到 ?agent= URL 参数（为默认值时移除）。
            agent_client.agent = st.selectbox(
                "使用的 Agent",
                options=agent_list,
                index=agent_idx,
                key="agent",
                bind="query-params",
                on_change=fetch_user_threads_cached.clear,
            )
            use_streaming = st.toggle("流式输出", value=True)
            # 带回调的音频开关：关闭时清除缓存的音频
            enable_audio = st.toggle(
                "生成语音",
                value=True,
                disabled=not voice or not voice.tts,
                help="在 .env 中配置 VOICE_TTS_PROVIDER 后启用"
                if not voice or not voice.tts
                else None,
                on_change=lambda: (
                    st.session_state.pop("last_audio", None)
                    if not st.session_state.get("enable_audio", True)
                    else None
                ),
                key="enable_audio",
            )
            # 工具调用属于实现细节，因此除非用户主动启用，否则保持隐藏。
            # 参见 tool_calls_visible() 和 draw_messages()。
            st.toggle(
                "显示工具调用",
                value=False,
                key="show_tool_calls",
                help="仅供调试与演示：开启后展示每次工具调用、子 Agent 转交的输入与输出；"
                "关闭时只显示最终回复",
            )

            # 显示用户 ID（用于调试或用户信息）
            st.text_input("用户 ID（只读）", value=user_id, disabled=True)

        @st.dialog("架构图")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[在 GitHub 查看完整大图](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "应用托管于 [Streamlit Cloud](https://share.streamlit.io/)，FastAPI 服务运行于 [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: 架构图", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: 隐私说明", use_container_width=True):
            st.write(
                "本应用中的提示词、回复和反馈会被匿名记录并保存到 LangSmith，仅用于产品评估与改进。"
            )

        @st.dialog("分享/恢复对话")
        def share_chat_dialog() -> None:
            # st.context.url 是浏览器 URL（已去除查询字符串）。重建
            # 参数，包含 agent，以便 thread 通过正确的图恢复。
            if not st.context.url:
                st.error("无法确定应用地址以生成分享链接。")
                return
            query = urllib.parse.urlencode(
                {
                    "thread_id": st.session_state.thread_id,
                    "agent": agent_client.agent,
                    USER_ID_COOKIE: user_id,
                }
            )
            chat_url = f"{st.context.url}?{query}"
            st.markdown(f"**对话链接：**\n```text\n{chat_url}\n```")
            st.info("复制以上链接以分享或稍后恢复该对话")

        if st.button(":material/upload: 分享/恢复对话", use_container_width=True):
            share_chat_dialog()

        "[查看源代码](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "由 [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) 在 Oakland 用 :material/favorite: 制作"
        )

    # 绘制已有消息
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "你好！我是一个简单聊天机器人，有什么想问的尽管说！"
            case "interrupt-agent":
                WELCOME = "你好！我是一个中断型 Agent，告诉我你的生日，我来帮你预测性格！"
            case "research-assistant":
                WELCOME = "你好！我是带联网搜索和计算器功能的 AI 研究助手，有什么想问的尽管问！"
            case "loop-agent":
                WELCOME = "你好！我是一个 ReAct 循环 Agent：会反复「思考 → 调用工具 → 观察结果」直到把你的事情办妥。试试让我算点复杂的或查点资料？"
            case "code-reviewer":
                WELCOME = "你好！我是代码库审查助手：用只读工具分析 git 历史与源码。试试让我「review 最近几次提交」或「定位某个函数在哪里实现」？"
            case "rag-assistant":
                WELCOME = """你好！我是能检索公司手册的 AI 助手，可以帮你查询员工手册里的福利、远程办公、休假政策、公司价值观等信息。有什么想问的尽管问！"""
            case _:
                WELCOME = "你好！我是一个 AI Agent，有什么想问的尽管说！"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() 期望一个消息的异步迭代器
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # 为最后一条 AI 消息渲染已保存的音频（若存在）
    # 这确保音频在 st.rerun() 调用之间持久保留
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # 若用户提供了新输入则生成新消息
    # 若可用则使用语音管理器，否则回退到常规输入
    # 必需：在应用 .env（而非服务 .env）中设置 VOICE_STT_PROVIDER、VOICE_TTS_PROVIDER、OPENAI_API_KEY
    # 以启用语音功能。
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input()

    if user_input:
        is_first_message = len(messages) == 0
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # 为流式响应生成 TTS 音频
                # 注意：draw_messages() 将最终消息存入 st.session_state.messages，
                # 并将容器引用存入 st.session_state.last_message
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # 仅为有内容的 AI 响应生成音频
                    if last_msg.type == "ai" and last_msg.content:
                        # 使用 audio_only=True，因为文本已由 draw_messages() 流式输出
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # 渲染带可选语音的 AI 响应
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            if is_first_message:
                fetch_user_threads_cached.clear()
            st.rerun()  # 清除过期的容器
        except AgentClientError as e:
            st.error(f"生成回复时出错：{e}")
            st.stop()

    # 若已生成消息，则显示反馈组件
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    绘制一组聊天消息——要么重放已有消息，
    要么流式输出新消息。

    此函数包含额外逻辑来处理流式 token 和工具调用。
    - 使用占位容器在流式 token 到达时渲染它们。
    - 使用状态容器渲染工具调用，但仅在用户选择查看时
      （参见 tool_calls_visible()）。跟踪工具输入和输出并
      相应地更新状态容器。保持隐藏的调用仍会从流中读取，
      以保持消息对齐，并在其运行期间用临时提示代替它们。

    此函数还需要跟踪会话状态中的最后一个消息容器，
    因为后续消息可能绘制到同一容器。这也用于
    在最新聊天消息中绘制反馈组件。

    Args:
        messages_aiter: 要绘制的消息的异步迭代器。
        is_new: 消息是否为新增。
    """

    # 跟踪最后一个消息容器
    last_message_type = None
    st.session_state.last_message = None

    # 中间流式 token 的占位符
    streaming_content = ""
    streaming_placeholder = None

    # 遍历消息并绘制它们
    while msg := await anext(messages_agen, None):
        # str 消息表示正在流式输出的中间 token
        if isinstance(msg, str):
            # 如果占位符为空，这是新消息的第一个 token
            # 正在流式输出。我们需要进行设置。
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"意外的消息类型：{type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # 来自用户的消息，最简单的情况
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # 来自 agent 的消息是最复杂的情况，因为我们需要
            # 处理流式 token 和工具调用。
            case "ai":
                # 如果正在渲染新消息，将消息存储到会话状态中
                if is_new:
                    st.session_state.messages.append(msg)

                # 如果最后一条消息类型不是 AI，创建新的聊天消息
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # 如果消息有内容，将其写出。
                    # 重置流式变量，为下一条消息做准备。
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # 工具调用和子 agent 转移都是实现
                        # 细节，因此它们共用一个开关，默认隐藏。
                        # 在细节隐藏时，调用仍必须在
                        # 下方被消费，否则流将失去对齐。
                        details = tool_calls_visible()
                        transfers = [tc for tc in msg.tool_calls if "transfer_to" in tc["name"]]
                        regular_calls = [
                            tc for tc in msg.tool_calls if "transfer_to" not in tc["name"]
                        ]
                        status_state = "running" if is_new else "complete"

                        # 按 ID 跟踪每个工具调用的状态容器，以便结果
                        # 映射回正确的容器。
                        call_results = {}
                        if details:
                            for tool_call in transfers:
                                call_results[tool_call["id"]] = st.status(
                                    f"""💼 子 Agent：{tool_call["name"]}""",
                                    state=status_state,
                                )

                            if regular_calls:
                                # 每条消息一个容器，而不是每次调用一个，这样
                                # 一轮包含多次查找仍显示为单行。
                                if len(regular_calls) == 1:
                                    label = f"""🛠️ 工具调用：{regular_calls[0]["name"]}"""
                                else:
                                    label = f"🛠️ 工具调用（{len(regular_calls)}）"
                                aggregate = st.status(label, state=status_state, type="compact")
                                for tool_call in regular_calls:
                                    call_results[tool_call["id"]] = aggregate

                        # 在细节隐藏时，工具运行期间没有可观察的内容，
                        # 因此显示一个临时提示，而不是让消息
                        # 看起来停滞。工具返回后清除它。
                        running_hint = st.empty() if is_new and not details else None
                        if running_hint:
                            running_hint.caption(":material/progress_activity: 正在处理…")

                        # 预期每个工具调用对应一个 ToolMessage。
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results.get(tool_call["id"])
                                if status is not None:
                                    status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # 只有非转移工具调用会到达此处。即使调用
                            # 未显示，结果也必须被消费。
                            status = call_results.get(tool_call["id"])
                            if status is not None:
                                if len(regular_calls) > 1:
                                    status.write(f"**{tool_call['name']}**")
                                status.write("输入：")
                                status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"意外的消息类型：{tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # 如果消息是新的则记录它，并用结果更新
                            # 正确的状态容器
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if status is None:
                                continue
                            if tool_result.tool_call_id:
                                status = call_results.get(tool_result.tool_call_id, status)
                            status.write("输出：")
                            status.write(tool_result.content)
                            status.update(state="complete")

                        if running_hint:
                            running_hint.empty()

            case "custom":
                # bg-task-agent 使用的 CustomData 示例
                # 参见：
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("收到来自 Agent 的意外自定义数据")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # 遇到意外消息类型时，记录错误并停止
            case _:
                st.error(f"意外的消息类型：{msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """绘制反馈组件并记录用户的反馈。"""

    # 跟踪最后发送的反馈，避免发送重复项
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # 如果反馈值或运行 ID 发生变化，发送新的反馈记录
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # 将反馈值（索引）归一化为 0 到 1 之间的分数
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "页内人工反馈"},
            )
        except AgentClientError as e:
            st.error(f"记录反馈时出错：{e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("反馈已记录", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    此函数将 agent 输出隔离到状态容器中。
    它处理初始工具调用消息之后的所有消息，
    直到到达最终 AI 消息。

    增强以支持嵌套多 agent 层级及交接返回消息。

    当工具调用隐藏时，调用方传入 ``status=None``：子 agent
    消息仍会被消费以保持流对齐，只是不
    绘制。

    Args:
        messages_agen: 消息的异步生成器
        status: 当前 agent 的状态容器，或当工具调用
            隐藏时为 None
        is_new: 消息是新增还是重放
    """
    nested_popovers = {}

    # 查找转移 Success 工具调用消息
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # 继续读取，直到获得显式的交接返回
    while True:
        # 读取下一条消息
        sub_msg = await anext(messages_agen)

        # 这应该只在 skip_stream 标志被移除时发生
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # 处理带嵌套弹出框的工具结果
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**输出：**")
            popover.write(sub_msg.content)
            continue

        # 处理 transfer_back_to 工具调用——这些表示子 agent 正在返回控制权
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # 处理 transfer_back_to 工具调用
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # 读取对应的工具结果
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # 处理完 transfer back 后，该 agent 的工作就结束了
            if status:
                status.update(state="complete")
            break

        # 在同一嵌套状态中展示内容和工具调用
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # 检查这是否是嵌套的 transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # 为子 agent 创建嵌套状态容器
                        nested_status = status.status(
                            f"""💼 子 Agent：{tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # 递归处理该子 agent 的子 agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # 常规工具调用默认隐藏，除非侧边栏选择显示。对应的
                        # 工具结果仍会被上方的循环读取，只是不绘制，
                        # 因此跳过 popover 可保持流式对齐。
                        if not tool_calls_visible():
                            continue
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**工具：**{tc['name']}")
                        popover.write("**输入：**")
                        popover.write(tc["args"])
                        # 使用工具调用 ID 存储 popover 引用
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
