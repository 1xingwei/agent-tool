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
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def tool_calls_visible() -> bool:
    """Whether tool calls should be rendered in the chat.

    Tool calls are an implementation detail: the user asked a question and wants the
    answer, not the machinery behind it. So they stay hidden by default and are only
    drawn when the sidebar toggle opts in (useful for debugging and demos).

    Sub-agent transfers follow the same switch. They arrive as tool calls too, and a
    raw name like ``transfer_to_research_expert`` is no more meaningful to a user than
    any other internal function name.
    """
    return bool(st.session_state.get("show_tool_calls", False))


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


@st.cache_data(ttl=600, show_spinner=False)
def fetch_user_threads_cached(
    base_url: str, user_id: str, agent_id: str | None = None, limit: int = 20
) -> UserThreads:
    """
    Fetch and cache user threads using the new synchronous get_user_threads method.
    """
    client = AgentClient(base_url=base_url, get_info=False)
    return client.get_user_threads(user_id=user_id, agent=agent_id, limit=limit)


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
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

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
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

    # Initialize voice manager (once per session)
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            # Read the agent from the URL so history is fetched through the graph that
            # created the thread.
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

    # Keep thread_id in the URL so the address bar is directly shareable.
    st.query_params["thread_id"] = st.session_state.thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "基于 LangGraph、FastAPI、Streamlit 构建的 AI Agent 服务全功能工具包"
        ""

        if st.button(":material/chat: 新对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # Clear saved audio when starting new chat
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
            # Sync the selection to the ?agent= URL param (dropped when it's the default).
            agent_client.agent = st.selectbox(
                "使用的 Agent",
                options=agent_list,
                index=agent_idx,
                key="agent",
                bind="query-params",
                on_change=fetch_user_threads_cached.clear,
            )
            use_streaming = st.toggle("流式输出", value=True)
            # Audio toggle with callback: clears cached audio when toggled off
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
            # Tool calls are implementation detail, so they stay hidden unless the
            # user opts in. See tool_calls_visible() and draw_messages().
            st.toggle(
                "显示工具调用",
                value=False,
                key="show_tool_calls",
                help="仅供调试与演示：开启后展示每次工具调用、子 Agent 转交的输入与输出；"
                "关闭时只显示最终回复",
            )

            # Display user ID (for debugging or user information)
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
            # st.context.url is the browser URL (with query string stripped). Rebuild
            # the params, including the agent so the thread resumes through the right graph.
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

    # Draw existing messages
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

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Render saved audio for the last AI message (if it exists)
    # This ensures audio persists across st.rerun() calls
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

    # Generate new message if the user provided new input
    # Use voice manager if available, otherwise fall back to regular input
    # REQUIRED: Set VOICE_STT_PROVIDER, VOICE_TTS_PROVIDER, OPENAI_API_KEY
    # in app .env (NOT service .env) to enable voice features.
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
                # Generate TTS audio for streaming response
                # Note: draw_messages() stores the final message in st.session_state.messages
                # and the container reference in st.session_state.last_message
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # Only generate audio for AI responses with content
                    if last_msg.type == "ai" and last_msg.content:
                        # Use audio_only=True since text was already streamed by draw_messages()
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
                # Render AI response with optional voice
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            if is_first_message:
                fetch_user_threads_cached.clear()
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"生成回复时出错：{e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls, but only when the user opted into
      seeing them (see tool_calls_visible()). Track the tool inputs and outputs and
      update the status container accordingly. Calls that stay hidden are still read
      off the stream so the messages keep their alignment, and a transient hint
      stands in for them while they run.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
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
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Tool calls and sub-agent transfers are both implementation
                        # detail, so they share one switch and are hidden by default.
                        # With the details hidden the calls still have to be consumed
                        # below, otherwise the stream would fall out of alignment.
                        details = tool_calls_visible()
                        transfers = [tc for tc in msg.tool_calls if "transfer_to" in tc["name"]]
                        regular_calls = [
                            tc for tc in msg.tool_calls if "transfer_to" not in tc["name"]
                        ]
                        status_state = "running" if is_new else "complete"

                        # Track the status container for each tool call by ID so results
                        # are mapped back to the correct container.
                        call_results = {}
                        if details:
                            for tool_call in transfers:
                                call_results[tool_call["id"]] = st.status(
                                    f"""💼 子 Agent：{tool_call["name"]}""",
                                    state=status_state,
                                )

                            if regular_calls:
                                # One container per message rather than one per call, so
                                # a turn with several lookups still reads as a single row.
                                if len(regular_calls) == 1:
                                    label = f"""🛠️ 工具调用：{regular_calls[0]["name"]}"""
                                else:
                                    label = f"🛠️ 工具调用（{len(regular_calls)}）"
                                aggregate = st.status(label, state=status_state, type="compact")
                                for tool_call in regular_calls:
                                    call_results[tool_call["id"]] = aggregate

                        # With the details hidden there is nothing to watch while the
                        # tools run, so show a transient hint instead of leaving the
                        # message looking stalled. It is cleared once they return.
                        running_hint = st.empty() if is_new and not details else None
                        if running_hint:
                            running_hint.caption(":material/progress_activity: 正在处理…")

                        # Expect one ToolMessage for each tool call.
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results.get(tool_call["id"])
                                if status is not None:
                                    status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # Only non-transfer tool calls reach this point. The result
                            # must be consumed even when the call is not displayed.
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

                            # Record the message if it's new, and update the correct
                            # status container with the result
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
                # CustomData example used by the bg-task-agent
                # See:
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

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"意外的消息类型：{msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
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
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    When tool calls are hidden the caller passes ``status=None``: the sub-agent
    messages are still consumed so the stream keeps its alignment, they are just not
    drawn.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent, or None when tool calls
            are hidden
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back
    while True:
        # Read next message
        sub_msg = await anext(messages_agen)

        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**输出：**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 子 Agent：{tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool calls are hidden unless the sidebar opts in. The
                        # matching tool result is still read by the loop above, it just
                        # is not drawn, so skipping the popover keeps the stream aligned.
                        if not tool_calls_visible():
                            continue
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**工具：**{tc['name']}")
                        popover.write("**输入：**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
