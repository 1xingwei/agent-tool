import asyncio
import json
import os
import queue
from collections.abc import AsyncGenerator, Coroutine, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast, final

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """在同步方法里跑一个一次性协程。

    streamlit_app.py 里调用本类的同步方法时位于异步 `main()` 内。streamlit 的
    `@st.cache_data` 在当前线程直接执行被缓存函数（cache_utils.py:385），没有线程池
    卸载，因此此时裸用 `asyncio.run` 会抛「cannot be called from a running event
    loop」；检测到运行中的 loop 就退到工作线程。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _iterate_sync[T](agen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    """同步迭代异步生成器。

    必须在同一个 loop 上逐个 `__anext__`：每次新建 loop 会让已绑定的 httpx
    连接池落到已关闭的 loop 上。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        yield from _drive_iter(agen)
        return
    # 已有运行中的 loop（如 streamlit 的 async main）：整个迭代退到工作线程，
    # 否则 `loop.run_until_complete` 会抛「Cannot run the event loop while
    # another loop is running」。跨线程用 queue 桥接逐项转发。
    out: queue.Queue = queue.Queue()
    sentinel = object()

    def _worker() -> None:
        try:
            for item in _drive_iter(agen):
                out.put(item)
        except BaseException as e:
            out.put(e)
        finally:
            out.put(sentinel)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_worker)
        while True:
            item = out.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def _drive_iter[T](agen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    """在专用 loop 上逐个驱动 `__anext__`；`finally` 里 `aclose` 是为了调用方提前 break 时也能关闭响应流。"""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.run_until_complete(agen.aclose())
        loop.close()


class AgentClientError(Exception):
    pass


@final
class _UnknownEvent:
    pass


# 哨兵：未知事件类型不产出任何值，但也不能被当成 [DONE]（流结束）。
_UNKNOWN_EVENT = _UnknownEvent()


class AgentClient:
    """用于与 agent 服务交互的客户端。"""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        初始化客户端。

        Args:
            base_url (str): agent 服务的基础 URL。
            agent (str): 要使用的默认 agent 名称。
            timeout (float, optional): 请求超时时间。
            get_info (bool, optional): 初始化时是否获取 agent 信息。
                默认：True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        异步调用 agent。仅返回最终消息。

        Args:
            message (str): 要发送给 agent 的消息
            model (str, optional): 用于 agent 的 LLM 模型
            thread_id (str, optional): 用于继续对话的 thread ID
            user_id (str, optional): 用于跨多个 thread 继续对话的 user ID
            agent_config (dict[str, Any], optional): 传递给 agent 的额外配置

        Returns:
            AnyMessage: agent 的响应
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """同步调用 agent。仅返回最终消息。"""
        return _run_sync(
            self.ainvoke(
                message,
                model=model,
                thread_id=thread_id,
                user_id=user_id,
                agent_config=agent_config,
            )
        )

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None | _UnknownEvent:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # 将 JSON 格式的消息转换为 AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # 直接产出 str token
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
                case _:
                    # 未知类型（如未来新增的 heartbeat）：跳过，而不是当成流结束
                    return _UNKNOWN_EVENT
        return _UNKNOWN_EVENT

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """同步流式输出 agent 的响应。"""
        return _iterate_sync(
            self.astream(
                message,
                model=model,
                thread_id=thread_id,
                user_id=user_id,
                agent_config=agent_config,
                stream_tokens=stream_tokens,
            )
        )

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        以异步方式流式返回 agent 的响应。

        agent 处理过程中的每条中间消息都会以 AnyMessage 形式产出。
        如果 stream_tokens 为 True（默认值），响应还会产出
        流式模型生成的内容 token。

        Args:
            message (str): 发送给 agent 的消息
            model (str, optional): agent 使用的 LLM 模型
            thread_id (str, optional): 用于继续对话的 thread ID
            user_id (str, optional): 用于跨多个 thread 继续对话的 user ID
            agent_config (dict[str, Any], optional): 传递给 agent 的额外配置
            stream_tokens (bool, optional): 在 token 生成时流式返回
                默认：True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: agent 的响应
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    lines = response.aiter_lines()
                    try:
                        async for line in lines:
                            if line.strip():
                                parsed = self._parse_stream_line(line)
                                if parsed is None:
                                    break
                                if parsed is _UNKNOWN_EVENT:
                                    continue
                                # 不要产出空字符串 token，它们会导致生成器出现问题
                                if parsed != "":
                                    yield cast(ChatMessage | str, parsed)
                    finally:
                        await lines.aclose()  # type: ignore[missing-attribute]
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        为一次运行创建反馈记录。

        这是 LangSmith create_feedback API 的简单封装，因此
        凭据可以存储在服务端并由服务端管理，而不是客户端。
        参见：https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str, agent: str | None = None) -> ChatHistory:
        """
        获取聊天历史。

        Args:
            thread_id (str, optional): 用于标识对话的 thread ID
            agent (str, optional): 其图应解释该 thread 的 agent。
        """
        agent = agent or self.agent
        request = ChatHistoryInput(thread_id=thread_id)
        url = f"{self.base_url}/{agent}/history" if agent else f"{self.base_url}/history"
        try:
            response = httpx.post(
                url,
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    def _user_threads_request(
        self, user_id: str, agent: str | None, limit: int
    ) -> tuple[str, dict[str, Any]]:
        agent_id = agent or self.agent
        url = f"{self.base_url}/{agent_id}/threads" if agent_id else f"{self.base_url}/threads"
        return url, UserThreadsInput(user_id=user_id, limit=limit).model_dump()

    def get_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """列出用户的对话 thread。"""
        return _run_sync(self.aget_user_threads(user_id, agent=agent, limit=limit))

    async def aget_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """
        以异步方式列出用户的对话 thread。

        Args:
            user_id (str): 要列出 thread 的 user ID。
            agent (str, optional): 要列出其 thread 的 agent。
            limit (int, optional): 返回的 thread 最大数量。
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())
