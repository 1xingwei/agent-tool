import inspect
import json
import logging
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from core import settings
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)
from service.agui import router as agui_router
from service.threads import list_user_threads
from service.utils import (
    convert_message_content_to_string,
    ensure_model_available,
    langchain_to_chat_message,
    messages_from_checkpoint,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def custom_generate_unique_id(route: APIRoute) -> str:
    """为 OpenAPI 客户端生成惯用的 operation ID。"""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def _resolve_agent(agent_id: str) -> AgentGraph:
    """解析 agent id，将未知 id 映射为 404 而非裸 500。

    `get_agent` 对未注册的 id 会抛出 KeyError。若不处理，它会逃出端点的 `try`
    变成不透明的 500——在 `/stream` 上更糟，因为响应生成器体执行时
    200 状态已提交，客户端会看到 200 加空响应体。

    特意定义在此处而非 `service.utils`：`get_agent` 从本模块的全局变量读取，
    这是所有 service 测试打补丁的接缝（`patch("service.service.get_agent", ...)`）。
    放在 `service.utils` 会悄悄改变所有这些测试的接缝目标。

    只映射 KeyError：RuntimeError 表示惰性加载的 agent 从未 `load()`，
    这是服务器状态 bug，应继续以 500 暴露，而非伪装成 404。
    """
    try:
        return get_agent(agent_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} not found"
        ) from None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    可配置的 lifespan，用于初始化相应的数据库 checkpointer、store
    以及支持异步加载的 agent——例如启动 MCP 客户端。
    """
    try:
        # 初始化 checkpointer（用于短期记忆）和 store（用于长期记忆）
        async with initialize_database() as saver, initialize_store() as store:
            # 设置两个组件
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Store 可以是 SQLite 或 Postgres，两者都需要设置
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            if not settings.AUTH_SECRET:
                logger.warning(
                    "AUTH_SECRET is not configured — all API endpoints are unauthenticated. "
                    "Set AUTH_SECRET in your environment to enable bearer token authentication."
                )

            # 为 agent 配置两个记忆组件和异步加载
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                    agent = get_agent(a.key)
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # 继续处理其他 agent，而非让启动失败
                    continue
                # 设置 checkpointer 用于 thread 作用域记忆（对话历史）
                agent.checkpointer = saver
                # 设置 store 用于长期记忆（跨对话知识）
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
router = APIRouter(dependencies=[Depends(verify_bearer)])
# AG-UI 协议端点继承相同的 bearer 认证——见 service/agui.py
router.include_router(agui_router)


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(
    user_input: UserInput, agent: AgentGraph, agent_id: str
) -> tuple[dict[str, Any], UUID]:
    """
    解析用户输入并处理所需的 interrupt 恢复。
    返回 agent 调用的 kwargs 和 run_id。
    """
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        ensure_model_available(user_input.model)
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # 为 Langchain 初始化 Langfuse CallbackHandler（追踪）
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # 检查保留键（包括 'model'，即使不在 configurable 中）
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        metadata={"user_id": user_id, "agent_id": agent_id},
        run_id=run_id,
        callbacks=callbacks,
    )

    # 检查需要恢复的 interrupt
    state = await agent.aget_state(config=config)

    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # 假设用户输入是对从 interrupt 恢复 agent 执行的响应
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    用用户输入调用 agent 以获取最终响应。

    若未提供 agent_id，将使用默认 agent。
    使用 thread_id 持久化并继续多轮对话。run_id kwarg
    也会附加到消息上以记录反馈。
    使用 user_id 跨多个 thread 持久化并继续对话。
    """
    # NOTE: 目前这仅返回最后一条消息或 interrupt。
    # 当 agent 输出多条 AIMessage 时（例如 interrupt-agent 中的后台步骤，
    # 或 research-assistant 中的工具步骤），会被省略。可以说，
    # 你可能希望包含它。那种情况下，你可以更新 API 返回 ChatMessage 列表。
    agent: AgentGraph = _resolve_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent, agent_id)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        # 因 interrupt 而停止的运行会在任一流模式的最终事件上报告它，
        # 因此在回退到最后一条消息之前先检查 interrupt。
        if "__interrupt__" in response:
            # 将第一个 interrupt 的值作为 AIMessage 返回
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        elif response_type == "values":
            # 正常响应，agent 成功完成
            output = langchain_to_chat_message(response["messages"][-1])
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    从 agent 生成消息流。

    这是 /stream 端点的主力方法。
    """
    agent: AgentGraph = _resolve_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent, agent_id)

    try:
        # 处理来自图的流式事件，并通过 SSE 流产出消息。
        async for stream_event in agent.astream(  # type: ignore[no-matching-overload]
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # 根据子图处理不同的流事件结构
            if len(stream_event) == 3:
                # 使用 subgraphs=True 时：(node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # 不使用 subgraphs 时：(stream_mode, event)
                stream_mode, event = stream_event
            new_messages: list[Any] = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # 处理 agent interrupt 的简单方法。
                    # 在更复杂的实现中，我们可以添加
                    # 某种结构化的 ChatMessage 类型来返回 interrupt 值。
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # 使用 langgraph-supervisor 库的特殊情况
                    if "supervisor" in node or "sub-agent" in node:
                        # 来自实际 agent 的唯一工具是 handoff 和 handback 工具
                        if update_messages and isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # 若这是子 agent，我们希望保留最后 2 条消息——handback 工具及其结果
                                update_messages = update_messages[-2:]
                            else:
                                # 若这是 supervisor，我们希望只保留最后一条消息——handoff 结果。该工具来自 'agent' 节点。
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph 流式输出可能发出元组：(field_name, field_value)
            # 例如 ('content', <str>)、('tool_calls', [ToolCall,...])、('additional_kwargs', {...}) 等。
            # 我们只将支持的字段累积到 `parts` 中，跳过不支持的元数据。
            # 更多信息见：https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # 将 parts 存入临时 dict
                    current_message[key] = value
                else:
                    # 如果有正在进行的完整消息，则添加它
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # 添加任何剩余的消息片段
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph 会重新发送输入消息，这显得很奇怪，因此丢弃它
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # 由于某些原因，astream("messages") 会导致非 LLM 节点发送额外消息。
                # 丢弃它们。
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # 在 OpenAI 的上下文中，空内容通常意味着
                    # 模型正在请求调用某个工具。
                    # 因此我们只打印非空内容。
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    将 agent 的响应流式返回给用户输入，包括中间消息和 token。

    如果未提供 agent_id，将使用默认 agent。
    使用 thread_id 来持久化并继续多轮对话。run_id kwarg
    也会附加到所有消息上，用于记录反馈。
    使用 user_id 来跨多个 thread 持久化并继续对话。

    设置 `stream_tokens=false` 可返回中间消息但不逐 token 返回。
    """
    # 在构造响应之前先解析：一旦 StreamingResponse 已提交
    # 200 状态，未知的 agent id 就无法再转为 404，客户端
    # 将收到一个空响应体，与成功的空流无法区分。
    _resolve_agent(agent_id)
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    为某次 run 向 LangSmith 记录反馈。

    这是 LangSmith create_feedback API 的简单封装，因此
    凭据可以存储在服务中并由服务管理，而不是客户端。
    参见：https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    kwargs = feedback.kwargs or {}
    # 反馈是可选的 LangSmith 集成；没有 key 时执行空操作，而不是返回 500。
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.info("LANGSMITH_API_KEY not set; skipping feedback recording")
        return FeedbackResponse()
    client = LangsmithClient()
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/{agent_id}/history", operation_id="history_with_agent_id")
@router.post("/history")
async def history(input: ChatHistoryInput, agent_id: str = DEFAULT_AGENT) -> ChatHistory:
    """
    获取某个 thread 和 agent 的聊天历史。

    如果未提供 agent_id，将使用默认 agent。
    """
    agent: AgentGraph = _resolve_agent(agent_id)
    config = RunnableConfig(configurable={"thread_id": input.thread_id})
    try:
        messages: list[BaseMessage] = []
        # Functional-API agent 将对话保存在 `__previous__` 中，而 aget_state
        # 不会返回它，因此先读取原始 checkpoint，仅对图回退。
        checkpointer = getattr(agent, "checkpointer", None)
        if checkpointer:
            tup = await checkpointer.aget_tuple(config)
            if tup and "__previous__" in (tup.checkpoint.get("channel_values") or {}):
                messages = messages_from_checkpoint(tup.checkpoint)
        if not messages:
            state_snapshot = await agent.aget_state(config=config)
            # 从未写入过的 thread 其 `values == {}`，因此索引访问
            # 会抛出 KeyError，下面的处理器会将其转为 500。
            messages = state_snapshot.values.get("messages", [])
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.get("/{agent_id}/threads", operation_id="threads_with_agent_id")
@router.get("/threads")
async def threads(
    input: UserThreadsInput = Depends(), agent_id: str = DEFAULT_AGENT
) -> UserThreads:
    """
    列出某个 agent 下用户的对话 thread，按最近更新排序。

    `user_id` 由调用方断言，不会与请求上的凭据进行校验，
    因此任何持有 bearer token 的人都能列出任意用户的 thread——与
    /history 相同的信任模型。在最终用户能够访问之前，请自行在前面加上授权。
    """
    agent: AgentGraph = _resolve_agent(agent_id)
    checkpointer = getattr(agent, "checkpointer", None)
    if not checkpointer:
        return UserThreads(threads=[])

    try:
        summaries = await list_user_threads(checkpointer, input.user_id, agent_id, input.limit)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

    return UserThreads(threads=summaries)


@app.get("/health")
async def health_check():
    """健康检查端点。"""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
