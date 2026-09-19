"""agent 服务的 AG-UI 协议端点。

把服务里的任意 agent 通过 AG-UI 协议（https://docs.ag-ui.com）暴露出去，
以便接入 CopilotKit 等兼容 AG-UI 的前端。LangGraph -> AG-UI 的事件转换
由官方 `ag-ui-langgraph` 包完成；本模块只负责把它接进服务的 agent 注册表、
鉴权与追踪。

用法（含如何接入客户端）见 docs/02-AG-UI协议支持.md。
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from ag_ui.core import EventType, RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph import LangGraphAgent
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler  # type: ignore[import-untyped]

from agents import DEFAULT_AGENT, AgentGraph, get_agent
from core import settings
from service.utils import ensure_model_available

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agui")

# 由协议管理（thread_id 来自 RunAgentInput）或 checkpointer 管理，
# 因此客户端不能通过 forwardedProps.configurable 覆盖它们。
RESERVED_CONFIGURABLE_KEYS = {"thread_id", "checkpoint_id", "checkpoint_ns"}


def _base_config(input_data: RunAgentInput, agent_id: str) -> RunnableConfig:
    """为一次 AG-UI 运行构建基础 RunnableConfig。

    客户端可以在 `forwardedProps.configurable` 中传入可配置值（例如 `model`、`user_id` 或自定义 agent
    配置）——这相当于原生 API 的 `model` / `user_id` / `agent_config` 字段。`thread_id` 由
    `ag-ui-langgraph` 包自身从 AG-UI 输入中获取。
    """
    forwarded: dict[str, Any] = input_data.forwarded_props or {}
    configurable = forwarded.get("configurable") or {}
    if not isinstance(configurable, dict):
        raise HTTPException(status_code=422, detail="forwardedProps.configurable must be an object")
    if overlap := RESERVED_CONFIGURABLE_KEYS & configurable.keys():
        raise HTTPException(
            status_code=422,
            detail=f"forwardedProps.configurable contains reserved keys: {overlap}",
        )

    if (model := configurable.get("model")) is not None:
        ensure_model_available(model)

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        callbacks.append(CallbackHandler())

    configurable = dict(configurable)
    user_id = configurable.setdefault("user_id", str(uuid4()))

    return RunnableConfig(
        configurable=configurable,
        # 记录在 checkpoint 元数据中，使 AG-UI thread 也能出现在 /threads 中。
        metadata={"user_id": user_id, "agent_id": agent_id},
        callbacks=callbacks,
    )


async def _event_stream(
    agent_id: str,
    graph: AgentGraph,
    input_data: RunAgentInput,
    config: RunnableConfig,
    encoder: EventEncoder,
) -> AsyncGenerator[str, None]:
    # 每个请求新建一个 LangGraphAgent：它持有每次运行的状态，且构建开销很低。
    agent = LangGraphAgent(name=agent_id, graph=graph, config=config)  # type: ignore[arg-type]
    async for event in agent.run(input_data):
        # 不要转发 RAW 透传事件。标准 AG-UI 客户端会忽略它们，
        # 而且它们会向调用方暴露服务端内部信息——包括来自 on_chat_model_start 的完整渲染提示词。
        # 仅当该端点由受信任的中间层消费、且你需要完整的事件洪流
        # （例如用于 AG-UI Event Inspector）时，才移除该过滤器。
        if event.type == EventType.RAW:
            continue
        yield encoder.encode(event)


@router.post("/run", operation_id="agui_run_default")
@router.post("/{agent_id}/run", operation_id="agui_run")
async def agui_run(
    input_data: RunAgentInput, request: Request, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """
    通过 AG-UI 协议运行 agent，以 SSE 流式发送 AG-UI 事件。

    将 AG-UI 客户端（例如 CopilotKit 的 runtime 或 HttpAgent）指向此端点。
    在多次运行间使用相同的 threadId 以延续对话——thread 会
    持久化到服务的 checkpointer 中，并与原生 API 共享。
    """
    try:
        graph: AgentGraph = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    config = _base_config(input_data, agent_id)
    encoder = EventEncoder(accept=request.headers.get("accept", ""))
    return StreamingResponse(
        _event_stream(agent_id, graph, input_data, config, encoder),
        media_type=encoder.get_content_type(),
    )
