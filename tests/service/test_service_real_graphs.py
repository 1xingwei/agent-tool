"""针对真实编译图的 /invoke 和 /stream 集成测试。

test_service.py 中的单元测试驱动的是 AsyncMock agent，因此它们断言的每个元组形状和
事件顺序都是手工构造的，而非由 LangGraph 产生。这些测试在真实 checkpointer 上对真实图
运行相同的端点，因此 LangGraph 在报告中断、排序流事件或暴露待处理任务方面的变化都会在此显现。
"""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter, interrupt

from agents.utils import CustomData
from service import app


async def greet(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content="let me check")]}


async def ask_color(state: MessagesState) -> MessagesState:
    answer = interrupt("What is your favorite color?")
    return {"messages": [AIMessage(content=f"Your favorite color is {answer}")]}


def build_interrupt_agent(checkpointer):
    """在中断前先发出一条消息，使中断在流中途到达。"""
    graph = StateGraph(MessagesState)
    graph.add_node("greet", greet)
    graph.add_node("ask", ask_color)
    graph.set_entry_point("greet")
    graph.add_edge("greet", "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=checkpointer)


async def count_messages(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content=f"heard {len(state['messages'])} messages")]}


def build_counting_agent(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("count", count_messages)
    graph.set_entry_point("count")
    graph.add_edge("count", END)
    return graph.compile(checkpointer=checkpointer)


async def working(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content="working on it")]}


async def finished(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content="all done")]}


def build_two_step_agent(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("working", working)
    graph.add_node("finished", finished)
    graph.set_entry_point("working")
    graph.add_edge("working", "finished")
    graph.add_edge("finished", END)
    return graph.compile(checkpointer=checkpointer)


async def report_progress(state: MessagesState, writer: StreamWriter) -> MessagesState:
    CustomData(data={"status": "running"}).dispatch(writer)
    return {"messages": [AIMessage(content="all done")]}


def build_custom_data_agent(checkpointer):
    """bg-task-agent 的形态：一个节点在写入消息的同时写入自定义数据。"""
    graph = StateGraph(MessagesState)
    graph.add_node("report", report_progress)
    graph.set_entry_point("report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def checkpointer(request, tmp_path):
    """待处理的中断和累积的消息都会经由 checkpointer 往返，因此依赖它们的测试
    同样针对真实数据库运行。"""
    if request.param == "memory":
        yield MemorySaver()
    else:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
            yield saver


@asynccontextmanager
async def client_for(agents: dict[str, Any]) -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = httpx.ASGITransport(app=app)
    with patch("service.service.get_agent", side_effect=lambda agent_id: agents[agent_id]):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def stream_events(client: httpx.AsyncClient, path: str, body: dict) -> list[dict[str, Any]]:
    events = []
    async with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def streamed_messages(events: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [
        (e["content"]["type"], e["content"]["content"]) for e in events if e["type"] == "message"
    ]


async def history_of(client: httpx.AsyncClient, path: str, thread_id: str) -> list[tuple[str, str]]:
    response = await client.post(path, json={"thread_id": thread_id})
    assert response.status_code == 200
    return [(m["type"], m["content"]) for m in response.json()["messages"]]


@pytest.mark.asyncio
async def test_invoke_resumes_an_interrupted_thread(checkpointer) -> None:
    """第二次调用必须恢复待处理的中断，而不是开启新一轮。"""
    agents = {"interrupt-graph": build_interrupt_agent(checkpointer)}
    async with client_for(agents) as client:
        first = await client.post(
            "/interrupt-graph/invoke", json={"message": "hi", "thread_id": "t"}
        )
        assert first.status_code == 200
        assert (first.json()["type"], first.json()["content"]) == (
            "ai",
            "What is your favorite color?",
        )

        second = await client.post(
            "/interrupt-graph/invoke", json={"message": "blue", "thread_id": "t"}
        )
        assert second.status_code == 200
        assert (second.json()["type"], second.json()["content"]) == (
            "ai",
            "Your favorite color is blue",
        )

        assert await history_of(client, "/interrupt-graph/history", "t") == [
            ("human", "hi"),
            ("ai", "let me check"),
            ("ai", "Your favorite color is blue"),
        ]


@pytest.mark.asyncio
async def test_stream_resumes_an_interrupted_thread(checkpointer) -> None:
    agents = {"interrupt-graph": build_interrupt_agent(checkpointer)}
    async with client_for(agents) as client:
        first = await stream_events(
            client, "/interrupt-graph/stream", {"message": "hi", "thread_id": "t"}
        )
        assert streamed_messages(first) == [
            ("ai", "let me check"),
            ("ai", "What is your favorite color?"),
        ]

        second = await stream_events(
            client, "/interrupt-graph/stream", {"message": "blue", "thread_id": "t"}
        )
        assert streamed_messages(second) == [("ai", "Your favorite color is blue")]

        assert await history_of(client, "/interrupt-graph/history", "t") == [
            ("human", "hi"),
            ("ai", "let me check"),
            ("ai", "Your favorite color is blue"),
        ]


@pytest.mark.asyncio
async def test_invoke_accumulates_state_across_turns(checkpointer) -> None:
    agents = {"counting-agent": build_counting_agent(checkpointer)}
    async with client_for(agents) as client:
        contents = []
        for message in ["first", "second"]:
            response = await client.post(
                "/counting-agent/invoke", json={"message": message, "thread_id": "t"}
            )
            assert response.status_code == 200
            contents.append(response.json()["content"])

        assert contents == ["heard 1 messages", "heard 3 messages"]
        assert await history_of(client, "/counting-agent/history", "t") == [
            ("human", "first"),
            ("ai", "heard 1 messages"),
            ("human", "second"),
            ("ai", "heard 3 messages"),
        ]


@pytest.mark.asyncio
async def test_invoke_returns_only_the_final_message() -> None:
    """固定记录已知限制：中间 AIMessage 会被 /invoke 丢弃。"""
    agents = {"two-step-agent": build_two_step_agent(MemorySaver())}
    async with client_for(agents) as client:
        response = await client.post(
            "/two-step-agent/invoke", json={"message": "hi", "thread_id": "t"}
        )
        assert response.status_code == 200
        assert (response.json()["type"], response.json()["content"]) == ("ai", "all done")

        assert await history_of(client, "/two-step-agent/history", "t") == [
            ("human", "hi"),
            ("ai", "working on it"),
            ("ai", "all done"),
        ]


@pytest.mark.asyncio
async def test_stream_forwards_custom_data() -> None:
    agents = {"custom-agent": build_custom_data_agent(MemorySaver())}
    async with client_for(agents) as client:
        events = await stream_events(
            client, "/custom-agent/stream", {"message": "hi", "thread_id": "t"}
        )

        messages = [e["content"] for e in events if e["type"] == "message"]
        assert [m["type"] for m in messages] == ["custom", "ai"]
        assert messages[0]["custom_data"] == {"status": "running"}
        assert messages[1]["content"] == "all done"
