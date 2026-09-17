"""针对真实 SQLite checkpointer 的 /threads 集成测试。

test_service.py 中的单元测试使用 fake checkpointer，因此无法发现数据库
拒绝的 metadata 过滤条件，或 LangGraph 实际并不成立的 head checkpoint 假设。
这些测试改为通过真实 checkpointer 运行真实图。
"""

from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.func import entrypoint
from langgraph.graph import END, MessagesState, StateGraph

from service import app


async def echo(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}


def build_graph_agent(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("echo", echo)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)


def build_subgraph_agent(checkpointer):
    """调用带 checkpoint 的子图的图，与 supervisor agent 的做法类似。"""
    inner = StateGraph(MessagesState)
    inner.add_node("echo", echo)
    inner.set_entry_point("echo")
    inner.add_edge("echo", END)

    outer = StateGraph(MessagesState)
    outer.add_node("worker", inner.compile())
    outer.set_entry_point("worker")
    outer.add_edge("worker", END)
    return outer.compile(checkpointer=checkpointer)


def build_functional_agent(checkpointer):
    @entrypoint(checkpointer=checkpointer)
    async def functional(inputs: dict, *, previous: dict | None = None) -> dict:
        messages = inputs["messages"]
        if previous:
            messages = previous["messages"] + messages
        response = AIMessage(content=f"echo: {messages[-1].content}")
        return entrypoint.final(
            value={"messages": [response]}, save={"messages": messages + [response]}
        )

    return functional


async def run_turns(agent, thread_id: str, user_id: str, agent_id: str, messages: list[str]):
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        metadata={"user_id": user_id, "agent_id": agent_id},
    )
    for message in messages:
        await agent.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)


@pytest_asyncio.fixture
async def seeded(tmp_path):
    """预置两个 agent × 两个用户，包含单轮和多轮 thread。"""
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        agents = {
            "graph-agent": build_graph_agent(checkpointer),
            "functional-agent": build_functional_agent(checkpointer),
            "subgraph-agent": build_subgraph_agent(checkpointer),
        }
        threads = {
            ("graph-agent", "alice", "g-alice-single"): ["only turn"],
            ("graph-agent", "alice", "g-alice-multi"): ["first turn", "second", "third"],
            ("graph-agent", "bob", "g-bob-single"): ["bob only turn"],
            ("functional-agent", "alice", "f-alice-single"): ["fn only turn"],
            ("functional-agent", "alice", "f-alice-multi"): ["fn first turn", "fn second"],
            ("subgraph-agent", "alice", "s-alice-single"): ["sub only turn"],
            ("subgraph-agent", "alice", "s-alice-multi"): ["sub first turn", "sub second"],
        }
        for (agent_id, user_id, thread_id), messages in threads.items():
            await run_turns(agents[agent_id], thread_id, user_id, agent_id, messages)

        transport = httpx.ASGITransport(app=app)
        lookup = {"side_effect": lambda agent_id: agents[agent_id]}
        with (
            patch("service.service.get_agent", **lookup),
            patch("service.agui.get_agent", **lookup),
        ):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client


@pytest.mark.asyncio
async def test_threads_lists_single_and_multi_turn_threads(seeded) -> None:
    """单轮 thread 永远不会越过其 head checkpoint——它们仍必须被列出。"""
    response = await seeded.get("/graph-agent/threads", params={"user_id": "alice"})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert [t["thread_id"] for t in threads] == ["g-alice-multi", "g-alice-single"]
    assert [t["title"] for t in threads] == ["first turn", "only turn"]
    assert all(t["updated_at"] for t in threads)


@pytest.mark.asyncio
async def test_threads_isolates_users_and_agents(seeded) -> None:
    async def thread_ids(agent_id: str, user_id: str) -> list[str]:
        response = await seeded.get(f"/{agent_id}/threads", params={"user_id": user_id})
        assert response.status_code == 200
        return sorted(t["thread_id"] for t in response.json()["threads"])

    assert await thread_ids("graph-agent", "alice") == ["g-alice-multi", "g-alice-single"]
    assert await thread_ids("graph-agent", "bob") == ["g-bob-single"]
    assert await thread_ids("functional-agent", "alice") == ["f-alice-multi", "f-alice-single"]
    assert await thread_ids("functional-agent", "bob") == []
    assert await thread_ids("graph-agent", "nobody") == []


@pytest.mark.asyncio
async def test_threads_titles_functional_api_agent(seeded) -> None:
    """Functional-API agent 将消息保存在 `__previous__` 中，而非 `messages` 通道。"""
    response = await seeded.get("/functional-agent/threads", params={"user_id": "alice"})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert [t["thread_id"] for t in threads] == ["f-alice-multi", "f-alice-single"]
    assert [t["title"] for t in threads] == ["fn first turn", "fn only turn"]


@pytest.mark.asyncio
async def test_threads_lists_subgraph_threads_once(seeded) -> None:
    """子图运行会在嵌套命名空间下写入自己的 head checkpoint。

    这些 checkpoint 继承父运行的 user_id/agent_id metadata，因此会匹配同一
    查询，否则会被列为该 thread 的额外副本，每个副本都要单独查找 tip。
    """
    tip_lookups: list[str] = []
    original = AsyncSqliteSaver.aget_tuple

    async def counting_aget_tuple(self, config):
        tip_lookups.append(config["configurable"]["thread_id"])
        return await original(self, config)

    with patch.object(AsyncSqliteSaver, "aget_tuple", counting_aget_tuple):
        response = await seeded.get("/subgraph-agent/threads", params={"user_id": "alice"})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert [t["thread_id"] for t in threads] == ["s-alice-multi", "s-alice-single"]
    assert [t["title"] for t in threads] == ["sub first turn", "sub only turn"]
    assert sorted(tip_lookups) == ["s-alice-multi", "s-alice-single"]


@pytest.mark.asyncio
async def test_threads_orders_by_most_recent_update(seeded) -> None:
    """对最旧 thread 的回复应将其移到列表顶部。"""
    before = await seeded.get("/graph-agent/threads", params={"user_id": "alice"})
    assert before.json()["threads"][-1]["thread_id"] == "g-alice-single"

    response = await seeded.post(
        "/graph-agent/invoke",
        json={"message": "a later reply", "thread_id": "g-alice-single", "user_id": "alice"},
    )
    assert response.status_code == 200

    after = await seeded.get("/graph-agent/threads", params={"user_id": "alice"})
    assert [t["thread_id"] for t in after.json()["threads"]] == [
        "g-alice-single",
        "g-alice-multi",
    ]


@pytest.mark.asyncio
async def test_threads_respects_limit(seeded) -> None:
    response = await seeded.get("/graph-agent/threads", params={"user_id": "alice", "limit": 1})

    assert response.status_code == 200
    assert [t["thread_id"] for t in response.json()["threads"]] == ["g-alice-multi"]


@pytest.mark.asyncio
async def test_agui_runs_are_listed_by_threads(seeded) -> None:
    """AG-UI 运行记录相同的 metadata，因此其 thread 与其他 thread 一样被列出。"""
    response = await seeded.post(
        "/agui/graph-agent/run",
        json={
            "threadId": "agui-thread",
            "runId": "agui-run",
            "messages": [{"id": "m1", "role": "user", "content": "from ag-ui"}],
            "tools": [],
            "context": [],
            "state": {},
            "forwardedProps": {"configurable": {"user_id": "alice"}},
        },
    )
    assert response.status_code == 200

    threads = (await seeded.get("/graph-agent/threads", params={"user_id": "alice"})).json()
    listed = {t["thread_id"]: t for t in threads["threads"]}
    assert "agui-thread" in listed
    assert listed["agui-thread"]["title"] == "from ag-ui"
    assert listed["agui-thread"]["agent_id"] == "graph-agent"


@pytest.mark.asyncio
async def test_history_returns_full_conversation(seeded) -> None:
    for agent_id, thread_id, first, turns in [
        ("graph-agent", "g-alice-multi", "first turn", 3),
        ("functional-agent", "f-alice-multi", "fn first turn", 2),
    ]:
        response = await seeded.post(f"/{agent_id}/history", json={"thread_id": thread_id})
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert [m["type"] for m in messages] == ["human", "ai"] * turns
        assert messages[0]["content"] == first
