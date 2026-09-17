"""针对 fake checkpointer 的 /threads 单元测试。

仅覆盖无法让真实 checkpointer 按需执行的情况：行数与页数上限（需要比值得预置的更多
checkpoint）、返回本应被过滤器排除的行的 checkpointer、缺失时间戳，以及查询失败。真实数据库
能够覆盖的行为——过滤、排序、标题、限制、子图 thread——改在 test_threads_sqlite.py 中针对
SQLite 测试，而非针对此 fake。
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


class FakeCheckpointTuple:
    def __init__(self, thread_id: str, checkpoint_id: str, checkpoint: dict, metadata: dict):
        self.config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        self.checkpoint = checkpoint
        self.metadata = metadata


class FakeCheckpointer:
    """具有 /threads 所依赖语义的 checkpointer。

    checkpoint 按 checkpoint_id 全局排序，`alist` 将元数据过滤器作为精确匹配应用，每个 thread 的
    首个 checkpoint 以 step -1 写入，与 LangGraph 写入输入 checkpoint 的方式一致。
    """

    def __init__(self):
        self.rows: list[FakeCheckpointTuple] = []
        self.alist_filters: list[dict | None] = []
        self._next_id = 0

    def _checkpoint_id(self) -> str:
        self._next_id += 1
        return f"cp-{self._next_id:06d}"

    def add_thread(
        self,
        thread_id: str,
        user_id: str = "user-123",
        agent_id: str = "research-assistant",
        turns: int = 1,
        title: str = "Hello",
        subgraph_heads: int = 0,
        tip_ts: str | None = "2024-07-31T20:14:19.804150+00:00",
    ) -> None:
        def add(step: int, channel_values: dict, ts: str | None) -> None:
            self.rows.append(
                FakeCheckpointTuple(
                    thread_id,
                    self._checkpoint_id(),
                    {"ts": ts, "channel_values": channel_values},
                    {"step": step, "user_id": user_id, "agent_id": agent_id},
                )
            )

        add(-1, {"__start__": {"messages": [HumanMessage(content=title)]}}, "2024-01-01T00:00:00Z")
        # 子图运行写入自己的 head，继承父运行的元数据。
        for _ in range(subgraph_heads):
            add(-1, {"__start__": {}}, "2024-01-01T00:00:00Z")

        messages: list = []
        for turn in range(turns):
            messages = messages + [
                HumanMessage(content=title if turn == 0 else f"{title} {turn}"),
                AIMessage(content="reply"),
            ]
            add(
                turn * 2,
                {"messages": messages},
                tip_ts if turn == turns - 1 else "2024-01-01T00:00:01Z",
            )

    async def alist(self, config, *, filter=None, before=None, limit=None):
        self.alist_filters.append(filter)
        rows = sorted(self.rows, key=lambda r: r.config["configurable"]["checkpoint_id"])
        rows.reverse()
        yielded = 0
        for row in rows:
            if filter and any(row.metadata.get(key) != value for key, value in filter.items()):
                continue
            if (
                before
                and row.config["configurable"]["checkpoint_id"]
                >= (before["configurable"]["checkpoint_id"])
            ):
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    async def aget_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        rows = [r for r in self.rows if r.config["configurable"]["thread_id"] == thread_id]
        if not rows:
            return None
        return max(rows, key=lambda r: r.config["configurable"]["checkpoint_id"])


def test_threads_without_checkpointer_returns_empty(test_client, mock_agent) -> None:
    """测试当 agent 未配置 checkpointer 时 /threads 返回空列表。"""
    mock_agent.checkpointer = None

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert response.json() == {"threads": []}


def test_threads_filters_by_agent_id(test_client) -> None:
    """测试 /{agent_id}/threads 将 checkpointer 查询限定到所请求的 agent。

    过滤器中的 agent_id 是跨 agent 隔离的保证，因此应针对查询本身断言，而不仅仅针对返回的
    thread。
    """
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-mine", agent_id="custom-agent", title="Mine")
    checkpointer.add_thread("thread-theirs", agent_id="other-agent", title="Theirs")

    custom_agent = AsyncMock()
    custom_agent.checkpointer = checkpointer
    default_agent = AsyncMock()
    default_agent.checkpointer = None

    agent_calls = {"default": 0, "custom": 0}

    def agent_lookup(agent_id):
        if agent_id == "custom-agent":
            agent_calls["custom"] += 1
            return custom_agent
        agent_calls["default"] += 1
        return default_agent

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.get("/custom-agent/threads", params={"user_id": "user-123"})

    assert response.status_code == 200
    payload = response.json()
    assert [thread["thread_id"] for thread in payload["threads"]] == ["thread-mine"]
    assert checkpointer.alist_filters == [
        {"user_id": "user-123", "agent_id": "custom-agent", "step": -1}
    ]
    assert agent_calls == {"custom": 1, "default": 0}


@pytest.mark.parametrize("limit", [0, -1, 101, 999999])
def test_threads_rejects_out_of_range_limit(test_client, mock_agent, limit: int) -> None:
    """测试 /threads 会限制 limit，使客户端无法请求无界扫描。"""
    mock_agent.checkpointer = FakeCheckpointer()

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": limit})

    assert response.status_code == 422


def test_threads_skips_checkpoints_with_mismatched_metadata(test_client, mock_agent) -> None:
    """测试忽略过滤条件的 checkpointer 无法泄露其他用户的 thread。"""

    class LeakyCheckpointer(FakeCheckpointer):
        async def alist(self, config, *, filter=None, before=None, limit=None):
            async for row in super().alist(config, before=before, limit=limit):
                yield row

    checkpointer = LeakyCheckpointer()
    checkpointer.add_thread("thread-mine", title="Mine")
    checkpointer.add_thread("thread-other-user", user_id="other-user", title="Other user")
    checkpointer.add_thread("thread-other-agent", agent_id="other-agent", title="Other agent")
    checkpointer.add_thread("thread-no-metadata", user_id=None, agent_id=None, title="No metadata")
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert [t["thread_id"] for t in response.json()["threads"]] == ["thread-mine"]


def test_threads_pages_past_subgraph_heads(test_client, mock_agent) -> None:
    """测试子图 head 行不会把真实 thread 挤出结果。

    带子图的 agent 每个 thread 会写入多行 head，因此一页行数覆盖的
    thread 数量远少于调用方请求的数量。
    """
    checkpointer = FakeCheckpointer()
    for index in range(30):
        checkpointer.add_thread(f"thread-{index:02d}", title=f"Thread {index}", subgraph_heads=9)
    mock_agent.checkpointer = checkpointer

    with patch("service.threads.HEAD_PAGE_SIZE", 20):
        response = test_client.get("/threads", params={"user_id": "user-123", "limit": 30})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert len(threads) == 30
    assert threads[0]["thread_id"] == "thread-29"


def test_threads_bounds_total_rows_scanned(test_client, mock_agent) -> None:
    """测试 head 分页在达到行数上限时停止，而不是扫描整张表。"""
    checkpointer = FakeCheckpointer()
    for index in range(60):
        checkpointer.add_thread(f"thread-{index:02d}", title=f"Thread {index}", subgraph_heads=9)
    mock_agent.checkpointer = checkpointer

    with (
        patch("service.threads.HEAD_PAGE_SIZE", 20),
        patch("service.threads.MAX_HEAD_ROWS", 100),
    ):
        response = test_client.get("/threads", params={"user_id": "user-123", "limit": 30})

    assert response.status_code == 200
    assert len(checkpointer.alist_filters) == 5
    assert len(response.json()["threads"]) == 10


def test_threads_tolerates_missing_timestamp(test_client, mock_agent) -> None:
    """测试 /threads 会返回 tip checkpoint 没有时间戳的 thread。"""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-no-ts", title="No timestamp", tip_ts=None)
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["threads"][0]["thread_id"] == "thread-no-ts"
    assert payload["threads"][0]["updated_at"] is None


def test_threads_checkpointer_error_returns_500(test_client, mock_agent) -> None:
    async def broken_alist(*args, **kwargs):
        raise RuntimeError("db unavailable")
        yield  # pragma: no cover

    mock_agent.checkpointer = type("Checkpointer", (), {})()
    mock_agent.checkpointer.alist = broken_alist

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})
    assert response.status_code == 500
