"""用于 /threads 端点的 thread 枚举。

thread 派生自 checkpointer 而非它们自己的表，因此列出
它们意味着按 LangGraph 写入的方式来读取 checkpoint 元数据。由此
可能实现的不变量记录在下面的常量中。
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from schema import ThreadSummary
from service.utils import convert_message_content_to_string, messages_from_checkpoint

logger = logging.getLogger(__name__)

# LangGraph 每个 thread 在 step -1 写入一次输入 checkpoint；后续轮次从
# 最后一个 step 继续。不要换成其他 step——单轮 thread 永远不会到达 step 1。
THREAD_HEAD_STEP = -1

# head 按 thread 创建顺序排列，因此多取一些再按 tip 重新排序，以近似
# 「最近更新」。创建时间早于所取最旧 head 的 thread 会掉出。
MAX_THREAD_HEADS = 200
HEAD_PAGE_SIZE = 200

# 一个 head 行并不总是对应一个不同的 thread：带子图的 agent 每次子图调用
# 都会写入一个，继承父级的元数据。这限定了为补偿它而做的分页。
MAX_HEAD_ROWS = 1000

TITLE_MAX_LENGTH = 60


async def _list_thread_heads(checkpointer: Any, user_id: str, agent_id: str) -> list[Any]:
    """每个 thread 返回一个 head checkpoint，最新的 thread 在前。"""
    heads: list[Any] = []
    seen: set[str] = set()
    rows_scanned = 0
    before = None
    while len(seen) < MAX_THREAD_HEADS and rows_scanned < MAX_HEAD_ROWS:
        page = [
            c
            async for c in checkpointer.alist(
                None,
                filter={"user_id": user_id, "agent_id": agent_id, "step": THREAD_HEAD_STEP},
                before=before,
                limit=HEAD_PAGE_SIZE,
            )
        ]
        if not page:
            break
        rows_scanned += len(page)
        short_page = len(page) < HEAD_PAGE_SIZE
        for row in page:
            thread_id = row.config["configurable"]["thread_id"]
            if thread_id in seen:
                continue
            seen.add(thread_id)
            heads.append(row)
        if short_page:
            break
        before = RunnableConfig(
            configurable={"checkpoint_id": page[-1].config["configurable"]["checkpoint_id"]}
        )
    return heads


async def list_user_threads(
    checkpointer: Any, user_id: str, agent_id: str, limit: int
) -> list[ThreadSummary]:
    """列出某个 agent 下用户的 thread，按最近更新排序。"""
    summaries: list[tuple[str, ThreadSummary]] = []
    for head in await _list_thread_heads(checkpointer, user_id, agent_id):
        thread_id = head.config["configurable"]["thread_id"]
        stored_user_id = head.metadata.get("user_id")
        stored_agent_id = head.metadata.get("agent_id")
        if stored_user_id != user_id or stored_agent_id != agent_id:
            logger.warning(
                f"Checkpointer returned thread {thread_id} with user_id "
                f"{stored_user_id!r}/agent_id {stored_agent_id!r}, expected "
                f"{user_id!r}/{agent_id!r} — skipping to avoid a "
                "cross-user or cross-agent leak."
            )
            continue

        # head 还没有消息，因此标题和 updated_at 来自 tip。
        tip = await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))
        if tip is None:
            continue
        messages = messages_from_checkpoint(tip.checkpoint)
        first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        summaries.append(
            (
                tip.config["configurable"]["checkpoint_id"],
                ThreadSummary(
                    thread_id=thread_id,
                    agent_id=agent_id,
                    updated_at=tip.checkpoint.get("ts"),
                    title=convert_message_content_to_string(first_human.content)[:TITLE_MAX_LENGTH]
                    if first_human
                    else None,
                ),
            )
        )

    # checkpoint ID 是按时间排序的 UUID，因此 tip 的 ID 按最后更新时间排序。
    summaries.sort(key=lambda item: item[0], reverse=True)
    return [summary for _, summary in summaries[:limit]]
