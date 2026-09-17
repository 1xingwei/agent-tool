import os

import pytest

from client import AgentClient

# 通过环境变量与 scripts/smoke_test.sh 共享，使脚本可以验证
# 该 thread 的 checkpoint 确实落入了预期的后端。单独运行测试时
# 回退到固定 id。
THREAD_ID = os.environ.get("SMOKE_THREAD_ID", "smoke-test-persistence-thread")


@pytest.mark.docker
def test_checkpointer_persists_history():
    """确认配置的 checkpointer 能跨轮次持久化对话状态。

    与后端无关：测试服务启动时所用的 DATABASE_TYPE。scripts/smoke_test.sh
    会针对 postgres 和 mongo 分别运行此测试，然后单独验证数据确实落入了
    该后端（仅凭此测试无法区分后端，因为任何可用的 checkpointer 都会通过）。
    需要由实时数据库支撑的运行中服务（USE_FAKE_MODEL=true）。

    invoke 和 get_history 均使用默认 agent。由于 /history 是 agent 感知的，
    且 AgentClient 会将其限定到客户端选定的 agent，创建 thread 的同一个图
    也会读回它——这正是往返能够成立的原因，因为现在每个 agent 都是拥有
    自身状态的独立图。
    """
    client = AgentClient("http://localhost:8080")

    client.invoke("Tell me a joke?", thread_id=THREAD_ID, model="fake")
    client.invoke("Tell me another?", thread_id=THREAD_ID, model="fake")

    history = client.get_history(thread_id=THREAD_ID)
    human_messages = [m for m in history.messages if m.type == "human"]
    assert len(human_messages) == 2
    assert human_messages[0].content == "Tell me a joke?"
    assert human_messages[1].content == "Tell me another?"


@pytest.mark.docker
def test_threads_lists_user_threads():
    """确认 /threads 通过配置的 checkpointer 枚举 thread。

    单元测试使用 fake checkpointer，SQLite 集成测试只能证明 SQLite 驱动，
    因此这里检查 metadata 过滤条件（包括 step -1 的 head 查找）在 Postgres
    和 Mongo 上行为一致。需要由实时数据库支撑的运行中服务
    （USE_FAKE_MODEL=true）。
    """
    client = AgentClient("http://localhost:8080")
    user_id = f"{THREAD_ID}-user"
    other_user_id = f"{THREAD_ID}-other"

    single_turn = f"{THREAD_ID}-single"
    multi_turn = f"{THREAD_ID}-multi"
    client.invoke("Only turn", thread_id=single_turn, user_id=user_id, model="fake")
    client.invoke("First turn", thread_id=multi_turn, user_id=user_id, model="fake")
    client.invoke("Second turn", thread_id=multi_turn, user_id=user_id, model="fake")
    client.invoke("Not mine", thread_id=f"{THREAD_ID}-other", user_id=other_user_id, model="fake")

    threads = client.get_user_threads(user_id=user_id).threads
    # 最近更新的排在最前；单轮 thread 即使从未越过其 head checkpoint，
    # 也必须被列出。
    assert [t.thread_id for t in threads] == [multi_turn, single_turn]
    assert [t.title for t in threads] == ["First turn", "Only turn"]
    assert all(t.updated_at for t in threads)

    other_threads = client.get_user_threads(user_id=other_user_id).threads
    assert [t.thread_id for t in other_threads] == [f"{THREAD_ID}-other"]
