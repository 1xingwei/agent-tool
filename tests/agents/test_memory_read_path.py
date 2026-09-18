"""长期记忆读写路径的测试（docs/15 §7.2 的 P0-1 / P0-2）。

这些用例守的是两个曾经静默失效的契约：

- **P0-1**：store 不传 `index` 时 LangGraph 会**静默关闭语义检索** ——
  `asearch(query=...)` 不报错、仍返回结果，但按主键序且 `score is None`。
  因此断言必须落在 `score is not None`，而不是「结果非空」。
- **P0-2**：`remember_review` 曾经只写不读，结论写进去了但从未被使用。

用例分两组：
- 不需要 embedding 的（用 `InMemoryStore` / 假 embedding），无网可跑；
- 需要真实本地模型的，标 `network`（首次会下载模型）。
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import AsyncSqliteStore

from agents.code_reviewer import recall_reviews


def _index(dims: int = 64):
    """构造一个确定性的假 embedding 索引配置（离线、可复现）。"""
    return {"dims": dims, "embed": DeterministicFakeEmbedding(size=dims), "fields": ["$"]}


@pytest.mark.asyncio
async def test_store_without_index_returns_none_scores(tmp_path) -> None:
    """守卫 P0-1：不传 `index` 时 score 必为 None（静默降级的确切形态）。

    这条用例的意义是**把「静默」变成「显式」**：如果将来有人在
    `get_sqlite_store` 里误删了 `index` 参数，或者 LangGraph 改了默认行为，
    这里会失败，而不是悄悄退化成主键序。
    """
    db = str(tmp_path / "no_index.sqlite")
    async with AsyncSqliteStore.from_conn_string(db) as store:
        await store.setup()
        await store.aput(("ns", "u"), "k1", {"conclusion": "a"})
        await store.aput(("ns", "u"), "k2", {"conclusion": "b"})

        items = await store.asearch(("ns", "u"), query="anything", limit=5)
        assert items, "无 index 时仍返回结果 —— 这正是它危险的地方"
        assert all(getattr(it, "score", None) is None for it in items)


@pytest.mark.asyncio
async def test_store_with_index_returns_scores(tmp_path) -> None:
    """守卫 P0-1 的目标状态：传了 index，score 才有值且按降序。"""
    db = str(tmp_path / "with_index.sqlite")
    async with AsyncSqliteStore.from_conn_string(db, index=_index()) as store:
        await store.setup()
        ns = ("ns", "u")
        for i in range(6):
            await store.aput(ns, f"k{i}", {"conclusion": f"memory {i}"})

        items = await store.asearch(ns, query="memory 3", limit=6)
        scores = [it.score for it in items]
        assert all(s is not None for s in scores), "P0-1 回归：score 又变成 None"
        assert scores == sorted(scores, reverse=True), "未按相似度降序"


@pytest.mark.asyncio
async def test_get_sqlite_store_configures_index(tmp_path, monkeypatch) -> None:
    """守卫 `get_sqlite_store` 真的把 index 传下去了。

    直接断言构造参数，而不是只断言「能拿到 store」—— 后者在 index 缺失时也成立。
    """
    from memory import sqlite as sqlite_mod

    monkeypatch.setattr(sqlite_mod.settings, "SQLITE_STORE_PATH", str(tmp_path / "s.sqlite"))
    monkeypatch.setattr(sqlite_mod, "build_store_index", lambda: _index())

    async with sqlite_mod.get_sqlite_store() as store:
        assert store.index_config is not None, "P0-1 回归：index 未传入"
        assert store.index_config["dims"] == 64


@pytest.mark.asyncio
async def test_recall_reviews_returns_matching_memory(tmp_path) -> None:
    """守卫 P0-2：读路径能把已写入的结论召回出来。"""
    db = str(tmp_path / "recall.sqlite")
    async with AsyncSqliteStore.from_conn_string(db, index=_index()) as store:
        await store.setup()
        ns = ("code-reviewer", "u1")
        await store.aput(ns, "review-1", {"conclusion": "登录模块硬编码了密钥，需移入环境变量。"})
        await store.aput(ns, "review-2", {"conclusion": "分页接口缺少上限，可能被拖库。"})

        out = await recall_reviews(
            {"messages": [HumanMessage(content="登录密钥怎么处理")]},  # type: ignore[arg-type]
            {"configurable": {"user_id": "u1"}},
            store,
        )

    recalled = out["recalled_reviews"]
    assert recalled, "P0-2 回归：读路径返回空"
    assert "密钥" in recalled
    assert "相似度" in recalled, "召回内容应带相似度，便于观测"


@pytest.mark.asyncio
async def test_recall_reviews_skips_when_scores_are_none() -> None:
    """读路径必须区分「检索到」和「没检索」。

    store 未开启语义检索时 `asearch` 仍返回结果（主键序、score=None）。
    此时把内容当成「相关历史」注入提示词是**错误信息**，
    必须跳过而不是照抄。
    """
    store = InMemoryStore()  # 未配置 index → score 恒为 None
    await store.aput(("code-reviewer", "u1"), "k1", {"conclusion": "无关内容"})

    out = await recall_reviews(
        {"messages": [HumanMessage(content="任意问题")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "u1"}},
        store,
    )
    assert out["recalled_reviews"] == "", "score 无效时不应注入召回内容"


@pytest.mark.asyncio
async def test_recall_reviews_without_store_is_noop() -> None:
    """与 `remember_review` 对齐：store=None 时安全返回。

    注意 `messages: []` 是 `MessagesState` 的必填键，本节点不改动消息仍需显式带上，
    所以只断言召回内容为空，不比对整个字典。
    """
    out = await recall_reviews(
        {"messages": [HumanMessage(content="x")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "u1"}},
        None,
    )
    assert out["recalled_reviews"] == ""


@pytest.mark.asyncio
async def test_recall_reviews_isolates_namespace() -> None:
    """不同 user_id 的记忆不能互相串。"""
    store = InMemoryStore(index=_index())
    await store.aput(("code-reviewer", "alice"), "k", {"conclusion": "alice 的秘密"})

    out = await recall_reviews(
        {"messages": [HumanMessage(content="秘密")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "bob"}},
        store,
    )
    assert "alice" not in out["recalled_reviews"]


@pytest.mark.asyncio
async def test_recall_reviews_handles_empty_and_no_human_message() -> None:
    """边界：没有人类消息、或内容为空时都应安全返回。"""
    store = InMemoryStore(index=_index())

    out = await recall_reviews(
        {"messages": [AIMessage(content="assistant only")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "u1"}},
        store,
    )
    assert out["recalled_reviews"] == ""

    out2 = await recall_reviews(
        {"messages": [HumanMessage(content="   ")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "u1"}},
        store,
    )
    assert out2["recalled_reviews"] == ""


@pytest.mark.asyncio
async def test_recall_failure_does_not_break_review(monkeypatch) -> None:
    """store 抛异常时，读路径失败不应中断本次审查。"""

    class ExplodingStore:
        async def asearch(self, *a, **kw):
            raise RuntimeError("store down")

    out = await recall_reviews(
        {"messages": [HumanMessage(content="问题")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "u1"}},
        ExplodingStore(),  # type: ignore[arg-type]
    )
    assert out["recalled_reviews"] == ""


@pytest.mark.asyncio
async def test_recall_reviews_returns_messages_key() -> None:
    """结构契约：返回值必须带 `messages` 键。

    `MessagesState` 把 `messages` 声明为必填，节点返回的增量状态若缺这个键，
    类型检查会报 `Missing required key 'messages' for TypedDict`。
    """
    out = await recall_reviews(
        {"messages": [HumanMessage(content="x")]},  # type: ignore[arg-type]
        {"configurable": {"user_id": "u1"}},
        None,
    )
    assert "messages" in out


def test_recalled_reviews_is_injected_into_system_prompt() -> None:
    """守卫 P0-2 的最后一环：召回内容真的进了 SystemMessage。"""
    from agents.code_reviewer import wrap_model
    from core.llm import FakeToolModel

    captured: dict = {}

    class SpyModel(FakeToolModel):
        def invoke(self, input, config=None, **kwargs):  # type: ignore[override]
            captured["messages"] = input
            return super().invoke(input, config, **kwargs)

    runnable = wrap_model(SpyModel(responses=["ok"]))  # type: ignore[arg-type]
    runnable.invoke(
        {
            "messages": [HumanMessage(content="登录密钥")],
            "recalled_reviews": "- (review-1, 相似度 0.660) 硬编码密钥需移入环境变量",
        }
    )

    system = captured["messages"][0].content
    assert "长期记忆" in system
    assert "硬编码密钥" in system


def test_model_sees_folded_view() -> None:
    """折叠必须发生在 model 的入参上 —— 这是 P0-7 唯一有意义的判据。

    反例：把 apply_distillation 从 build_messages 里去掉，本用例必须变红。
    """
    from core.llm import FakeToolModel

    captured: dict = {}

    class SpyModel(FakeToolModel):
        def invoke(self, input, config=None, **kwargs):  # type: ignore[override]
            captured["count"] = len(input) if isinstance(input, list) else None
            return super().invoke(input, config, **kwargs)

    from agents.code_reviewer import wrap_model

    # 30 条远超 `settings.DISTILL_KEEP_MESSAGES`（默认 12），必然触发折叠。
    history = [HumanMessage(content=f"第 {i} 条") for i in range(30)]
    runnable = wrap_model(SpyModel(responses=["ok"]))  # type: ignore[arg-type]
    runnable.invoke({"messages": history, "distilled_summary": "摘要"})

    assert captured["count"] < len(history), "model 没有看到折叠后的视图"


def test_code_reviewer_graph_has_recall_before_model() -> None:
    """结构守卫：recall 必须在 model 之前，否则召回内容进不了提示词。

    P0-7 之后 model 前面多了一层 `distill_history`，因此这条断言的是
    **可达性**而不是直接连边：只要从 `recall_reviews` 出发能走到 `model`，
    且路上不绕过 model，召回内容就仍然进得了提示词。直接钉死
    `("recall_reviews", "model")` 会让每次插入新节点都误报一次。
    """
    from agents.code_reviewer import code_reviewer

    graph = code_reviewer.get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    # 从 recall_reviews 出发做一次可达性搜索
    seen = {"recall_reviews"}
    frontier = ["recall_reviews"]
    while frontier:
        node = frontier.pop()
        for source, target in edges:
            if source == node and target not in seen:
                seen.add(target)
                frontier.append(target)

    assert "model" in seen, "recall_reviews 必须能走到 model，否则召回内容进不了提示词"
    assert ("guard_input", "model") not in edges, "guard_input 应经 recall_reviews 再到 model"
    # 蒸馏节点必须也在 model 之前，否则它折叠的消息模型看不到
    assert ("distill_history", "model") in edges
