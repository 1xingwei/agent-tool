"""检索工具的测试（docs/15 §7.2 的 P0-3 / P0-4）。

- **P0-3**：`format_contexts` 曾经只取 `page_content`、丢掉全部 metadata，
  而 `rag_assistant` 的 instructions 却要求模型给出来源 —— 模型手里没有来源信息。
- **P0-4**：`load_chroma_db` 曾经每次工具调用都重建 embedding + Chroma + retriever。

全部离线（不读真实向量库、不调用模型）。
"""

import pytest
from langchain_core.documents import Document

import agents.tools as tools


def _doc(content: str, **meta) -> Document:
    return Document(page_content=content, metadata=meta)


def test_format_contexts_keeps_source_and_page() -> None:
    """P0-3：来源与页码必须出现在输出里。"""
    out = tools.format_contexts(
        [
            _doc("使命是让协作更简单。", source="./data/handbook.pdf", page=2),
            _doc("年假 15 天。", source="./data/benefits.pdf", page=0),
        ]
    )
    assert "[来源 1: ./data/handbook.pdf, 第 3 页]" in out
    assert "[来源 2: ./data/benefits.pdf, 第 1 页]" in out
    assert "使命是让协作更简单" in out


def test_format_contexts_page_is_one_based() -> None:
    """PyPDFLoader 的 page 从 0 开始，展示要给人类看的 1-based 页码。"""
    out = tools.format_contexts([_doc("x", source="a.pdf", page=0)])
    assert "第 1 页" in out, "page=0 应显示为第 1 页"


def test_format_contexts_falls_back_when_metadata_missing() -> None:
    """没有 source 时回退为 unknown，而不是抛出或显示 None。"""
    out = tools.format_contexts([_doc("无来源片段"), _doc("有待标题", title="手册")])
    assert "来源 1: unknown" in out
    assert "来源 2: 手册" in out


def test_format_contexts_handles_empty_input() -> None:
    assert tools.format_contexts([]) == ""
    assert tools.format_contexts(None) == ""  # type: ignore[arg-type]


def test_format_contexts_does_not_drop_content() -> None:
    """守卫原缺陷：正文本身不能被弄丢。"""
    docs = [_doc(f"body-{i}", source=f"s{i}.pdf") for i in range(3)]
    out = tools.format_contexts(docs)
    for i in range(3):
        assert f"body-{i}" in out


def test_load_chroma_db_is_a_singleton(monkeypatch) -> None:
    """P0-4：连续调用只应构造一次 Chroma。"""
    calls = {"n": 0}
    real_chroma = tools.Chroma

    class CountingChroma(real_chroma):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            calls["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tools, "Chroma", CountingChroma)
    monkeypatch.setattr(tools, "get_embedding_model", lambda: _FakeEmbeddings())
    tools.reset_chroma_retriever()

    try:
        first = tools.load_chroma_db()
        for _ in range(4):
            assert tools.load_chroma_db() is first
        assert calls["n"] == 1, f"P0-4 回归：Chroma 构造了 {calls['n']} 次"
    finally:
        tools.reset_chroma_retriever()


def test_load_chroma_db_raises_actionable_error_when_no_embedding(monkeypatch) -> None:
    """embedding 不可用时要给出可操作的错误，而不是一个含糊的 RuntimeError。"""
    monkeypatch.setattr(tools, "get_embedding_model", lambda: None)
    tools.reset_chroma_retriever()

    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER|OPENAI_API_KEY"):
        tools.load_chroma_db()


def test_rag_top_k_is_configurable(monkeypatch) -> None:
    """召回条数走配置，不再是硬编码的 k=5。"""
    from core.settings import settings

    monkeypatch.setattr(settings, "RAG_TOP_K", 9)
    monkeypatch.setattr(tools, "get_embedding_model", lambda: _FakeEmbeddings())

    captured: dict = {}
    real_chroma = tools.Chroma

    class CapturingChroma(real_chroma):  # type: ignore[misc,valid-type]
        def as_retriever(self, **kwargs):
            captured.update(kwargs)
            return super().as_retriever(**kwargs)

    monkeypatch.setattr(tools, "Chroma", CapturingChroma)
    tools.reset_chroma_retriever()
    try:
        tools.load_chroma_db()
    finally:
        tools.reset_chroma_retriever()

    assert captured.get("search_kwargs", {}).get("k") == 9


class _FakeEmbeddings:
    """最小 embedding 桩：只需要维度自洽即可，避免测试触发模型下载。"""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]
