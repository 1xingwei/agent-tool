"""混合检索与索引对账的测试（docs/15 P0-8 / P0-9 / P0-6）。

- **P0-8**：向量与 FTS5 双路召回 + RRF 融合；FTS 侧车库缺失时退化纯向量。
- **P0-9**：源文件哈希 + 稳定 chunk id，重建幂等。
- **P0-6**：评测指标（Recall@5 / MRR）的计算正确性。

全部离线：不读真实向量库、不下载模型。
"""

import logging
import sqlite3

import pytest
from langchain_core.documents import Document

from rag import hybrid_retriever as hr


def _doc(content: str, chunk_id: str, source: str = "a.md", page: int = 0) -> Document:
    return Document(
        page_content=content, metadata={"chunk_id": chunk_id, "source": source, "page": page}
    )


# --- P0-8 RRF 融合 ---


def test_rrf_favors_documents_in_both_lists() -> None:
    """同一片段出现在两路里，融合分应高于只出现在一路里的。"""
    vector = [_doc("only-vector", "v"), _doc("both", "b")]
    fts = [_doc("only-fts", "f"), _doc("both", "b")]

    merged = hr._rrf_merge([vector, fts], top_k=3)
    keys = [(d.metadata or {}).get("chunk_id") for d in merged]

    assert keys[0] == "b", f"两路都命中的片段应排第一，实际 {keys}"
    assert set(keys) == {"b", "v", "f"}


def test_rrf_respects_top_k() -> None:
    docs = [_doc(f"d{i}", f"k{i}") for i in range(5)]
    merged = hr._rrf_merge([docs], top_k=2)
    assert len(merged) == 2


def test_rrf_dedupes_same_chunk_across_lists() -> None:
    """同一 chunk_id 不能因为两路都出现而产生两个结果。"""
    vector = [_doc("same", "dup")]
    fts = [_doc("same", "dup")]
    merged = hr._rrf_merge([vector, fts], top_k=5)
    assert len(merged) == 1


# --- P0-8 降级路径 ---


def test_fts_hits_returns_empty_when_sidecar_missing(monkeypatch) -> None:
    """FTS 侧车库不存在时必须安全返回空，而不是抛异常。"""
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", "var/does_not_exist_xyz.sqlite")
    assert hr._fts_hits("anything", 5) == []


def test_hybrid_search_falls_back_to_vector_without_sidecar(monkeypatch) -> None:
    """没有 FTS 侧车库时，结果应等于纯向量结果。"""
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", "var/does_not_exist_xyz.sqlite")
    monkeypatch.setattr(hr, "_vector_hits", lambda q, k: [_doc("v1", "k1"), _doc("v2", "k2")])

    out = hr.hybrid_search("q", top_k=2)
    assert [d.page_content for d in out] == ["v1", "v2"]


def test_hybrid_search_merges_when_sidecar_present(monkeypatch, tmp_path) -> None:
    """有 FTS 侧车库时，两路都要参与融合。"""
    db = str(tmp_path / "fts.sqlite")
    conn = sqlite3.connect(db)
    conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id, source, page, content)")
    conn.execute("INSERT INTO chunks VALUES ('f1', 'a.md', 0, 'keyword matching content')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", db)
    monkeypatch.setattr(hr, "_vector_hits", lambda q, k: [_doc("semantic", "v1")])

    out = hr.hybrid_search("keyword", top_k=2)
    texts = [d.page_content for d in out]
    assert "semantic" in texts and "keyword matching content" in texts


def test_vector_hits_honors_requested_k(monkeypatch) -> None:
    """P0-8 守卫（docs/19 R2）：向量路必须按调用方给的 k 召回。

    反例：把 `_vector_hits` 改回 `load_chroma_db().invoke(query)[:k]` —— 那个
    retriever 的 `search_kwargs` 在 `agents/tools.py` 被钉成
    `{"k": settings.RAG_TOP_K}`（默认 5），于是 `[:k]` 是永不生效的截断，
    `HYBRID_RECALL_K`（默认 20）对向量路完全失效。本用例届时必须变红。
    """
    from core.settings import settings

    class _FakeVectorStore:
        def similarity_search(self, query: str, k: int) -> list[Document]:
            return [_doc(f"d{i}", f"c{i}") for i in range(k)]

    class _FakeRetriever:
        vectorstore = _FakeVectorStore()

    monkeypatch.setattr(hr, "load_chroma_db", lambda: _FakeRetriever())

    out = hr._vector_hits("q", settings.HYBRID_RECALL_K)

    assert settings.HYBRID_RECALL_K > settings.RAG_TOP_K, "前置：默认配置下 recall_k 应大于 top_k"
    assert len(out) == settings.HYBRID_RECALL_K, "向量路没有按请求的 k 召回（R2 回归）"


def _make_sidecar(tmp_path, rows: list[tuple[str, str]]) -> str:
    """建一个含多行 chunk 的 FTS5 侧车库，返回 db 路径。"""
    db = str(tmp_path / "fts.sqlite")
    conn = sqlite3.connect(db)
    conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id, source, page, content)")
    for chunk_id, content in rows:
        conn.execute("INSERT INTO chunks VALUES (?, 'a.md', 0, ?)", (chunk_id, content))
    conn.commit()
    conn.close()
    return db


# --- P0-8 守卫（docs/20 F2a）：FTS 查询串规范化 ---


def test_fts_hits_normalizes_hostile_queries(monkeypatch, tmp_path, caplog) -> None:
    """引号/连字符/裸操作符等常见写法不得让 FTS 查询抛错，且同形 token 能命中。

    反例注入：把 `_normalize_fts_query` 改回直传原始串，本用例必须变红。
    """
    db = _make_sidecar(
        tmp_path,
        [
            ("f1", "handbook policy for employees"),
            ("f2", "mission and vision statement"),
            ("f3", "year-end bonus"),
        ],
    )
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", db)

    expected = {"f1", "f2", "f3"}
    for q in ['what is the "policy', "handbook-policy", "policy AND", "mission OR", "* draft"]:
        out = hr._fts_hits(q, 5)  # 必须不抛 OperationalError
        ids = {d.metadata["chunk_id"] for d in out}
        assert ids <= expected, f"查询 {q!r} 返回了未知 chunk: {ids}"
    # 至少能命中同形 token（handbook-policy 拆成 handbook OR policy → 命中 f1）
    out = hr._fts_hits("handbook-policy", 5)
    assert "f1" in {d.metadata["chunk_id"] for d in out}


def test_fts_hits_no_syntax_error_when_empty_after_normalization(monkeypatch, tmp_path) -> None:
    """全操作符/空查询规范化后不能抛 sqlite 语法错。"""
    db = _make_sidecar(tmp_path, [("f1", "some content")])
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", db)
    for q in ["AND", "OR NOT", "", "   "]:
        assert hr._fts_hits(q, 5) == []


def test_fts_hits_warns_when_normalization_empties_query(monkeypatch, tmp_path, caplog) -> None:
    """规范化后空查询这条空手出口也要留 warning，不能静默退化（docs/20 第五轮 V4）。

    反例注入：把 `:50` 的 `logger.warning` 删掉恢复静默，本用例必须变红。
    """
    db = _make_sidecar(tmp_path, [("f1", "some content")])
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", db)

    with caplog.at_level(logging.WARNING, logger=hr.logger.name):
        hr._fts_hits("AND NOT OR", 5)

    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_fts_hits_warns_on_zero_hits(monkeypatch, tmp_path, caplog) -> None:
    """FTS 侧车库存在但 0 命中时必须产生 warning，不能静默退化纯向量（docs/20 F2b）。

    反例注入：把 warning 降回 debug，本用例必须变红。
    """
    db = _make_sidecar(tmp_path, [("f1", "some content")])
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", db)

    with caplog.at_level(logging.WARNING, logger=hr.logger.name):
        hr._fts_hits("no-such-term-anywhere", 5)

    assert any("0 命中" in r.message or "no hits" in r.message for r in caplog.records)


def test_fts_hits_warns_on_failure(monkeypatch, tmp_path, caplog) -> None:
    """FTS 查询真实失败时必须产生 warning（docs/20 F2b）。"""

    bogus = str(tmp_path / "bogus.sqlite")
    with open(bogus, "w", encoding="utf-8") as f:
        f.write("not a sqlite database")
    monkeypatch.setattr(hr.settings, "CHROMA_FTS_DB", bogus)

    with caplog.at_level(logging.WARNING, logger=hr.logger.name):
        hr._fts_hits("anything", 5)

    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# --- P0-9 索引对账 ---


def test_file_hash_is_stable_and_content_sensitive(tmp_path) -> None:
    """同一内容哈希稳定；内容变则哈希变 —— 这是重建幂等的判据。"""
    import create_chroma_db as ingest

    f = tmp_path / "doc.md"
    f.write_text("hello world", encoding="utf-8")
    h1 = ingest.file_hash(str(f))
    h2 = ingest.file_hash(str(f))
    assert h1 == h2

    f.write_text("hello world!", encoding="utf-8")
    assert ingest.file_hash(str(f)) != h1


def test_ingest_stamps_source_hash_and_stable_ids(tmp_path) -> None:
    """建库时 chunk 必须带 source_hash，且 id 由 source+hash+序号稳定生成。"""
    import create_chroma_db as ingest
    from langchain_core.embeddings import DeterministicFakeEmbedding

    (tmp_path / "guide.md").write_text("AcmeTech mission is reliability.", encoding="utf-8")
    chroma = ingest.create_chroma_db(
        str(tmp_path),
        db_name=str(tmp_path / "db"),
        fts_db=str(tmp_path / "fts.sqlite"),  # 不隔离就会写生产索引（docs/19 R1）
        embeddings=DeterministicFakeEmbedding(size=32),
    )

    data = chroma.get()
    assert data["metadatas"], "应有写入的 chunk"
    for meta in data["metadatas"]:
        assert meta.get("source_hash"), "P0-9 回归：chunk 缺 source_hash"

    # 稳定 id：同一文件重建后 id 集合一致
    ids_1 = set(data["ids"])
    chroma2 = ingest.create_chroma_db(
        str(tmp_path),
        db_name=str(tmp_path / "db2"),
        fts_db=str(tmp_path / "fts.sqlite"),  # 同上
        embeddings=DeterministicFakeEmbedding(size=32),
    )
    assert set(chroma2.get()["ids"]) == ids_1, "P0-9 回归：同内容重建 id 不稳定"


def test_ingest_builds_fts_sidecar(tmp_path, monkeypatch) -> None:
    """建库应同时生成 FTS5 侧车库，且能被词法召回。"""
    import create_chroma_db as ingest
    from langchain_core.embeddings import DeterministicFakeEmbedding

    fts_db = str(tmp_path / "fts.sqlite")
    monkeypatch.setattr(ingest.settings, "CHROMA_FTS_DB", fts_db)
    (tmp_path / "guide.md").write_text(
        "The pension plan vests after three years.", encoding="utf-8"
    )
    ingest.create_chroma_db(
        str(tmp_path),
        db_name=str(tmp_path / "db"),
        embeddings=DeterministicFakeEmbedding(size=32),
    )

    conn = sqlite3.connect(fts_db)
    try:
        rows = conn.execute("SELECT source, content FROM chunks").fetchall()
    finally:
        conn.close()
    assert rows, "P0-8 回归：未生成 FTS 侧车库"
    assert any("pension" in content for _, content in rows)


# --- P0-6 评测指标 ---


def test_eval_hit_rank_matches_by_basename() -> None:
    import eval_retrieval as ev

    docs = [
        _doc("x", "k1", source="./data/other.md"),
        _doc("y", "k2", source="./data/target.md"),
    ]
    assert ev._hit_rank(docs, ["target.md"]) == 2
    assert ev._hit_rank(docs, ["missing.md"]) is None


def test_eval_recall_and_mrr(monkeypatch) -> None:
    import eval_retrieval as ev

    golden = [
        {"question": "q1", "expected_sources": ["target.md"], "ground_truth": ""},
        {"question": "q2", "expected_sources": ["target.md"], "ground_truth": ""},
    ]
    monkeypatch.setattr(ev, "GOLDEN_SET", golden)

    def fake_search(query, top_k=5):
        # q1 命中在第 1 位；q2 未命中
        if query == "q1":
            return [_doc("hit", "k", source="target.md")]
        return [_doc("miss", "k", source="other.md")]

    monkeypatch.setattr(ev, "hybrid_search", fake_search)
    result = ev.evaluate()

    assert result["queries"] == 2
    assert result["recall"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)  # (1/1 + 0) / 2
