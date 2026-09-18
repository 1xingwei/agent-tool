"""混合检索读模型（docs/15 P0-8）：Chroma 向量 + SQLite FTS5，RRF 融合。

向量检索能召回「语义相近但用词不同」的片段；FTS5 能精确命中关键词
（版本号、专有名词、数字）。单一信号各有所短，RRF 把两路排名融合成一路。

侧车库（`settings.CHROMA_FTS_DB`）由 `scripts/create_chroma_db.py` 在建库时重建，
是**可丢弃的派生索引**，不是真相来源（P0-9 理念）。读取侧要容忍它不存在：
缺失时退化为纯向量，保持「clone 下来就能跑、检索挂了对话不塌」的既有哲学。
"""

import logging
import os
import sqlite3

from langchain_core.documents import Document

from agents.tools import load_chroma_db
from core.settings import settings

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 平滑常数


def _fts_hits(query: str, k: int) -> list[Document]:
    """从 FTS5 侧车库召回 top-k 片段；侧车库缺失或查询失败时返回空列表。"""
    if not os.path.exists(settings.CHROMA_FTS_DB):
        return []
    try:
        conn = sqlite3.connect(settings.CHROMA_FTS_DB)
        try:
            # FTS5 的 bm25() 分数：越小越相关
            rows = conn.execute(
                "SELECT chunk_id, source, page, content FROM chunks "
                "WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
                (query, k),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.debug("FTS5 检索失败，退化为纯向量：%s", e)
        return []
    docs = []
    for chunk_id, source, page, content in rows:
        docs.append(
            Document(
                page_content=content,
                metadata={"chunk_id": chunk_id, "source": source, "page": page},
            )
        )
    return docs


def _vector_hits(query: str, k: int) -> list[Document]:
    """从 Chroma 召回 top-k 片段（进程内共享 retriever 单例）。"""
    retriever = load_chroma_db()
    return retriever.invoke(query)[:k]


def _rrf_merge(listings: list[list[Document]], k: int, top_k: int) -> list[Document]:
    """对多路排名做 RRF 融合。

    RRF 分数 = Σ 1 / (RRF_K + rank)，rank 从 1 计。按分数降序后去重取 top_k。
    """
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for listing in listings:
        for rank, doc in enumerate(listing, 1):
            chunk_id = (doc.metadata or {}).get("chunk_id")
            key = chunk_id or doc.page_content[:128]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            by_key.setdefault(key, doc)
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    return [by_key[k] for k in ranked]


def hybrid_search(query: str, top_k: int | None = None) -> list[Document]:
    """Chroma 向量 + FTS5 混合召回，RRF 融合后取 top_k（默认 `RAG_TOP_K`）。

    FTS 侧车库缺失时退化为纯向量 —— 调用方无需感知差异。
    """
    top_k = top_k or settings.RAG_TOP_K
    recall_k = settings.HYBRID_RECALL_K

    vector = _vector_hits(query, recall_k)
    fts = _fts_hits(query, recall_k)
    if not fts:
        return vector[:top_k]
    return _rrf_merge([vector, fts], recall_k, top_k)
