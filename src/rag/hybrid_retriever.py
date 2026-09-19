"""混合检索读模型（docs/15 P0-8）：Chroma 向量 + SQLite FTS5，RRF 融合。

向量检索能召回「语义相近但用词不同」的片段；FTS5 能精确命中关键词
（版本号、专有名词、数字）。单一信号各有所短，RRF 把两路排名融合成一路。

侧车库（`settings.CHROMA_FTS_DB`）由 `scripts/create_chroma_db.py` 在建库时重建，
是**可丢弃的派生索引**，不是真相来源（P0-9 理念）。读取侧要容忍它不存在：
缺失时退化为纯向量，保持「clone 下来就能跑、检索挂了对话不塌」的既有哲学。
"""

import logging
import os
import re
import sqlite3

from langchain_core.documents import Document

from agents.tools import load_chroma_db
from core.settings import settings

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 平滑常数

_FTS_STOPWORDS = {"AND", "OR", "NOT", "NEAR"}
_FTS_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff\u3400-\u4dbf]+")


def _normalize_fts_query(query: str) -> str:
    """把用户原始串转成安全的 FTS5 MATCH 表达式。

    直接吃原始串会让连字符词（被解析成列名）、引号（unterminated string）、
    裸 AND/OR（语法错）等常见自然语言写法抛 OperationalError（docs/20 F2a）。
    这里按非字母数字切分，丢弃 FTS5 操作符，每个 token 加双引号后用 OR 连接，
    从而让「查询健壮性」与「词级精确召回」两者兼得。返回空串表示无从查询。
    """
    tokens = [
        t for t in _FTS_TOKEN_RE.findall(query) if t.upper() not in _FTS_STOPWORDS and t.strip()
    ]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _fts_hits(query: str, k: int) -> list[Document]:
    """从 FTS5 侧车库召回 top-k 片段；侧车库缺失或查询失败时返回空列表。"""
    if not os.path.exists(settings.CHROMA_FTS_DB):
        return []
    normalized = _normalize_fts_query(query)
    if not normalized:
        logger.warning("FTS5 查询规范化后为空，退化为纯向量（query=%r）", query)
        return []
    try:
        conn = sqlite3.connect(settings.CHROMA_FTS_DB)
        try:
            # FTS5 的 bm25() 分数：越小越相关
            rows = conn.execute(
                "SELECT chunk_id, source, page, content FROM chunks "
                "WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
                (normalized, k),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("FTS5 检索失败，退化为纯向量：%s", e)
        return []
    docs = []
    for chunk_id, source, page, content in rows:
        docs.append(
            Document(
                page_content=content,
                metadata={"chunk_id": chunk_id, "source": source, "page": page},
            )
        )
    if not docs:
        logger.warning("FTS5 侧车库存在但 0 命中：%r（query=%r）", normalized, query)
    return docs


def _vector_hits(query: str, k: int) -> list[Document]:
    """从 Chroma 召回 top-k 片段。

    **不复用** `load_chroma_db()` 返回的那个 retriever：它的 `search_kwargs` 在
    `agents/tools.py` 构造时就被钉成 `{"k": settings.RAG_TOP_K}`（默认 5），
    所以 `retriever.invoke(query)[:k]` 里的 `[:k]` 是个**永不生效的截断** ——
    调用方传 20 只会拿到 5 条，`HYBRID_RECALL_K`（默认 20）对向量路完全失效，
    两路候选池不对等，RRF 融合会系统性偏向 FTS（docs/19 R2）。
    这里直接向底层 vectorstore 要 k 条，让 `k` 真正生效。
    """
    return load_chroma_db().vectorstore.similarity_search(query, k=k)


def _rrf_merge(listings: list[list[Document]], top_k: int) -> list[Document]:
    """对多路排名做 RRF 融合。

    平滑常数固定取模块级 `RRF_K`、**不由调用方传** —— 它定义的是 RRF 算法本身，
    不是每次检索的可调项（此前签名里挂着一个从未被使用的 `k` 参数，容易让人
    以为融合行为可调）。RRF 分数 = Σ 1 / (RRF_K + rank)，rank 从 1 计；
    按分数降序后去重取 top_k。
    """
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for listing in listings:
        for rank, doc in enumerate(listing, 1):
            chunk_id = (doc.metadata or {}).get("chunk_id")
            key = chunk_id or doc.page_content[:128]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            by_key.setdefault(key, doc)
    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k]
    return [by_key[key] for key in ranked]


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
    return _rrf_merge([vector, fts], top_k)
