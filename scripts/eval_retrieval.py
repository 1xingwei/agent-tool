"""RAG 检索评测（docs/15 P0-6）：在 golden set 上跑 Recall@5 / MRR。

用法（先建库，再评测）：

    uv run python scripts/create_chroma_db.py --folder data
    uv run python scripts/eval_retrieval.py

指标口径（与 docs/15 §7.7 一致，纯本地零 key）：
- Recall@5：golden 里「应命中的 source」有多少出现在前 5 个召回片段里。
- MRR：每个 query 的第一个命中的逆排名（1/rank），取平均。

输出一张按文件分组的对比表，改检索配置前后可横向对比。
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrieval_golden_set import GOLDEN_SET

from rag.hybrid_retriever import hybrid_search


def _basename(meta_source: str) -> str:
    """把召回结果的 source 归一化成文件名（去掉路径前后缀）。"""
    return Path(str(meta_source)).name


def _hit_rank(docs, expected_sources: list[str]) -> int | None:
    """返回第一个命中的 rank（1-based）；全部未命中返回 None。"""
    for rank, doc in enumerate(docs[:5], 1):
        src = _basename((doc.metadata or {}).get("source", ""))
        if src in expected_sources:
            return rank
    return None


def evaluate(recall_at: int = 5) -> dict:
    per_source = defaultdict(lambda: {"hits": 0, "rr": 0.0, "total": 0})
    for item in GOLDEN_SET:
        expected = item["expected_sources"]
        docs = hybrid_search(item["question"], top_k=recall_at)
        rank = _hit_rank(docs, expected)
        for src in expected:
            per_source[src]["total"] += 1
            if rank is not None:
                per_source[src]["hits"] += 1
                per_source[src]["rr"] += 1.0 / max(rank, 1)

    table = []
    total_hits = total_queries = total_rr = 0
    for src in sorted(per_source):
        row = per_source[src]
        recall = row["hits"] / row["total"] if row["total"] else 0.0
        mrr = row["rr"] / row["total"] if row["total"] else 0.0
        table.append((src, recall, mrr))
        total_hits += row["hits"]
        total_queries += row["total"]
        total_rr += row["rr"]

    overall_recall = total_hits / total_queries if total_queries else 0.0
    overall_mrr = total_rr / total_queries if total_queries else 0.0

    print(f"{'来源':<32} {f'Recall@{recall_at}':>10} {'MRR':>8}")
    print("-" * 52)
    for src, recall, mrr in table:
        print(f"{src:<32} {recall:>10.3f} {mrr:>8.3f}")
    print("-" * 52)
    print(f"{'合计':<32} {overall_recall:>10.3f} {overall_mrr:>8.3f}")

    return {
        "recall": overall_recall,
        "mrr": overall_mrr,
        "queries": total_queries,
        "per_source": {src: {"recall": r, "mrr": m} for src, r, m in table},
    }


if __name__ == "__main__":
    evaluate()
