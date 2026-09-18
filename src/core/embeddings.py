"""Embedding 提供方抽象：为长期记忆 store 的语义检索提供向量化能力。

设计约束（对齐项目既有哲学）：

1. **默认零外部服务**：默认使用本地 fastembed（ONNX），不需要任何 API key。
2. **不破坏分层**：`core` 层只暴露「拿到一个 Embeddings 实例」的契约，
   不知道 store 的存在；`memory` 层消费它。
3. **降级而非崩溃**：任何构造失败都返回 `None` 并由调用方决定是否关闭语义检索，
   **不抛异常** —— 因为语义检索是增强项，不是对话的前提。

为什么不直接 `OpenAIEmbeddings()`：本机只配置了 DeepSeek，没有 `OPENAI_API_KEY`，
硬编码 OpenAI 会让需要向量化的功能整体不可用（实测抛 RuntimeError）。
"""

import logging
from enum import StrEnum
from typing import Any

from langchain_core.embeddings import Embeddings

from core.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(StrEnum):
    """Embedding 提供方。"""

    LOCAL = "local"
    """本地 fastembed（ONNX），默认。无 API key 依赖。"""

    OPENAI = "openai"
    """OpenAI 兼容接口，需要 OPENAI_API_KEY。"""


def get_embedding_model() -> Embeddings | None:
    """按配置构造 embedding 实例；失败时返回 `None` 并告警。

    **构造后立刻做一次探针嵌入**，而不是只返回对象：`FastEmbedEmbeddings` 等实现
    是惰性加载的，模型文件损坏或缺失时构造函数不会报错，
    错误要等到第一次真正 `embed_query` 才暴露。提前暴露可以让上层
    在启动时就知道语义检索是否可用，而不是在用户提问时才失败。
    """
    provider = settings.EMBEDDING_PROVIDER

    try:
        if provider == EmbeddingProvider.LOCAL:
            embeddings = _build_local()
        elif provider == EmbeddingProvider.OPENAI:
            embeddings = _build_openai()
        else:
            logger.warning(
                "Unknown EMBEDDING_PROVIDER %r; store semantic search disabled.", provider
            )
            return None

        # 强制加载并验证：惰性实现（fastembed）在这里才会真正读模型
        embeddings.embed_query("warmup")
        return embeddings
    except Exception as e:
        logger.warning(
            "Embedding provider %r unavailable (%s: %s); "
            "store semantic search will be disabled and the store falls back to plain KV.",
            provider,
            type(e).__name__,
            e,
        )
        return None


def _build_local() -> Embeddings:
    """构造本地 fastembed embedding。

    `langchain_community` 的 `FastEmbedEmbeddings` 依赖 `fastembed` 包；
    它不在默认依赖里，缺失时抛出带安装提示的 ImportError。
    """
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
    except ImportError as e:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "EMBEDDING_PROVIDER=local 需要 fastembed：请执行 `uv add fastembed`。"
        ) from e

    kwargs = {"model_name": settings.EMBEDDING_MODEL}
    if settings.EMBEDDING_CACHE_DIR:
        kwargs["cache_dir"] = settings.EMBEDDING_CACHE_DIR
    return FastEmbedEmbeddings(**kwargs)


def _build_openai() -> Embeddings:
    """构造 OpenAI embedding；未配置 key 时抛出明确错误。"""
    from langchain_openai import OpenAIEmbeddings

    api_key = settings.OPENAI_API_KEY
    if api_key is None:
        raise ValueError("EMBEDDING_PROVIDER=openai 需要设置 OPENAI_API_KEY")
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=api_key)


def build_store_index() -> dict[str, Any] | None:
    """构造 store 的 `index` 配置；embedding 不可用时返回 `None`。

    LangGraph 的 store **不传 `index` 时语义检索是关闭的**，且行为是静默的
    （实测：返回插入序、`score` 恒为 `None`，不报错）。所以这里显式构造，
    并让调用方知道是否真的开启了。

    Returns:
        dict | None: 形如 `{"dims": int, "embed": Embeddings, "fields": ["$"]}`；
            或 None 表示应省略 `index` 参数、退回纯 KV。
    """
    embeddings = get_embedding_model()
    if embeddings is None:
        return None

    dims = settings.EMBEDDING_DIMS
    if dims is None:
        dims = _probe_dims(embeddings)
        if dims is None:
            return None

    return {"dims": dims, "embed": embeddings, "fields": ["$"]}


def _probe_dims(embeddings: Embeddings) -> int | None:
    """通过嵌入一个探针字符串推断向量维度。

    仅当 `EMBEDDING_DIMS` 未显式配置时使用。维度必须与 collection 一致，
    否则写入会失败，所以这里宁可多算一次也不猜。

    注意 `bge-small-zh-v1.5` 是 **512** 维（不是常被误记的 384），
    所以不要凭模型名字硬编码维度。
    """
    try:
        return len(embeddings.embed_query("dimension probe"))
    except Exception as e:
        logger.warning(
            "Failed to probe embedding dimensions (%s: %s); store semantic search disabled.",
            type(e).__name__,
            e,
        )
        return None
