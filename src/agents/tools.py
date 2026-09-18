import ipaddress
import math
import re
from html import unescape

import httpx
import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool

from core.embeddings import get_embedding_model
from core.settings import settings


def calculator_func(expression: str) -> str:
    """使用 numexpr 计算数学表达式。

    当你需要使用 numexpr 回答数学问题时很有用。
    此工具仅用于数学问题，不用于其他任何用途。只输入
    数学表达式。

    Args:
        expression (str): 有效的 numexpr 格式数学表达式。

    Returns:
        str: 数学表达式的结果。
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # 限制对全局变量的访问
                local_dict=local_dict,  # 添加常用数学函数
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


def web_search_func(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网页并返回顶部结果。

    Args:
        query (str): 搜索查询。
        max_results (int, optional): 返回的最大结果数。默认为 5。

    Returns:
        str: 格式化的搜索结果。
    """
    from ddgs import DDGS

    proxy = settings.WEB_SEARCH_PROXY
    backends = settings.WEB_SEARCH_BACKENDS.split(",")
    last_error = ""
    for backend in backends:
        for _ in range(2):
            try:
                with DDGS(proxy=proxy or None, timeout=10) as ddgs:
                    results = ddgs.text(query, max_results=max_results, backend=backend)
                if results:
                    return "\n\n".join(
                        f"{i}. {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
                        for i, r in enumerate(results, 1)
                    )
            except Exception as e:
                last_error = str(e)
    return f"No search results found. {last_error}"


web_search: BaseTool = tool(web_search_func)
web_search.name = "WebSearch"


FETCH_URL_HARD_CAP = 100_000  # 响应体硬上限（字符），防模型把上下文撑爆


def _is_private_address(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # 解析不出合法 IP，宁拦勿放
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def _hostname_blocked(hostname: str) -> bool:
    hostname = hostname.lower()
    return hostname == "localhost" or hostname.endswith(".internal") or hostname.endswith(".local")


def _validate_url_safe(url: str) -> bool:
    """SSRF 防护：URL 白名单 + 解析后的 IP 私网检测（docs/11 §1.3）。

    hostname 是 IP 字面量时直接校验该 IP（跳过 DNS —— DNS 结果可能已被 mock /
    污染，字面量本身即可判定）；是域名时校验其全部 A/AAAA 记录。
    """
    if not url.startswith(("http://", "https://")):
        return False
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname
    if not hostname or _hostname_blocked(hostname):
        return False
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return not _is_private_address(hostname)
    try:
        import socket

        addrinfo = socket.getaddrinfo(hostname, None)
    except OSError:
        return False
    return all(_is_private_address(str(sockaddr[0])) is False for _, _, _, _, sockaddr in addrinfo)


def fetch_url_func(url: str, max_chars: int = 4000) -> str:
    """抓取网页并返回其可见文本。

    在 web_search 之后使用，当搜索摘要不含答案时 —— 版本号、日期和表格通常
    只在页面正文里。结果按段落截断。

    Args:
        url (str): 要抓取的 http / https 地址。
        max_chars (int, optional): 返回文本的最大字符数（上限 100000）。
    """
    if not _validate_url_safe(url):
        return "ERROR: URL is not allowed (must be public http/https)."
    limit = min(max_chars, FETCH_URL_HARD_CAP)
    timeout = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=15.0)
    try:
        with httpx.Client(
            proxy=settings.WEB_SEARCH_PROXY or None,
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            current = url
            for _ in range(3):
                resp = client.get(current, headers={"User-Agent": "agent-tool"})
                content_length = resp.headers.get("content-length")
                if content_length and int(content_length) > limit * 4:
                    return "ERROR: page too large to fetch."
                if resp.is_redirect:
                    current = resp.headers.get("location")
                    if not current or not _validate_url_safe(current):
                        return "ERROR: redirect target is not allowed."
                    continue
                resp.raise_for_status()
                text = _html_to_text(resp.text)
                return text[:limit]
            return "ERROR: too many redirects."
    except httpx.HTTPError as e:
        return f"ERROR: could not fetch URL: {e}"
    except ValueError as e:
        return f"ERROR: could not fetch URL: {e}"


def _html_to_text(html: str) -> str:
    """剥掉 script/style/svg 块与标签，把实体还原为可读文本。"""
    html = re.sub(r"<(script|style|svg)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = unescape(html)
    return re.sub(r"[ \t]+", " ", html)


fetch_url: BaseTool = tool(fetch_url_func)
fetch_url.name = "fetch_url"


# 格式化检索到的文档：保留 metadata，让模型能引用来源
def format_contexts(docs) -> str:
    """把检索到的文档格式化为带来源标注的上下文。

    原先只取 `page_content`、把 `doc.metadata` 整个丢掉，导致
    `rag_assistant` 的 instructions 要求模型「给出来源链接」，
    但模型手里根本没有任何来源信息 —— 只能编或干脆不给。
    这里把 source / page 提到每段之前，使引用可追溯。
    """
    if not docs:
        return ""

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("title") or "unknown"
        page = meta.get("page")
        # PyPDFLoader 的 page 从 0 开始计数，展示时转成人类习惯的 1-based
        loc = f", 第 {page + 1} 页" if isinstance(page, int) else ""
        parts.append(f"[来源 {i}: {source}{loc}]\n{doc.page_content}")
    return "\n\n".join(parts)


# retriever 与 embedding 客户端在进程内只构建一次。
# 原实现每次工具调用都重建 OpenAIEmbeddings + Chroma + retriever，
# 而 Chroma 的客户端构造会读持久化目录，属于明显的热路径浪费。
_chroma_retriever = None


def load_chroma_db():
    """返回进程内共享的 Chroma retriever（惰性单例）。

    embedding 走 `core.embeddings`，与长期记忆 store 同源；
    默认本地 fastembed，因此没有 OpenAI key 也能建库与检索。
    """
    global _chroma_retriever
    if _chroma_retriever is not None:
        return _chroma_retriever

    embeddings = get_embedding_model()
    if embeddings is None:
        raise RuntimeError(
            "无法初始化 embedding；请检查 EMBEDDING_PROVIDER / OPENAI_API_KEY 配置。"
        )

    chroma_db = Chroma(persist_directory=settings.CHROMA_DIR, embedding_function=embeddings)
    _chroma_retriever = chroma_db.as_retriever(search_kwargs={"k": settings.RAG_TOP_K})
    return _chroma_retriever


def reset_chroma_retriever() -> None:
    """清空缓存的 retriever，供测试与建库后刷新使用。"""
    global _chroma_retriever
    _chroma_retriever = None


def database_search_func(query: str) -> str:
    """在 chroma_db（向量）+ FTS5（词法）中搜索公司手册中的信息。"""
    from rag.hybrid_retriever import hybrid_search

    documents = hybrid_search(query)
    return format_contexts(documents)


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # 将 name 更新为你的数据库用途
