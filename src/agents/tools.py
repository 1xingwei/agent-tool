import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

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


# 格式化检索到的文档
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # 为项目描述数据库创建嵌入函数
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # 加载已存储的向量数据库
    chroma_db = Chroma(persist_directory=settings.CHROMA_DIR, embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """在 chroma_db 中搜索公司手册中的信息。"""
    # 获取 chroma 检索器
    retriever = load_chroma_db()

    # 在数据库中搜索相关文档
    documents = retriever.invoke(query)

    # 将文档格式化为字符串
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # 将 name 更新为你的数据库用途
