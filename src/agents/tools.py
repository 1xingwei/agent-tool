import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

from core.settings import settings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
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
    """Searches the web using DuckDuckGo and returns top results.

    Args:
        query (str): The search query.
        max_results (int, optional): Max results to return. Defaults to 5.

    Returns:
        str: Formatted search results.
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


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory=settings.CHROMA_DIR, embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database
