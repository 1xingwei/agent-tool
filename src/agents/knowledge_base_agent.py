import logging
import os
from typing import Any, cast

from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from core import get_model, settings

logger = logging.getLogger(__name__)


# 定义状态
class AgentState(MessagesState, total=False):
    """Knowledge Base agent 的状态。"""

    remaining_steps: RemainingSteps
    retrieved_documents: list[dict[str, Any]]
    kb_documents: str


# 创建 retriever
def get_kb_retriever():
    """创建并返回一个 Knowledge Base retriever 实例。"""
    # 从环境变量获取 Knowledge Base ID
    kb_id = os.environ.get("AWS_KB_ID", "")
    if not kb_id:
        raise ValueError("AWS_KB_ID environment variable must be set")

    # 使用指定的 Knowledge Base ID 创建 retriever
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
            }
        },
    )
    return retriever


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """为 Knowledge Base agent 的模型包装系统提示词。"""

    def create_system_message(state):
        base_prompt = """You are a helpful assistant that provides accurate information based on retrieved documents.

        You will receive a query along with relevant documents retrieved from a knowledge base. Use these documents to inform your response.

        Follow these guidelines:
        1. Base your answer primarily on the retrieved documents
        2. If the documents contain the answer, provide it clearly and concisely
        3. If the documents are insufficient, state that you don't have enough information
        4. Never make up facts or information not present in the documents
        5. Always cite the source documents when referring to specific information
        6. If the documents contradict each other, acknowledge this and explain the different perspectives

        Format your response in a clear, conversational manner. Use markdown formatting when appropriate.
        """

        # 检查是否检索到文档
        if "kb_documents" in state:
            # 将文档信息追加到系统提示词
            document_prompt = f"\n\nI've retrieved the following documents that may be relevant to the query:\n\n{state['kb_documents']}\n\nPlease use these documents to inform your response to the user's query. Only use information from these documents and clearly indicate when you are unsure."
            return [SystemMessage(content=base_prompt + document_prompt)] + state["messages"]
        else:
            # 未检索到文档
            no_docs_prompt = (
                "\n\nNo relevant documents were found in the knowledge base for this query."
            )
            return [SystemMessage(content=base_prompt + no_docs_prompt)] + state["messages"]

    preprocessor = RunnableLambda(
        create_system_message,
        name="StateModifier",
    )
    return RunnableSequence(preprocessor, model)


async def retrieve_documents(state: AgentState, config: RunnableConfig) -> AgentState:
    """从知识库中检索相关文档。"""
    # 获取最后一条人类消息
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        # 包含来自原始状态的消息
        return {"messages": [], "retrieved_documents": []}

    # 使用最后一条人类消息作为查询
    query = cast(str, human_messages[-1].content)

    try:
        # 初始化 retriever
        retriever = get_kb_retriever()

        # 检索文档
        retrieved_docs = await retriever.ainvoke(query)

        # 为状态创建文档摘要
        document_summaries = []
        for i, doc in enumerate(retrieved_docs, 1):
            summary = {
                "id": doc.metadata.get("id", f"doc-{i}"),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", f"Document {i}"),
                "content": doc.page_content,
                "relevance_score": doc.metadata.get("score", 0),
            }
            document_summaries.append(summary)

        logger.info(f"Retrieved {len(document_summaries)} documents for query: {query[:50]}...")

        return {"retrieved_documents": document_summaries, "messages": []}

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"retrieved_documents": [], "messages": []}


async def prepare_augmented_prompt(state: AgentState, config: RunnableConfig) -> AgentState:
    """准备一个用检索到的文档内容增强的提示词。"""
    # 获取检索到的文档
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {"messages": []}

    # 为模型格式化检索到的文档
    formatted_docs = "\n\n".join(
        [
            f"--- Document {i + 1} ---\n"
            f"Source: {doc.get('source', 'Unknown')}\n"
            f"Title: {doc.get('title', 'Unknown')}\n\n"
            f"{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]
    )

    # 将格式化后的文档存入状态
    return {"kb_documents": formatted_docs, "messages": []}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """基于检索到的文档生成响应。"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)

    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


# 定义图
agent = StateGraph(AgentState)

# 添加节点
agent.add_node("retrieve_documents", retrieve_documents)
agent.add_node("prepare_augmented_prompt", prepare_augmented_prompt)
agent.add_node("model", acall_model)

# 设置入口点
agent.set_entry_point("retrieve_documents")

# 添加边以定义流程
agent.add_edge("retrieve_documents", "prepare_augmented_prompt")
agent.add_edge("prepare_augmented_prompt", "model")
agent.add_edge("model", END)

# 编译 agent
kb_agent = agent.compile()
