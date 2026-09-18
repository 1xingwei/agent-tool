import hashlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.embeddings import get_embedding_model
from core.settings import settings

# 从 .env 文件加载环境变量
load_dotenv()


def file_hash(file_path: str) -> str:
    """计算源文件 sha256，作为 P0-9 索引对账字段（重建幂等的依据）。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def create_chroma_db(
    folder_path: str,
    db_name: str | None = None,
    fts_db: str | None = None,
    delete_chroma_db: bool = False,
    chunk_size: int = 2000,
    overlap: int = 500,
    embeddings: Embeddings | None = None,
):
    db_name = db_name or settings.CHROMA_DIR
    # FTS 侧车库同样允许调用方覆盖：默认落 settings.CHROMA_FTS_DB（相对 CWD 的 var/），
    # 但该默认值是**生产索引**，因此测试必须显式传 tmp_path，否则会把测试语料
    # 写进真实索引（`_build_fts_sidecar` 会先 DROP TABLE 再重建）。见 docs/19 R1。
    fts_db = fts_db or settings.CHROMA_FTS_DB
    if embeddings is None:
        # 与检索侧共用同一个 embedding 来源（core.embeddings）。
        # 建库与检索必须用**同一个模型**，否则向量空间不一致、检索结果静默失真。
        embeddings = get_embedding_model()
        if embeddings is None:
            raise RuntimeError(
                "无法初始化 embedding。请检查 EMBEDDING_PROVIDER（默认 local，需 `uv add fastembed`）"
                "或 OPENAI_API_KEY 配置。"
            )

    # 初始化 Chroma 向量存储
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=db_name,
    )

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # 遍历文件夹中的文件
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # 根据文件扩展名加载文档
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith((".md", ".txt")):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue  # 跳过不支持的文件类型

        # 加载文档并分割为块
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # P0-9 索引对账：每个 chunk 带源文件哈希 + 稳定 id（source+hash+序号），
        # 重建幂等 —— Chroma 按 id 覆盖，源文件未变时重复建库结果一致。
        source_hash = file_hash(file_path)
        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("source", filename)
            chunk.metadata["source_hash"] = source_hash
            chunk.metadata.setdefault("page", 0)
            ids.append(f"{filename}:{source_hash[:12]}:{i}")

        # 将块添加到 Chroma 向量存储
        chroma.add_documents(chunks, ids=ids)
        print(f"Document {filename} added to database.")

    _build_fts_sidecar(fts_db, folder_path, text_splitter)
    print(f"Vector database created and saved in {db_name}.")
    return chroma


def _build_fts_sidecar(fts_db: str, folder_path: str, text_splitter) -> None:
    """建 FTS5 侧车库（docs/15 P0-8）：与 Chroma 同 chunk、同 id，支撑词法召回。

    Chroma 只做向量；关键词命中（版本号、专有名词）要靠 FTS5。这里把在建库时
    已经切好的 chunk 原文按同一 id 写入 SQLite FTS5，供读取侧做 RRF 融合。
    侧车库是**可重建的派生索引**（P0-9 理念）：随时可删掉重建，不是真相来源。
    """
    import sqlite3

    Path(fts_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(fts_db)
    try:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id, source, page, content)")
        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith((".md", ".txt")):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue
            document = loader.load()
            chunks = text_splitter.split_documents(document)
            source_hash = file_hash(file_path)
            rows = []
            for i, chunk in enumerate(chunks):
                page = chunk.metadata.get("page", 0)
                rows.append(
                    (f"{filename}:{source_hash[:12]}:{i}", filename, page, chunk.page_content)
                )
            conn.executemany(
                "INSERT INTO chunks(chunk_id, source, page, content) VALUES (?, ?, ?, ?)", rows
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # 包含文档的文件夹路径
    folder_path = "./data"

    # 创建 Chroma 数据库
    chroma = create_chroma_db(folder_path=folder_path)

    # 从 Chroma 数据库创建检索器
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # 执行相似度搜索
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # 显示结果
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\nResult {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
