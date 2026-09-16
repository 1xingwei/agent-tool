import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.settings import settings

# Load environment variables from the .env file
load_dotenv()


def create_chroma_db(
    folder_path: str,
    db_name: str | None = None,
    delete_chroma_db: bool = False,
    chunk_size: int = 2000,
    overlap: int = 500,
    embeddings: Embeddings | None = None,
):
    db_name = db_name or settings.CHROMA_DIR
    if embeddings is None:
        embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=db_name,
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load document based on file extension
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith((".md", ".txt")):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue  # Skip unsupported file types

        # Load and split document into chunks
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # Add chunks to Chroma vector store
        chroma.add_documents(chunks)
        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = "./data"

    # Create the Chroma database
    chroma = create_chroma_db(folder_path=folder_path)

    # Create retriever from the Chroma database
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
