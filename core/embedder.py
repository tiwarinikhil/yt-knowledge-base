from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, INDEXES_DIR


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index(chunks: list[Document], index_id: str) -> FAISS:
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vectorstore.save_local(str(INDEXES_DIR / index_id))
    return vectorstore


def load_index(index_id: str) -> FAISS:
    index_path = INDEXES_DIR / index_id
    if not index_path.exists():
        raise FileNotFoundError(f"No FAISS index found at {index_path}")
    return FAISS.load_local(
        str(index_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def index_exists(index_id: str) -> bool:
    return (INDEXES_DIR / index_id).exists()
