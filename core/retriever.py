from langchain_core.documents import Document

from core.embedder import load_index


def retrieve(query: str, index_id: str, top_k: int = 5) -> list[Document]:
    vectorstore = load_index(index_id)
    return vectorstore.similarity_search(query, k=top_k)


def format_context(docs: list[Document]) -> str:
    formatted = [f"[{doc.metadata['title']}]\n{doc.page_content}" for doc in docs]
    return "\n\n---\n\n".join(formatted)
