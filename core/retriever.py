from langchain_core.documents import Document

from core.embedder import load_index


def _format_time(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def retrieve(query: str, index_id: str, top_k: int = 5) -> list[Document]:
    vectorstore = load_index(index_id)
    return vectorstore.similarity_search(query, k=top_k)


def format_context(docs: list[Document]) -> str:
    formatted = [
        f"[{doc.metadata['title']} | {_format_time(doc.metadata.get('start_time', 0.0))}]\n{doc.page_content}"
        for doc in docs
    ]
    return "\n\n---\n\n".join(formatted)


def get_timestamps(docs: list[Document]) -> list[dict]:
    results: list[dict] = []
    seen_times: list[float] = []

    for doc in docs:
        start_time = doc.metadata.get("start_time", 0.0)
        if any(abs(start_time - t) <= 5 for t in seen_times):
            continue
        seen_times.append(start_time)
        results.append(
            {
                "title": doc.metadata["title"],
                "start_time": start_time,
                "timestamp_label": _format_time(start_time),
                "youtube_url": doc.metadata.get(
                    "url",
                    f"https://www.youtube.com/watch?v={doc.metadata['video_id']}&t={int(start_time)}s",
                ),
            }
        )
        if len(results) == 3:
            break

    return results
