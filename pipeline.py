from urllib.parse import parse_qs, urlparse

from core.chunker import chunk_transcripts
from core.embedder import build_index, index_exists, load_index
from core.llm import answer
from core.retriever import format_context, get_timestamps, retrieve
from core.transcriber import process_url


def get_index_id(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if "list" in params:
        return params["list"][0]

    if "v" in params:
        return params["v"][0]

    return parsed.path.rstrip("/").split("/")[-1]


def ingest(url: str) -> str:
    index_id = get_index_id(url)

    if index_exists(index_id):
        return "Index already exists for this URL. Ready to chat."

    transcripts = process_url(url)
    chunks = chunk_transcripts(transcripts)
    build_index(chunks, index_id)

    return f"Done. Indexed {len(transcripts)} video(s). Ready to chat."


def query(question: str, index_id: str) -> dict:
    docs = retrieve(question, index_id)
    timestamps = get_timestamps(docs)
    context = format_context(docs)
    answer_text = answer(question, context)
    return {"answer": answer_text, "timestamps": timestamps}
