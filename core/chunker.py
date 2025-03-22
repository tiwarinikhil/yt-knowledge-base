from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_transcripts(transcripts: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for item in transcripts:
        chunks = splitter.split_text(item["transcript"])
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"video_id": item["video_id"], "title": item["title"]},
                )
            )

    return documents
