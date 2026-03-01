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
        video_id = item["video_id"]
        title = item["title"]
        segments = item.get("segments", [])
        chunks = splitter.split_text(item["transcript"])

        for chunk in chunks:
            start_time = 0.0
            for seg in segments:
                if seg["text"].strip() in chunk:
                    start_time = seg["start"]
                    break

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "video_id": video_id,
                        "title": title,
                        "start_time": start_time,
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s",
                    },
                )
            )

    return documents
