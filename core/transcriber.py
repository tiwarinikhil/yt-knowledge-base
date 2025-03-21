from pathlib import Path

import whisper
import yt_dlp

from config import DEVICE, TRANSCRIPTS_DIR, WHISPER_MODEL_SIZE


def get_video_ids(url: str) -> list[dict]:
    ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if "entries" in info:
        return [
            {"video_id": e["id"], "title": e["title"], "url": e["url"]}
            for e in info["entries"]
            if e
        ]

    return [{"video_id": info["id"], "title": info["title"], "url": url}]


def download_audio(video_id: str, url: str) -> Path:
    audio_path = TRANSCRIPTS_DIR / f"{video_id}.mp3"
    if audio_path.exists():
        return audio_path

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(TRANSCRIPTS_DIR / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return audio_path


def transcribe_audio(video_id: str, audio_path: Path) -> str:
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8")

    model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
    result = model.transcribe(str(audio_path))
    text: str = result["text"]

    transcript_path.write_text(text, encoding="utf-8")
    return text


def process_url(url: str) -> list[dict]:
    videos = get_video_ids(url)
    results = []

    for video in videos:
        print(f"Processing: {video['title']}")
        audio_path = download_audio(video["video_id"], video["url"])
        transcript = transcribe_audio(video["video_id"], audio_path)
        results.append(
            {
                "video_id": video["video_id"],
                "title": video["title"],
                "transcript": transcript,
            }
        )

    return results
