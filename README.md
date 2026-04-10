# YouTube Knowledge Base  

A local RAG-powered Q&A system for YouTube videos and playlists. Paste any
YouTube URL, the app transcribes it with Whisper, and you can chat with the
content using natural language. Answers include clickable timestamps that
link directly to the relevant moments in the video.

---

## How It Works

1. Paste a YouTube video or playlist URL
2. `yt-dlp` downloads the audio
3. Whisper (local) transcribes it to text with segment-level timestamps
4. LangChain splits the transcript into chunks, each tagged with a timestamp
5. HuggingFace sentence-transformers embeds the chunks into a local FAISS index
6. When you ask a question, the top matching chunks are sent to the LLM
7. The LLM answers based only on the video content
8. Timestamps and embedded video players are shown alongside the answer

---

## Features

- Full offline support with Ollama (no API keys needed)
- Switchable to OpenAI with one env variable
- Timestamp-aware answers with `[MM:SS]` links to exact moments
- YouTube iframe embeds in the chat UI — jump to relevant sections instantly
- Caches audio, transcripts, and indexes — reloading the same URL is instant
- Supports both single videos and full playlists

---

## Tech Stack

| Component | Library |
|---|---|
| Audio download | yt-dlp |
| Transcription | OpenAI Whisper (local) |
| Chunking + RAG | LangChain |
| Vector store | FAISS (local) |
| Embeddings | HuggingFace sentence-transformers |
| LLM (local) | Ollama (llama3.2) |
| LLM (cloud) | OpenAI (gpt-4o-mini) |
| UI | Gradio |

---

## Installation

**Prerequisites:** Python 3.10+, ffmpeg in PATH

```bash
git clone https://github.com/tiwarinikhil/yt-knowledge-base.git
cd yt-knowledge-base

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## LLM Backend

The backend is controlled by `LLM_BACKEND` in `.env`. Switch anytime and
restart the app — indexes are shared between both backends.

### Ollama (default, fully local)

```
# .env
LLM_BACKEND=ollama
```

Install Ollama from https://ollama.com/download, then pull the model:

```bash
ollama pull llama3.2
```

### OpenAI

```
# .env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-your-key-here
```

> Embeddings always run locally via HuggingFace — only the Q&A step uses
> the OpenAI API.

---

## Running the App

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## Usage

1. Paste a YouTube video or playlist URL
2. Click **Load Video** and wait for processing
3. Once status shows **Ready to chat**, ask questions in the chat box
4. Each answer shows timestamp links like `[02:35]` — click to jump to that moment
5. YouTube video embeds appear below the chat, starting at the relevant timestamps

Already-processed videos are cached — reloading the same URL is instant.

---

## Project Structure

```
yt-knowledge-base/
├── core/
│   ├── transcriber.py    # yt-dlp + Whisper + segment timestamps
│   ├── chunker.py        # text splitting with timestamp metadata
│   ├── embedder.py       # FAISS index
│   ├── retriever.py      # similarity search + timestamp extraction
│   └── llm.py            # Ollama / OpenAI
├── transcripts/          # cached audio, transcripts, and segment JSONs
├── indexes/              # FAISS indexes
├── pipeline.py           # orchestration
├── app.py                # Gradio UI with timestamp links and embeds
├── config.py             # all settings
└── .env                  # local config (not committed)
```
